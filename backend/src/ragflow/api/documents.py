"""LibreChat-compatible RAG endpoints (same paths, fields and shapes as rag-api).

Deliberate differences from rag-api, all bug fixes:

* ``GET /health`` answers a real 503 when unhealthy (rag-api sent 200 + a JSON array).
* ``/embed-upload`` and ``/local/embed`` report ``status: false`` when storing fails
  (rag-api reported success because its error dict was truthy).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from pathlib import Path
from typing import Annotated, Any

import aiofiles
import aiofiles.os
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse
from langchain_core.documents import Document
from pydantic import BaseModel

from ragflow.api.deps import Services, get_services
from ragflow.graphs.ingest import IngestContext, IngestState, run_ingest
from ragflow.ingest.loaders import clean_text, get_loader, process_documents
from ragflow.scope import resolve_scope, resolve_write_owner

logger = logging.getLogger(__name__)

router = APIRouter()

ServicesDep = Annotated[Services, Depends(get_services)]

UPLOAD_CHUNK = 64 * 1024
FILE_NOT_FOUND = "The specified file was not found."
PANDOC_NOT_INSTALLED = (
    "Pandoc is not installed on the server. Please contact your administrator for assistance."
)


TMP_DIRNAME = "_tmp"  # private temp uploads; never reachable through /local/embed
STORE_FAILED = "Failed to process/store the file data."


def default_error(err: object = "") -> str:
    return f"Something went wrong :/\n{err or ''}"


def client_error(message: str, exc: BaseException) -> str:
    """Log the real exception; give the client a message plus a reference id only.

    Exception text can carry DSNs, hostnames or file paths, so it never leaves the server.
    """
    ref = uuid.uuid4().hex[:12]
    logger.error("%s [ref %s]", message, ref, exc_info=exc)
    return f"{message} (ref {ref})"


# ---------------------------------------------------------------------------- models
class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict[str, Any]


class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


class QueryRequestBody(BaseModel):
    query: str
    file_id: str
    k: int = 4
    entity_id: str | None = None


class QueryMultipleBody(BaseModel):
    query: str
    file_ids: list[str]
    k: int = 4
    entity_id: str | None = None


# ---------------------------------------------------------------------------- helpers
def validate_file_path(base_dir: Path, file_path: str) -> Path | None:
    """Resolve ``file_path`` under ``base_dir``; None if blank or escaping it."""
    if not file_path or not file_path.strip():
        return None
    try:
        base = base_dir.resolve()
        candidate = (base / file_path).resolve()
        candidate.relative_to(base)
    except (ValueError, OSError):
        return None
    return candidate


def make_unique_temp_path(base_dir: Path, owner: str, filename: str | None) -> Path | None:
    """``{upload_dir}/_tmp/{hash(owner)}/{uuid}{suffix}``.

    Neither the caller-supplied filename nor the owner id is used as a path component
    (only a hash and a sanitised suffix), so neither can steer the write location.
    """
    name = Path(filename or "").name
    if name in ("", ".", ".."):
        return None
    suffix = Path(name).suffix.lower()
    if not suffix[1:].isalnum() or len(suffix) > 16:
        suffix = ""
    owner_dir = hashlib.sha256(owner.encode()).hexdigest()[:16]
    return base_dir.resolve() / TMP_DIRNAME / owner_dir / f"{uuid.uuid4().hex}{suffix}"


def local_embed_path(services: Services, owner: str, filepath: str) -> Path | None:
    """Resolve a ``/local/embed`` path, or None if the caller may not read it.

    Always confined to RAG_UPLOAD_DIR and never inside the private temp dir. With
    ``LOCAL_EMBED_SCOPE=owner`` (default) it must also sit under ``{upload_dir}/{owner}/``.
    """
    base = services.settings.rag_upload_dir.resolve()
    path = validate_file_path(base, filepath)
    if path is None or path.is_relative_to(base / TMP_DIRNAME):
        return None
    if services.settings.local_embed_scope == "owner":
        owner_dir = (base / owner).resolve()
        if owner_dir.parent != base or not path.is_relative_to(owner_dir):
            return None
    return path


async def save_upload(file: UploadFile, dest: Path) -> None:
    await aiofiles.os.makedirs(dest.parent, exist_ok=True)
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await file.read(UPLOAD_CHUNK):
            await out.write(chunk)


async def remove_quietly(path: Path) -> None:
    try:
        await aiofiles.os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Failed to remove temp file %s: %s", path, exc)


async def ingest(
    services: Services, *, file_id: str, owner: str, path: Path, filename: str, ctype: str | None
) -> IngestState:
    return await run_ingest(
        {
            "file_id": file_id,
            "owner": owner,
            "filepath": str(path),
            "filename": filename,
            "content_type": ctype,
        },
        IngestContext(services.store, services.settings),
    )


def ingest_error(state: IngestState) -> str | None:
    """Generic failure text; the batch exception itself is logged by the ingest graph."""
    return STORE_FAILED if state.get("status") == "rolled_back" else None


def order_documents(documents: list[Document]) -> list[Document]:
    """Group by ingestion attempt (oldest first), then chunk order within each attempt.

    Rows without attempt metadata (e.g. migrated legacy rows) keep their received order
    and come first, matching rag-api's ``_order_documents_by_chunk_index``.
    """
    legacy: list[Document] = []
    groups: dict[tuple[int, str], list[Document]] = {}
    for doc in documents:
        meta = doc.metadata or {}
        attempt, started = meta.get("ingestion_attempt_id"), meta.get("ingested_at_ns")
        if (
            not isinstance(attempt, str)
            or not isinstance(started, int)
            or isinstance(started, bool)
        ):
            legacy.append(doc)
            continue
        groups.setdefault((started, attempt), []).append(doc)

    ordered = list(legacy)
    for key in sorted(groups):
        group = groups[key]
        indexes = [d.metadata.get("chunk_index") for d in group]
        valid = all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in indexes)
        if valid and len(set(indexes)) == len(indexes):
            group = sorted(group, key=lambda d: int(d.metadata["chunk_index"]))
        ordered.extend(group)
    return ordered


async def scoped_existing(services: Services, ids: list[str], owners: tuple[str, ...]) -> list[str]:
    return await services.store.get_filtered_ids(ids, owners)


class FileSummaryResponse(BaseModel):
    file_id: str
    filename: str | None
    chunks: int
    added_at: int | None = None


# ---------------------------------------------------------------------------- routes
@router.get("/health")
async def health(services: ServicesDep) -> JSONResponse:
    try:
        healthy = await services.store.health()
    except Exception:
        logger.exception("Health check errored")
        return JSONResponse({"status": "DOWN"}, status_code=503)
    if healthy:
        return JSONResponse({"status": "UP"})
    return JSONResponse({"status": "DOWN"}, status_code=503)


@router.get("/ids")
async def get_all_ids(
    request: Request, services: ServicesDep, entity_id: str | None = None
) -> list[str]:
    scope = resolve_scope(request, entity_id)
    try:
        return list(set(await services.store.get_all_ids(scope.owners)))
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=client_error("Failed to get all IDs", exc)
        ) from exc


class InfoResponse(BaseModel):
    chat_model: str | None
    embeddings_model: str
    embeddings_provider: str
    vector_store: str


@router.get("/info", response_model=InfoResponse)
async def info(services: ServicesDep) -> InfoResponse:
    """Model and store names for the UI's status card (no credentials)."""
    s = services.settings
    chat = s.chat_model.split(":", 1)[-1] if services.chat_graph is not None else None
    return InfoResponse(
        chat_model=chat,
        embeddings_model=s.resolved_embeddings_model,
        embeddings_provider=str(s.embeddings_provider),
        vector_store=str(s.vector_db_type),
    )


@router.get("/files", response_model=list[FileSummaryResponse])
async def list_files(
    request: Request, services: ServicesDep, entity_id: str | None = None
) -> list[FileSummaryResponse]:
    """Files visible to the caller with chunk counts (new; used by the frontend)."""
    scope = resolve_scope(request, entity_id)
    files = await services.store.list_files(scope.owners)
    return [
        FileSummaryResponse(
            file_id=f.file_id, filename=f.filename, chunks=f.chunks, added_at=f.added_at
        )
        for f in files
    ]


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(
    request: Request,
    services: ServicesDep,
    ids: Annotated[list[str], Query(...)],
    entity_id: str | None = None,
) -> list[Document]:
    scope = resolve_scope(request, entity_id)
    try:
        existing = await scoped_existing(services, ids, scope.owners)
        # Out-of-scope reads as absent (404), so this is not an existence oracle.
        if not all(i in existing for i in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")
        documents = await services.store.get_documents_by_ids(ids, scope.owners)
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found for the given IDs")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=client_error("Error getting documents by IDs", exc)
        ) from exc
    return documents


@router.delete("/documents")
async def delete_documents(
    request: Request,
    services: ServicesDep,
    document_ids: Annotated[list[str], Body(...)],
    entity_id: str | None = None,
) -> dict[str, str]:
    scope = resolve_scope(request, entity_id)
    try:
        # Check existence first so a mixed request never half-deletes then 404s.
        existing = await scoped_existing(services, document_ids, scope.owners)
        if not all(i in existing for i in document_ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")
        await services.store.delete_scoped(document_ids, scope.owners)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=client_error("Failed to delete documents", exc)
        ) from exc
    n = len(document_ids)
    return {"message": f"Documents for {n} file{'s' if n > 1 else ''} deleted successfully"}


@router.post("/query")
async def query_embeddings_by_file_id(
    body: QueryRequestBody, request: Request, services: ServicesDep
) -> list[tuple[Document, float]]:
    scope = resolve_scope(request, body.entity_id)
    try:
        return await services.retriever.search(
            body.query, [body.file_id], list(scope.owners), body.k
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=client_error("Error in query embeddings", exc)
        ) from exc


@router.post("/query_multiple")
async def query_embeddings_by_file_ids(
    body: QueryMultipleBody, request: Request, services: ServicesDep
) -> list[tuple[Document, float]]:
    scope = resolve_scope(request, body.entity_id)
    try:
        results = await services.retriever.search(
            body.query, body.file_ids, list(scope.owners), body.k
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=client_error("Error in query multiple embeddings", exc)
        ) from exc
    if not results:
        raise HTTPException(status_code=404, detail="No documents found for the given query")
    return results


@router.post("/local/embed")
async def embed_local_file(
    document: StoreDocument, request: Request, services: ServicesDep, entity_id: str | None = None
) -> dict[str, Any]:
    owner = resolve_write_owner(request, entity_id)
    path = local_embed_path(services, owner, document.filepath)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=FILE_NOT_FOUND)
    try:
        state = await ingest(
            services,
            file_id=document.file_id,
            owner=owner,
            path=path,
            filename=document.filename,
            ctype=document.file_content_type,
        )
    except Exception as exc:
        if "No pandoc was found" in str(exc):
            detail = PANDOC_NOT_INSTALLED
        else:
            detail = default_error(client_error("Error during file processing", exc))
        raise HTTPException(status_code=400, detail=detail) from exc
    error = ingest_error(state)
    response: dict[str, Any] = {
        "status": error is None,
        "file_id": document.file_id,
        "filename": document.filename,
        "known_type": state.get("known_type"),
    }
    if error:
        response["message"] = error
    return response


async def _embed_upload(
    request: Request,
    services: Services,
    *,
    file_id: str,
    upload: UploadFile,
    entity_id: str | None,
) -> dict[str, Any]:
    owner = resolve_write_owner(request, entity_id)
    path = make_unique_temp_path(services.settings.rag_upload_dir, owner, upload.filename)
    if path is None:
        raise HTTPException(status_code=400, detail=default_error("Invalid request"))
    filename = upload.filename or path.name
    try:
        await save_upload(upload, path)
        state = await ingest(
            services,
            file_id=file_id,
            owner=owner,
            path=path,
            filename=filename,
            ctype=upload.content_type,
        )
    except HTTPException:
        raise
    except Exception as exc:
        detail = client_error("Error during file processing", exc)
        raise HTTPException(status_code=400, detail=detail) from exc
    finally:
        await remove_quietly(path)

    error = ingest_error(state)
    return {
        "status": error is None,
        "message": error or "File processed successfully.",
        "file_id": file_id,
        "filename": filename,
        "known_type": state.get("known_type"),
    }


@router.post("/embed")
async def embed_file(
    request: Request,
    services: ServicesDep,
    file_id: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
    entity_id: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    return await _embed_upload(request, services, file_id=file_id, upload=file, entity_id=entity_id)


@router.post("/embed-upload")
async def embed_file_upload(
    request: Request,
    services: ServicesDep,
    file_id: Annotated[str, Form(...)],
    uploaded_file: Annotated[UploadFile, File(...)],
    entity_id: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    return await _embed_upload(
        request, services, file_id=file_id, upload=uploaded_file, entity_id=entity_id
    )


@router.get("/documents/{id}/context")
async def load_document_context(
    request: Request, services: ServicesDep, id: str, entity_id: str | None = None
) -> str:
    scope = resolve_scope(request, entity_id)
    try:
        existing = await scoped_existing(services, [id], scope.owners)
        if id not in existing:
            raise HTTPException(status_code=404, detail="The specified file_id was not found")
        documents = await services.store.get_documents_by_ids([id], scope.owners)
        if not documents:
            raise HTTPException(status_code=404, detail="No document found for the given ID")
        return process_documents(order_documents(documents), services.settings.chunk_overlap)
    except HTTPException:
        raise
    except Exception as exc:
        detail = default_error(client_error("Error loading document context", exc))
        raise HTTPException(status_code=400, detail=detail) from exc


@router.post("/text")
async def extract_text(
    request: Request,
    services: ServicesDep,
    file_id: Annotated[str, Form(...)],
    file: Annotated[UploadFile, File(...)],
    entity_id: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Extract raw text without embedding or touching the store."""
    owner = resolve_write_owner(request, entity_id)
    path = make_unique_temp_path(services.settings.rag_upload_dir, owner, file.filename)
    if path is None:
        raise HTTPException(status_code=400, detail=default_error("Invalid request"))
    filename = file.filename or path.name
    try:
        await save_upload(file, path)
        loaded = get_loader(
            filename,
            file.content_type,
            str(path),
            raw_text=True,
            pdf_extract_images=services.settings.pdf_extract_images,
        )
        docs = await asyncio.to_thread(loaded.load)
        is_pdf = loaded.file_ext == "pdf"
        text = "\n".join(clean_text(d.page_content) if is_pdf else d.page_content for d in docs)
    except HTTPException:
        raise
    except Exception as exc:
        detail = client_error("Error during text extraction", exc)
        raise HTTPException(status_code=400, detail=detail) from exc
    finally:
        await remove_quietly(path)
    return {"text": text, "file_id": file_id, "filename": filename, "known_type": loaded.known_type}
