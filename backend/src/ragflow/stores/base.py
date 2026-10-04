"""Backend-neutral, owner-scoped vector store contract.

Every read and delete takes ``owners`` as a required argument: a caller-supplied file
id is never an authorization (same invariant as rag-api's ``ExtendedPgVector``).

Chunk metadata contract written by the ingest graph:

* ``file_id``, ``user_id`` (owner), ``chunk_index``, ``ingestion_attempt_id``
* ``digest`` (md5 of content) plus loader metadata (``source``, ``page``, ...)
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from langchain_core.documents import Document

CHUNK_ID_NAMESPACE = uuid.UUID("6f1c7b3e-2d4a-4d8e-9a51-0e7c2b9f4a10")


@dataclass(frozen=True)
class FileSummary:
    file_id: str
    filename: str | None
    chunks: int
    #: Latest ingestion time in epoch milliseconds (None for legacy rows).
    added_at: int | None = None


def ns_to_ms(value: object) -> int | None:
    """``ingested_at_ns`` metadata (int or numeric string) to epoch ms."""
    try:
        return int(str(value)) // 1_000_000 if value not in (None, "") else None
    except ValueError:
        return None


def chunk_uuid(owner: str, file_id: str, attempt_id: str, chunk_index: int) -> str:
    """Deterministic chunk id, unique per (owner, file, ingestion attempt, chunk).

    File ids are caller-supplied and not unique across owners, so the owner is part
    of the key; the attempt id makes re-uploads append rather than overwrite.
    """
    key = "\x1f".join((owner, file_id, attempt_id, str(chunk_index)))
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, key))


def unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


@runtime_checkable
class ScopedVectorStore(Protocol):
    #: Whether scores are distances (lower is better) so RAG_DISTANCE_THRESHOLD applies.
    scores_are_distances: bool

    async def aadd_documents(self, docs: Sequence[Document]) -> list[str]: ...

    async def asearch(
        self,
        embedding: list[float],
        k: int,
        file_ids: Sequence[str],
        owners: Sequence[str],
    ) -> list[tuple[Document, float]]: ...

    async def get_all_ids(self, owners: Sequence[str]) -> list[str]: ...

    async def get_filtered_ids(self, ids: Sequence[str], owners: Sequence[str]) -> list[str]: ...

    async def get_documents_by_ids(
        self, ids: Sequence[str], owners: Sequence[str]
    ) -> list[Document]: ...

    async def list_files(self, owners: Sequence[str]) -> list[FileSummary]: ...

    async def delete_scoped(self, ids: Sequence[str], owners: Sequence[str]) -> None: ...

    async def delete_attempt(self, file_id: str, owner: str, attempt_id: str) -> None:
        """Remove the chunks of one ingestion attempt (rollback)."""
        ...

    async def health(self) -> bool: ...

    async def aclose(self) -> None: ...


def summarize(docs: Sequence[Document]) -> list[FileSummary]:
    """Group chunks into per-file summaries (for stores without GROUP BY)."""
    counts: dict[str, int] = {}
    names: dict[str, str | None] = {}
    added: dict[str, int | None] = {}
    for doc in docs:
        fid = str(doc.metadata.get("file_id"))
        counts[fid] = counts.get(fid, 0) + 1
        source = doc.metadata.get("source")
        names.setdefault(fid, str(source).rsplit("/", 1)[-1] if source else None)
        ms = ns_to_ms(doc.metadata.get("ingested_at_ns"))
        if ms is not None and (added.get(fid) is None or ms > (added[fid] or 0)):
            added[fid] = ms
    return [FileSummary(fid, names[fid], n, added.get(fid)) for fid, n in counts.items()]
