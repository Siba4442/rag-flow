"""Ingestion as a LangGraph graph.

    load ─▶ split ─┬─▶ store_batch (fan-out via Send, PARALLEL_EXECUTION wide) ─┐
                   └──────────────────────── (no chunks) ──────────────────────┴─▶ finalize

Each batch embeds and writes independently. A failed batch records an error rather than
aborting the superstep, and ``finalize`` then deletes every chunk tagged with this
attempt's id, so a partial upload never lingers (rag-api's per-attempt rollback).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import operator
import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from ragflow.ingest.loaders import clean_text, get_loader
from ragflow.ingest.parsers import parse_pdf
from ragflow.settings import Settings
from ragflow.stores.base import ScopedVectorStore, chunk_uuid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestContext:
    store: ScopedVectorStore
    settings: Settings


class IngestInput(TypedDict):
    file_id: str
    owner: str
    filepath: str
    filename: str
    content_type: str | None


class IngestState(IngestInput, total=False):
    attempt_id: str
    known_type: bool
    file_ext: str
    pages: list[Document]
    chunks: list[Document]
    stored: Annotated[int, operator.add]
    errors: Annotated[list[str], operator.add]
    status: Literal["stored", "empty", "rolled_back"]


class BatchInput(TypedDict):
    batch: list[Document]


def _digest(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()


def _load_sync(state: IngestState, settings: Settings) -> dict[str, Any]:
    loaded = get_loader(
        state["filename"],
        state["content_type"],
        state["filepath"],
        pdf_extract_images=settings.pdf_extract_images,
    )
    pages = None
    if loaded.file_ext == "pdf":
        pages = parse_pdf(state["filepath"], settings)  # None -> default loader
    if pages is None:
        pages = loaded.load()
    else:
        loaded.cleanup()
    return {"pages": pages, "known_type": loaded.known_type, "file_ext": loaded.file_ext}


async def load(state: IngestState, runtime: Runtime[IngestContext]) -> dict[str, Any]:
    result = await asyncio.to_thread(_load_sync, state, runtime.context.settings)
    return {**result, "attempt_id": uuid.uuid4().hex, "stored": 0, "errors": []}


def _split_sync(state: IngestState, settings: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    pieces = splitter.split_documents(state.get("pages", []))
    is_pdf = state.get("file_ext") == "pdf"
    attempt = state["attempt_id"]
    ingested_at = time.time_ns()
    chunks: list[Document] = []
    for index, piece in enumerate(pieces):
        content = clean_text(piece.page_content) if is_pdf else piece.page_content
        if not content.strip():
            continue
        metadata = {
            **(piece.metadata or {}),
            # Ours last: a loader can never overwrite the owner or file id.
            "file_id": state["file_id"],
            "user_id": state["owner"],
            "digest": _digest(content),
            "chunk_index": index,
            "ingestion_attempt_id": attempt,
            "ingested_at_ns": ingested_at,
        }
        chunks.append(
            Document(
                id=chunk_uuid(state["owner"], state["file_id"], attempt, index),
                page_content=content,
                metadata=metadata,
            )
        )
    return chunks


async def split(state: IngestState, runtime: Runtime[IngestContext]) -> dict[str, Any]:
    chunks = await asyncio.to_thread(_split_sync, state, runtime.context.settings)
    return {"chunks": chunks, "pages": []}


def fan_out(state: IngestState, runtime: Runtime[IngestContext]) -> list[Send] | str:
    chunks = state.get("chunks", [])
    if not chunks:
        return "finalize"
    size = runtime.context.settings.embedding_batch_size
    size = size if size > 0 else len(chunks)
    return [
        Send("store_batch", {"batch": chunks[i : i + size]}) for i in range(0, len(chunks), size)
    ]


async def store_batch(state: BatchInput, runtime: Runtime[IngestContext]) -> dict[str, Any]:
    try:
        await runtime.context.store.aadd_documents(state["batch"])
    except Exception as exc:
        logger.exception("Embedding batch failed")
        return {"errors": [f"{type(exc).__name__}: {exc}"]}
    return {"stored": len(state["batch"])}


async def finalize(state: IngestState, runtime: Runtime[IngestContext]) -> dict[str, Any]:
    if state.get("errors"):
        logger.warning("Rolling back attempt %s for file %s", state["attempt_id"], state["file_id"])
        await runtime.context.store.delete_attempt(
            state["file_id"], state["owner"], state["attempt_id"]
        )
        return {"status": "rolled_back", "chunks": []}
    return {"status": "stored" if state.get("chunks") else "empty", "chunks": []}


def build_ingest_graph() -> CompiledStateGraph[
    IngestState, IngestContext, IngestInput, IngestState
]:
    graph = StateGraph(IngestState, context_schema=IngestContext, input_schema=IngestInput)
    graph.add_node("load", load)
    graph.add_node("split", split)
    graph.add_node("store_batch", store_batch)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "load")
    graph.add_edge("load", "split")
    graph.add_conditional_edges("split", fan_out, ["store_batch", "finalize"])
    graph.add_edge("store_batch", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(name="ingest")


INGEST_GRAPH = build_ingest_graph()


async def run_ingest(inputs: IngestInput, context: IngestContext) -> IngestState:
    """Run ingestion; ``PARALLEL_EXECUTION`` caps concurrent embedding batches."""
    result: IngestState = await INGEST_GRAPH.ainvoke(  # type: ignore[assignment]
        inputs,
        context=context,
        config={"max_concurrency": max(1, context.settings.parallel_execution)},
    )
    return result
