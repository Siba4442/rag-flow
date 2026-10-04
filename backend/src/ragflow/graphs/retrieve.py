"""Scoped similarity search as a LangGraph graph: ``embed_query -> search -> threshold``.

Backs ``/query`` and ``/query_multiple``, and the chat graph's retriever tool. Query
vectors (never scoped results) are cached, like rag-api's ``get_cached_query_embedding``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from ragflow.stores.base import ScopedVectorStore


class QueryVectorCache:
    """Tiny async-friendly LRU for query embeddings."""

    def __init__(self, embeddings: Embeddings, maxsize: int = 128) -> None:
        self._embeddings = embeddings
        self._maxsize = maxsize
        self._data: OrderedDict[str, list[float]] = OrderedDict()

    async def get(self, query: str) -> list[float]:
        if query in self._data:
            self._data.move_to_end(query)
            return self._data[query]
        vector = await self._embeddings.aembed_query(query)
        self._data[query] = vector
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)
        return vector


@dataclass(frozen=True)
class RetrieveContext:
    store: ScopedVectorStore
    query_cache: QueryVectorCache
    distance_threshold: float | None = None


class RetrieveInput(TypedDict):
    query: str
    file_ids: list[str]
    owners: list[str]
    k: int


class RetrieveState(RetrieveInput, total=False):
    embedding: list[float]
    results: list[tuple[Document, float]]


async def embed_query(state: RetrieveState, runtime: Runtime[RetrieveContext]) -> dict[str, Any]:
    return {"embedding": await runtime.context.query_cache.get(state["query"])}


async def search(state: RetrieveState, runtime: Runtime[RetrieveContext]) -> dict[str, Any]:
    embedding = state.get("embedding")
    if embedding is None:
        raise RuntimeError("search ran before embed_query")
    results = await runtime.context.store.asearch(
        embedding, state["k"], state["file_ids"], state["owners"]
    )
    return {"results": results}


def apply_threshold(state: RetrieveState, runtime: Runtime[RetrieveContext]) -> dict[str, Any]:
    """Drop matches farther than RAG_DISTANCE_THRESHOLD.

    Only for stores whose scores are distances; on similarity scores (Pinecone,
    Atlas) the same comparison would keep the weakest matches, as rag-api notes.
    """
    threshold = runtime.context.distance_threshold
    if threshold is None or not runtime.context.store.scores_are_distances:
        return {}
    return {"results": [(d, s) for d, s in state.get("results", []) if s <= threshold]}


def build_retrieve_graph() -> CompiledStateGraph[
    RetrieveState, RetrieveContext, RetrieveInput, RetrieveState
]:
    graph = StateGraph(RetrieveState, context_schema=RetrieveContext, input_schema=RetrieveInput)
    graph.add_node("embed_query", embed_query)
    graph.add_node("search", search)
    graph.add_node("apply_threshold", apply_threshold)
    graph.add_edge(START, "embed_query")
    graph.add_edge("embed_query", "search")
    graph.add_edge("search", "apply_threshold")
    graph.add_edge("apply_threshold", END)
    return graph.compile(name="retrieve")


RETRIEVE_GRAPH = build_retrieve_graph()


@dataclass
class Retriever:
    """Convenience wrapper used by the API and the chat graph."""

    context: RetrieveContext
    _graph: Any = field(default=RETRIEVE_GRAPH, repr=False)

    async def search(
        self, query: str, file_ids: list[str], owners: list[str], k: int
    ) -> list[tuple[Document, float]]:
        if not file_ids or not owners:
            return []
        state = await self._graph.ainvoke(
            {"query": query, "file_ids": file_ids, "owners": owners, "k": k},
            context=self.context,
        )
        results: list[tuple[Document, float]] = state.get("results", [])
        return results
