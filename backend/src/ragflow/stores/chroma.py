"""Chroma backend (``langchain-chroma``). The Chroma client is sync, so calls run in a thread."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragflow.stores.base import FileSummary, summarize, unique


def _where(ids: Sequence[str] | None, owners: Sequence[str]) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [{"user_id": {"$in": unique(owners)}}]
    if ids is not None:
        clauses.insert(0, {"file_id": {"$in": unique(ids)}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaScopedStore:
    scores_are_distances = True  # collection uses cosine space

    def __init__(self, store: Any) -> None:
        self._store = store

    @classmethod
    def create(
        cls,
        embeddings: Embeddings,
        *,
        collection_name: str,
        host: str | None = None,
        port: int = 8000,
        persist_dir: Path | None = None,
    ) -> ChromaScopedStore:
        import chromadb
        from langchain_chroma import Chroma

        client = (
            chromadb.HttpClient(host=host, port=port)
            if host
            else chromadb.PersistentClient(path=str(persist_dir or "./chroma"))
        )
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=client,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return cls(store)

    def _get(self, where: dict[str, Any]) -> list[Document]:
        res = self._store.get(where=where, include=["metadatas", "documents"])
        return [
            Document(id=i, page_content=text or "", metadata=dict(meta or {}))
            for i, text, meta in zip(res["ids"], res["documents"], res["metadatas"], strict=True)
        ]

    async def aadd_documents(self, docs: Sequence[Document]) -> list[str]:
        ids = [str(d.id) for d in docs]
        return list(await asyncio.to_thread(self._store.add_documents, list(docs), ids=ids))

    async def asearch(
        self, embedding: list[float], k: int, file_ids: Sequence[str], owners: Sequence[str]
    ) -> list[tuple[Document, float]]:
        if not file_ids or not owners:
            return []
        result = await asyncio.to_thread(
            self._store.similarity_search_by_vector_with_relevance_scores,
            embedding,
            k=k,
            filter=_where(file_ids, owners),
        )
        return list(result)

    async def get_all_ids(self, owners: Sequence[str]) -> list[str]:
        if not owners:
            return []
        docs = await asyncio.to_thread(self._get, _where(None, owners))
        return unique([str(d.metadata["file_id"]) for d in docs])

    async def get_filtered_ids(self, ids: Sequence[str], owners: Sequence[str]) -> list[str]:
        if not ids or not owners:
            return []
        docs = await asyncio.to_thread(self._get, _where(ids, owners))
        return unique([str(d.metadata["file_id"]) for d in docs])

    async def get_documents_by_ids(
        self, ids: Sequence[str], owners: Sequence[str]
    ) -> list[Document]:
        if not ids or not owners:
            return []
        return await asyncio.to_thread(self._get, _where(ids, owners))

    async def list_files(self, owners: Sequence[str]) -> list[FileSummary]:
        if not owners:
            return []
        return summarize(await asyncio.to_thread(self._get, _where(None, owners)))

    async def delete_scoped(self, ids: Sequence[str], owners: Sequence[str]) -> None:
        if ids and owners:
            await asyncio.to_thread(self._store.delete, where=_where(ids, owners))

    async def delete_attempt(self, file_id: str, owner: str, attempt_id: str) -> None:
        where = {
            "$and": [
                {"file_id": {"$eq": file_id}},
                {"user_id": {"$eq": owner}},
                {"ingestion_attempt_id": {"$eq": attempt_id}},
            ]
        }
        await asyncio.to_thread(self._store.delete, where=where)

    async def health(self) -> bool:
        try:
            await asyncio.to_thread(self._store._client.heartbeat)
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        return None
