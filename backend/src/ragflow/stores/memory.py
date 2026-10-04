"""In-process store for tests and for trying the app without a database.

Uses cosine distance so ``RAG_DISTANCE_THRESHOLD`` behaves as with pgvector.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragflow.stores.base import FileSummary, summarize, unique


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


class InMemoryScopedStore:
    scores_are_distances = True

    def __init__(self, embeddings: Embeddings) -> None:
        self._embeddings = embeddings
        self._rows: dict[str, tuple[Document, list[float]]] = {}

    async def aadd_documents(self, docs: Sequence[Document]) -> list[str]:
        vectors = await self._embeddings.aembed_documents([d.page_content for d in docs])
        ids: list[str] = []
        for doc, vec in zip(docs, vectors, strict=True):
            if doc.id is None:
                raise ValueError("documents must carry an id")
            self._rows[doc.id] = (doc, vec)
            ids.append(doc.id)
        return ids

    def _scoped(self, ids: Sequence[str], owners: Sequence[str]) -> list[Document]:
        wanted, allowed = set(ids), set(owners)
        return [
            doc
            for doc, _ in self._rows.values()
            if doc.metadata.get("file_id") in wanted and doc.metadata.get("user_id") in allowed
        ]

    async def asearch(
        self, embedding: list[float], k: int, file_ids: Sequence[str], owners: Sequence[str]
    ) -> list[tuple[Document, float]]:
        wanted, allowed = set(file_ids), set(owners)
        scored = [
            (doc, _cosine_distance(embedding, vec))
            for doc, vec in self._rows.values()
            if doc.metadata.get("file_id") in wanted and doc.metadata.get("user_id") in allowed
        ]
        scored.sort(key=lambda pair: pair[1])
        return scored[:k]

    async def get_all_ids(self, owners: Sequence[str]) -> list[str]:
        allowed = set(owners)
        return unique(
            [
                str(doc.metadata["file_id"])
                for doc, _ in self._rows.values()
                if doc.metadata.get("user_id") in allowed
            ]
        )

    async def get_filtered_ids(self, ids: Sequence[str], owners: Sequence[str]) -> list[str]:
        return unique([str(d.metadata["file_id"]) for d in self._scoped(ids, owners)])

    async def get_documents_by_ids(
        self, ids: Sequence[str], owners: Sequence[str]
    ) -> list[Document]:
        return self._scoped(ids, owners)

    async def list_files(self, owners: Sequence[str]) -> list[FileSummary]:
        allowed = set(owners)
        return summarize(
            [doc for doc, _ in self._rows.values() if doc.metadata.get("user_id") in allowed]
        )

    async def delete_scoped(self, ids: Sequence[str], owners: Sequence[str]) -> None:
        doomed = {d.id for d in self._scoped(ids, owners)}
        for key in doomed:
            if key is not None:
                self._rows.pop(key, None)

    async def delete_attempt(self, file_id: str, owner: str, attempt_id: str) -> None:
        for key, (doc, _) in list(self._rows.items()):
            meta = doc.metadata
            if (
                meta.get("file_id") == file_id
                and meta.get("user_id") == owner
                and meta.get("ingestion_attempt_id") == attempt_id
            ):
                del self._rows[key]

    async def health(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None
