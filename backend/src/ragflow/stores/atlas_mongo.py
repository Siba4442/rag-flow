"""MongoDB Atlas Vector Search backend, ported from rag-api's ``AtlasMongoVector``.

Metadata fields are stored top-level on each Mongo document (``file_id``, ``user_id``,
...), so scope predicates are plain Mongo queries. The Atlas search index must declare
``file_id`` and ``user_id`` as filter fields.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragflow.stores.base import FileSummary, ns_to_ms, unique

TEXT_KEY = "text"
_INTERNAL = ("_id", "embedding", TEXT_KEY)


def _scope(ids: Sequence[str] | None, owners: Sequence[str]) -> dict[str, Any]:
    clause: dict[str, Any] = {"user_id": {"$in": unique(owners)}}
    if ids is not None:
        clause["file_id"] = {"$in": unique(ids)}
    return clause


class AtlasMongoScopedStore:
    scores_are_distances = False  # Atlas returns cosine similarity

    def __init__(self, client: Any, store: Any) -> None:
        self._client = client
        self._store = store
        self._collection = store.collection

    @classmethod
    def create(
        cls, embeddings: Embeddings, *, uri: str, collection_name: str, index_name: str
    ) -> AtlasMongoScopedStore:
        from langchain_mongodb import MongoDBAtlasVectorSearch
        from pymongo import MongoClient

        client: MongoClient[dict[str, Any]] = MongoClient(uri)
        database = client.get_default_database()
        store = MongoDBAtlasVectorSearch(
            collection=database[collection_name],
            embedding=embeddings,
            index_name=index_name,
            text_key=TEXT_KEY,
        )
        return cls(client, store)

    async def aadd_documents(self, docs: Sequence[Document]) -> list[str]:
        ids = [str(d.id) for d in docs]
        return list(await asyncio.to_thread(self._store.add_documents, list(docs), ids=ids))

    async def asearch(
        self, embedding: list[float], k: int, file_ids: Sequence[str], owners: Sequence[str]
    ) -> list[tuple[Document, float]]:
        if not file_ids or not owners:
            return []
        results = await asyncio.to_thread(
            self._store._similarity_search_with_score,
            embedding,
            k=k,
            pre_filter=_scope(file_ids, owners),
        )
        out: list[tuple[Document, float]] = []
        for doc, score in results:
            meta = {k_: v for k_, v in doc.metadata.items() if k_ != "_id"}
            out.append((Document(id=doc.id, page_content=doc.page_content, metadata=meta), score))
        return out

    def _find(self, query: dict[str, Any]) -> list[Document]:
        cursor = self._collection.find(query, {"embedding": 0}).sort(
            [("file_id", 1), ("chunk_index", 1)]
        )
        return [
            Document(
                id=str(row["_id"]),
                page_content=row.get(TEXT_KEY, ""),
                metadata={k: v for k, v in row.items() if k not in _INTERNAL},
            )
            for row in cursor
        ]

    async def get_all_ids(self, owners: Sequence[str]) -> list[str]:
        if not owners:
            return []
        return list(
            await asyncio.to_thread(self._collection.distinct, "file_id", _scope(None, owners))
        )

    async def get_filtered_ids(self, ids: Sequence[str], owners: Sequence[str]) -> list[str]:
        if not ids or not owners:
            return []
        return list(
            await asyncio.to_thread(self._collection.distinct, "file_id", _scope(ids, owners))
        )

    async def get_documents_by_ids(
        self, ids: Sequence[str], owners: Sequence[str]
    ) -> list[Document]:
        if not ids or not owners:
            return []
        return await asyncio.to_thread(self._find, _scope(ids, owners))

    async def list_files(self, owners: Sequence[str]) -> list[FileSummary]:
        if not owners:
            return []
        pipeline = [
            {"$match": _scope(None, owners)},
            {
                "$group": {
                    "_id": "$file_id",
                    "source": {"$max": "$source"},
                    "n": {"$sum": 1},
                    "added": {"$max": "$ingested_at_ns"},
                }
            },
            {"$sort": {"_id": 1}},
        ]
        rows = await asyncio.to_thread(lambda: list(self._collection.aggregate(pipeline)))
        return [
            FileSummary(
                str(r["_id"]),
                str(r["source"]).rsplit("/", 1)[-1] if r.get("source") else None,
                int(r["n"]),
                ns_to_ms(r.get("added")),
            )
            for r in rows
        ]

    async def delete_scoped(self, ids: Sequence[str], owners: Sequence[str]) -> None:
        if ids and owners:
            await asyncio.to_thread(self._collection.delete_many, _scope(ids, owners))

    async def delete_attempt(self, file_id: str, owner: str, attempt_id: str) -> None:
        await asyncio.to_thread(
            self._collection.delete_many,
            {"file_id": file_id, "user_id": owner, "ingestion_attempt_id": attempt_id},
        )

    async def health(self) -> bool:
        try:
            await asyncio.to_thread(self._client.admin.command, "ping")
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        await asyncio.to_thread(self._client.close)
