"""Pinecone backend (``langchain-pinecone``).

Serverless indexes cannot delete or enumerate by metadata filter, so vector ids carry
the scope as a prefix: ``{owner}#{file_id}#{attempt}#{chunk_index}`` (components
percent-encoded). Listing an owner's or a file's chunks is then a ``list(prefix=...)``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import Any
from urllib.parse import quote

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragflow.stores.base import FileSummary, summarize, unique

TEXT_KEY = "text"
FETCH_BATCH = 100


def _q(value: str) -> str:
    return quote(value, safe="")


def vector_id(doc: Document) -> str:
    m = doc.metadata
    return "#".join(
        (
            _q(str(m["user_id"])),
            _q(str(m["file_id"])),
            _q(str(m["ingestion_attempt_id"])),
            str(m["chunk_index"]),
        )
    )


class PineconeScopedStore:
    scores_are_distances = False  # Pinecone returns cosine similarity

    def __init__(self, index: Any, store: Any, namespace: str | None) -> None:
        self._index = index
        self._store = store
        self._ns = namespace

    @classmethod
    def create(
        cls,
        embeddings: Embeddings,
        *,
        api_key: str | None,
        index_name: str,
        namespace: str | None,
    ) -> PineconeScopedStore:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone

        if not api_key:
            raise ValueError("PINECONE_API_KEY is required when VECTOR_DB_TYPE=pinecone")
        index = Pinecone(api_key=api_key).Index(index_name)
        store = PineconeVectorStore(
            index=index, embedding=embeddings, text_key=TEXT_KEY, namespace=namespace
        )
        return cls(index, store, namespace)

    # --------------------------------------------------------------- id helpers
    def _list_ids(self, prefix: str) -> list[str]:
        pages: Iterable[list[str]] = self._index.list(prefix=prefix, namespace=self._ns)
        return [vid for page in pages for vid in page]

    def _ids_for(self, file_ids: Sequence[str] | None, owners: Sequence[str]) -> list[str]:
        prefixes = (
            [f"{_q(o)}#{_q(f)}#" for o in unique(owners) for f in unique(file_ids)]
            if file_ids is not None
            else [f"{_q(o)}#" for o in unique(owners)]
        )
        return [vid for p in prefixes for vid in self._list_ids(p)]

    def _fetch(self, ids: list[str]) -> list[Document]:
        docs: list[Document] = []
        for start in range(0, len(ids), FETCH_BATCH):
            res = self._index.fetch(ids=ids[start : start + FETCH_BATCH], namespace=self._ns)
            for vid, vec in res.vectors.items():
                meta = dict(vec.metadata or {})
                text = str(meta.pop(TEXT_KEY, ""))
                docs.append(Document(id=vid, page_content=text, metadata=meta))
        docs.sort(key=lambda d: (str(d.metadata.get("file_id")), d.metadata.get("chunk_index", 0)))
        return docs

    def _delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), 1000):
            self._index.delete(ids=ids[start : start + 1000], namespace=self._ns)

    # --------------------------------------------------------------- protocol
    async def aadd_documents(self, docs: Sequence[Document]) -> list[str]:
        ids = [vector_id(d) for d in docs]
        return list(await asyncio.to_thread(self._store.add_documents, list(docs), ids=ids))

    async def asearch(
        self, embedding: list[float], k: int, file_ids: Sequence[str], owners: Sequence[str]
    ) -> list[tuple[Document, float]]:
        if not file_ids or not owners:
            return []
        flt = {"file_id": {"$in": unique(file_ids)}, "user_id": {"$in": unique(owners)}}
        result = await asyncio.to_thread(
            self._store.similarity_search_by_vector_with_score, embedding, k=k, filter=flt
        )
        return list(result)

    async def get_all_ids(self, owners: Sequence[str]) -> list[str]:
        if not owners:
            return []
        ids = await asyncio.to_thread(self._ids_for, None, owners)
        return unique([_unq_file(vid) for vid in ids])

    async def get_filtered_ids(self, ids: Sequence[str], owners: Sequence[str]) -> list[str]:
        if not ids or not owners:
            return []
        found = await asyncio.to_thread(self._ids_for, ids, owners)
        return unique([_unq_file(vid) for vid in found])

    async def get_documents_by_ids(
        self, ids: Sequence[str], owners: Sequence[str]
    ) -> list[Document]:
        if not ids or not owners:
            return []
        vids = await asyncio.to_thread(self._ids_for, ids, owners)
        return await asyncio.to_thread(self._fetch, vids)

    async def list_files(self, owners: Sequence[str]) -> list[FileSummary]:
        if not owners:
            return []
        vids = await asyncio.to_thread(self._ids_for, None, owners)
        return summarize(await asyncio.to_thread(self._fetch, vids))

    async def delete_scoped(self, ids: Sequence[str], owners: Sequence[str]) -> None:
        if ids and owners:
            vids = await asyncio.to_thread(self._ids_for, ids, owners)
            await asyncio.to_thread(self._delete, vids)

    async def delete_attempt(self, file_id: str, owner: str, attempt_id: str) -> None:
        prefix = f"{_q(owner)}#{_q(file_id)}#{_q(attempt_id)}#"
        vids = await asyncio.to_thread(self._list_ids, prefix)
        await asyncio.to_thread(self._delete, vids)

    async def health(self) -> bool:
        try:
            await asyncio.to_thread(self._index.describe_index_stats)
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        return None


def _unq_file(vector_id_: str) -> str:
    from urllib.parse import unquote

    return unquote(vector_id_.split("#")[1])
