"""pgvector backend on ``langchain_postgres.PGVectorStore`` (native async, psycopg3).

Unlike rag-api's legacy ``langchain_pg_embedding`` table (owner and file id inside a
JSON blob), ownership and file id are real indexed columns here, so the scope
predicate is a plain btree lookup. See ``scripts/migrate_legacy_pgvector.py``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex
from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ragflow.stores.base import FileSummary, unique

logger = logging.getLogger(__name__)

METADATA_COLUMNS = ("file_id", "user_id", "chunk_index", "ingestion_attempt_id")
#: pgvector's HNSW / IVFFlat indexes support at most 2000 dimensions on ``vector``.
MAX_INDEXED_DIMENSIONS = 2000


class PgVectorScopedStore:
    scores_are_distances = True  # cosine distance, same as rag-api

    def __init__(self, engine: AsyncEngine, store: PGVectorStore, table: str, schema: str) -> None:
        self._engine = engine
        self._store = store
        self._qualified = f'"{schema}"."{table}"'

    @classmethod
    async def create(
        cls,
        url: str,
        embeddings: Embeddings,
        *,
        table: str = "rag_chunks",
        schema: str = "public",
        vector_size: int | None = None,
    ) -> PgVectorScopedStore:
        """Open (creating if needed) the chunk table.

        ``vector_size`` skips probing the embeddings provider for the dimension, e.g. when
        migrating legacy rows whose vectors are copied rather than re-embedded.
        """
        engine = create_async_engine(url, pool_pre_ping=True)
        # from_engine with no loop: run on the caller's (uvicorn's) event loop
        # instead of PGEngine's private background thread.
        pg_engine = PGEngine.from_engine(engine)

        if not await _table_exists(engine, schema, table):
            if vector_size is None:
                vector_size = len(await embeddings.aembed_query("dimension probe"))
            logger.info("Creating %s.%s (vector(%d))", schema, table, vector_size)
            await pg_engine.ainit_vectorstore_table(
                table_name=table,
                vector_size=vector_size,
                schema_name=schema,
                metadata_columns=[
                    Column("file_id", "TEXT", nullable=False),
                    Column("user_id", "TEXT", nullable=False),
                    Column("chunk_index", "INTEGER"),
                    Column("ingestion_attempt_id", "TEXT"),
                ],
            )
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS "{table}_owner_file_idx" '
                        f'ON "{schema}"."{table}" (user_id, file_id)'
                    )
                )
                await conn.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS "{table}_attempt_idx" '
                        f'ON "{schema}"."{table}" (file_id, ingestion_attempt_id)'
                    )
                )
            created = True
        else:
            created = False

        store = await PGVectorStore.create(
            engine=pg_engine,
            embedding_service=embeddings,
            table_name=table,
            schema_name=schema,
            metadata_columns=list(METADATA_COLUMNS),
        )
        if created and vector_size is not None and vector_size <= MAX_INDEXED_DIMENSIONS:
            await store.aapply_vector_index(HNSWIndex(name=f"{table}_hnsw_idx"))
        elif created:
            # Every search is pre-filtered by (user_id, file_id) via the btree index, so
            # exact distance over the caller's own files stays fast without HNSW.
            logger.warning(
                "Embeddings have %s dimensions (> %d): skipping the HNSW index; searches "
                "use exact distance within the selected files. Set EMBEDDINGS_DIMENSIONS "
                "<= %d if your model supports it to enable the index.",
                vector_size,
                MAX_INDEXED_DIMENSIONS,
                MAX_INDEXED_DIMENSIONS,
            )
        return cls(engine, store, table, schema)

    # ------------------------------------------------------------------ writes
    async def aadd_documents(self, docs: Sequence[Document]) -> list[str]:
        return await self._store.aadd_documents(list(docs), ids=[d.id for d in docs])

    async def delete_scoped(self, ids: Sequence[str], owners: Sequence[str]) -> None:
        wanted, allowed = unique(ids), unique(owners)
        if not wanted or not allowed:
            return
        await self._execute(
            f"DELETE FROM {self._qualified} WHERE file_id IN :ids AND user_id IN :owners",
            ids=wanted,
            owners=allowed,
        )

    async def delete_attempt(self, file_id: str, owner: str, attempt_id: str) -> None:
        await self._execute(
            f"DELETE FROM {self._qualified} "
            "WHERE file_id = :file_id AND user_id = :owner AND ingestion_attempt_id = :attempt",
            file_id=file_id,
            owner=owner,
            attempt=attempt_id,
        )

    # ------------------------------------------------------------------ reads
    async def asearch(
        self, embedding: list[float], k: int, file_ids: Sequence[str], owners: Sequence[str]
    ) -> list[tuple[Document, float]]:
        if not file_ids or not owners:
            return []
        flt = {
            "$and": [
                {"file_id": {"$in": unique(file_ids)}},
                {"user_id": {"$in": unique(owners)}},
            ]
        }
        return await self._store.asimilarity_search_with_score_by_vector(embedding, k=k, filter=flt)

    async def get_all_ids(self, owners: Sequence[str]) -> list[str]:
        if not owners:
            return []
        rows = await self._fetch(
            f"SELECT DISTINCT file_id FROM {self._qualified} WHERE user_id IN :owners",
            owners=unique(owners),
        )
        return [r[0] for r in rows]

    async def get_filtered_ids(self, ids: Sequence[str], owners: Sequence[str]) -> list[str]:
        wanted, allowed = unique(ids), unique(owners)
        if not wanted or not allowed:
            return []
        rows = await self._fetch(
            f"SELECT DISTINCT file_id FROM {self._qualified} "
            "WHERE file_id IN :ids AND user_id IN :owners",
            ids=wanted,
            owners=allowed,
        )
        return [r[0] for r in rows]

    async def get_documents_by_ids(
        self, ids: Sequence[str], owners: Sequence[str]
    ) -> list[Document]:
        wanted, allowed = unique(ids), unique(owners)
        if not wanted or not allowed:
            return []
        rows = await self._fetch(
            "SELECT langchain_id, content, langchain_metadata, file_id, user_id, chunk_index, "
            f"ingestion_attempt_id FROM {self._qualified} "
            "WHERE file_id IN :ids AND user_id IN :owners ORDER BY file_id, chunk_index",
            ids=wanted,
            owners=allowed,
        )
        docs: list[Document] = []
        for row in rows:
            raw = row[2]
            metadata: dict[str, Any] = dict(json.loads(raw) if isinstance(raw, str) else raw or {})
            metadata.update(dict(zip(METADATA_COLUMNS, row[3:], strict=True)))
            docs.append(Document(id=str(row[0]), page_content=row[1], metadata=metadata))
        return docs

    async def list_files(self, owners: Sequence[str]) -> list[FileSummary]:
        if not owners:
            return []
        rows = await self._fetch(
            "SELECT file_id, max(langchain_metadata->>'source'), count(*) "
            f"FROM {self._qualified} WHERE user_id IN :owners GROUP BY file_id ORDER BY file_id",
            owners=unique(owners),
        )
        return [
            FileSummary(r[0], r[1].rsplit("/", 1)[-1] if r[1] else None, int(r[2])) for r in rows
        ]

    async def health(self) -> bool:
        try:
            await self._fetch("SELECT 1")
        except Exception:
            logger.exception("pgvector health check failed")
            return False
        return True

    async def aclose(self) -> None:
        await self._engine.dispose()

    # ------------------------------------------------------------------ helpers
    def _stmt(self, sql: str, params: dict[str, Any]) -> Any:
        stmt = text(sql)
        expanding = [k for k, v in params.items() if isinstance(v, list)]
        if expanding:
            stmt = stmt.bindparams(*(bindparam(k, expanding=True) for k in expanding))
        return stmt

    async def _fetch(self, sql: str, **params: Any) -> list[Any]:
        async with self._engine.connect() as conn:
            result = await conn.execute(self._stmt(sql, params), params)
            return list(result.fetchall())

    async def _execute(self, sql: str, **params: Any) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(self._stmt(sql, params), params)


async def _table_exists(engine: AsyncEngine, schema: str, table: str) -> bool:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = :schema AND table_name = :table"
            ),
            {"schema": schema, "table": table},
        )
        return result.first() is not None
