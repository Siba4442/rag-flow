"""Copy chunks from rag-api's legacy pgvector tables into RAG Flow's ``rag_chunks`` table.

rag-api (langchain_community PGVector) keeps every collection in
``langchain_pg_embedding`` with owner and file id inside the JSON ``cmetadata``. RAG
Flow stores them as indexed columns. This script copies rows across **without
re-embedding**: the stored vectors are reused, so keep the same embeddings model.

    uv run python scripts/migrate_legacy_pgvector.py --dry-run
    uv run python scripts/migrate_legacy_pgvector.py --collection testcollection

Rows without a ``user_id`` are skipped: rag-api already treated them as out of scope
for every caller. Re-running is safe (existing ids are left alone). The legacy tables
are never modified.
"""

from __future__ import annotations

import argparse
import asyncio

from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ragflow.settings import get_settings
from ragflow.stores.pgvector import PgVectorScopedStore

LEGACY_ROWS = """
    FROM langchain_pg_embedding e
    JOIN langchain_pg_collection c ON c.uuid = e.collection_id
    WHERE c.name = :collection
"""

COPY_SQL = (
    """
INSERT INTO {target} (langchain_id, content, embedding, file_id, user_id, chunk_index,
                      ingestion_attempt_id, langchain_metadata)
SELECT
    e.uuid,
    e.document,
    e.embedding,
    COALESCE(e.cmetadata::jsonb ->> 'file_id', e.custom_id),
    e.cmetadata::jsonb ->> 'user_id',
    NULLIF(e.cmetadata::jsonb ->> '_rag_chunk_index', '')::int,
    e.cmetadata::jsonb ->> '_rag_ingestion_attempt_id',
    (
        (e.cmetadata::jsonb
            - 'file_id' - 'user_id' - '_rag_chunk_index' - '_rag_ingestion_attempt_id'
            - '_rag_ingestion_attempt_started_at_ns')
        || jsonb_strip_nulls(jsonb_build_object(
            'ingested_at_ns',
            NULLIF(e.cmetadata::jsonb ->> '_rag_ingestion_attempt_started_at_ns', '')::bigint,
            'legacy', true
        ))
    )::json
"""
    + LEGACY_ROWS
    + """
    AND e.cmetadata::jsonb ->> 'user_id' IS NOT NULL
ON CONFLICT (langchain_id) DO NOTHING
"""
)


async def migrate(collection: str, dry_run: bool) -> int:
    settings = get_settings()
    schema = (settings.postgres_schema or "public").split(",")[0].strip()
    engine = create_async_engine(settings.sqlalchemy_url)
    params = {"collection": collection}
    async with engine.connect() as conn:
        total = (await conn.execute(text("SELECT count(*)" + LEGACY_ROWS), params)).scalar_one()
        ownerless = (
            await conn.execute(
                text(
                    "SELECT count(*)"
                    + LEGACY_ROWS
                    + " AND e.cmetadata::jsonb ->> 'user_id' IS NULL"
                ),
                params,
            )
        ).scalar_one()
        dims = (
            await conn.execute(
                text("SELECT vector_dims(e.embedding)" + LEGACY_ROWS + " LIMIT 1"), params
            )
        ).scalar()
    print(f"collection {collection!r}: {total} rows, {ownerless} without owner (skipped)")
    if not total or dims is None:
        await engine.dispose()
        return 0
    if dry_run:
        print(
            f"dry run: would copy {total - ownerless} rows of vector({dims}) "
            f"into {schema}.{settings.pgvector_table}"
        )
        await engine.dispose()
        return 0

    store = await PgVectorScopedStore.create(
        settings.sqlalchemy_url,
        # Vectors are copied, never computed: no provider credentials needed here.
        DeterministicFakeEmbedding(size=int(dims)),
        table=settings.pgvector_table,
        schema=schema,
        vector_size=int(dims),
    )
    await store.aclose()
    target = f'"{schema}"."{settings.pgvector_table}"'
    async with engine.begin() as conn:
        result = await conn.execute(text(COPY_SQL.format(target=target)), params)
    print(f"copied {result.rowcount} rows into {target}")
    await engine.dispose()
    return int(result.rowcount)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--collection", default=get_settings().collection_name)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(migrate(args.collection, args.dry_run))


if __name__ == "__main__":
    main()
