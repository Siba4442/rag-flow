"""Legacy rag-api pgvector rows migrate into rag_chunks without re-embedding."""

import importlib.util
import json
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from ragflow.settings import get_settings
from ragflow.stores.pgvector import PgVectorScopedStore

pytestmark = pytest.mark.integration

SCRIPT = Path(__file__).parents[2] / "scripts" / "migrate_legacy_pgvector.py"

LEGACY_DDL = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS langchain_pg_collection (
        uuid uuid PRIMARY KEY, name varchar, cmetadata json)""",
    """CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
        uuid uuid PRIMARY KEY, collection_id uuid REFERENCES langchain_pg_collection(uuid),
        embedding vector(3), document varchar, cmetadata jsonb, custom_id varchar)""",
]


async def test_migrates_owned_rows(pg_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from sqlalchemy.engine import make_url

    url = make_url(pg_url)
    for key, value in {
        "POSTGRES_USER": url.username,
        "POSTGRES_PASSWORD": url.password,
        "DB_HOST": url.host,
        "DB_PORT": str(url.port),
        "POSTGRES_DB": url.database,
        "PGVECTOR_TABLE": f"mig_{uuid.uuid4().hex[:8]}",
    }.items():
        monkeypatch.setenv(key, str(value))
    get_settings.cache_clear()

    engine = create_async_engine(pg_url)
    coll = uuid.uuid4()
    async with engine.begin() as conn:
        for ddl in LEGACY_DDL:
            await conn.execute(text(ddl))
        await conn.execute(
            text("INSERT INTO langchain_pg_collection VALUES (:u, 'legacy', '{}')"), {"u": coll}
        )
        rows = [
            (
                {
                    "file_id": "f1",
                    "user_id": "alice",
                    "page": 0,
                    "source": "/up/a.pdf",
                    "_rag_chunk_index": 0,
                    "_rag_ingestion_attempt_id": "x",
                    "_rag_ingestion_attempt_started_at_ns": 5,
                },
                "[1,0,0]",
            ),
            ({"file_id": "f1", "user_id": "alice", "_rag_chunk_index": 1}, "[0,1,0]"),
            ({"file_id": "f2"}, "[0,0,1]"),  # ownerless: skipped
        ]
        for meta, vec in rows:
            await conn.execute(
                text(
                    "INSERT INTO langchain_pg_embedding VALUES "
                    "(:u, :c, CAST(:v AS vector), 'text', CAST(:m AS jsonb), :f)"
                ),
                {
                    "u": uuid.uuid4(),
                    "c": coll,
                    "v": vec,
                    "m": json.dumps(meta),
                    "f": meta["file_id"],
                },
            )
    await engine.dispose()

    spec = importlib.util.spec_from_file_location("migrate", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert await module.migrate("legacy", dry_run=False) == 2
    assert await module.migrate("legacy", dry_run=False) == 0  # idempotent

    from langchain_core.embeddings import DeterministicFakeEmbedding

    settings = get_settings()
    store = await PgVectorScopedStore.create(
        pg_url, DeterministicFakeEmbedding(size=3), table=settings.pgvector_table
    )
    docs = await store.get_documents_by_ids(["f1"], ["alice"])
    assert [d.metadata["chunk_index"] for d in docs] == [0, 1]
    assert docs[0].metadata["ingested_at_ns"] == 5
    assert docs[0].metadata["source"] == "/up/a.pdf"
    assert await store.get_all_ids(["public"]) == []
    hits = await store.asearch([1.0, 0.0, 0.0], 1, ["f1"], ["alice"])
    assert hits[0][0].metadata["chunk_index"] == 0  # vectors were copied, not recomputed
    await store.aclose()
    get_settings.cache_clear()
