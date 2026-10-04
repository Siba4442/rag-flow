"""PGVectorStore backend and Postgres checkpointer against a real database."""

import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

from ragflow.stores.base import chunk_uuid
from ragflow.stores.pgvector import PgVectorScopedStore

pytestmark = pytest.mark.integration

EMB = DeterministicFakeEmbedding(size=16)


def _doc(file_id: str, owner: str, idx: int, attempt: str = "a1") -> Document:
    return Document(
        id=chunk_uuid(owner, file_id, attempt, idx),
        page_content=f"{file_id} chunk {idx} by {owner}",
        metadata={
            "file_id": file_id,
            "user_id": owner,
            "chunk_index": idx,
            "ingestion_attempt_id": attempt,
            "source": f"{file_id}.txt",
        },
    )


@pytest.fixture
async def store(pg_url: str) -> AsyncIterator[PgVectorScopedStore]:
    s = await PgVectorScopedStore.create(pg_url, EMB, table=f"t_{uuid.uuid4().hex[:10]}")
    await s.aadd_documents(
        [_doc("f1", "alice", 0), _doc("f1", "alice", 1), _doc("f2", "bob", 0), _doc("f1", "bob", 0)]
    )
    yield s
    await s.aclose()


async def test_scoped_reads(store: PgVectorScopedStore) -> None:
    assert sorted(await store.get_all_ids(["alice"])) == ["f1"]
    assert await store.get_filtered_ids(["f2"], ["alice"]) == []
    docs = await store.get_documents_by_ids(["f1"], ["alice"])
    assert [d.metadata["chunk_index"] for d in docs] == [0, 1]
    assert {d.metadata["user_id"] for d in docs} == {"alice"}
    assert docs[0].metadata["source"] == "f1.txt"


async def test_search_is_owner_scoped_and_returns_distances(store: PgVectorScopedStore) -> None:
    hits = await store.asearch(await EMB.aembed_query("f1 chunk 0 by bob"), 10, ["f1"], ["alice"])
    assert hits
    assert all(d.metadata["user_id"] == "alice" for d, _ in hits)
    assert all(0.0 <= score <= 2.0 for _, score in hits)


async def test_deletes(store: PgVectorScopedStore) -> None:
    await store.aadd_documents([_doc("f1", "alice", 0, attempt="a2")])
    await store.delete_attempt("f1", "alice", "a2")
    assert len(await store.get_documents_by_ids(["f1"], ["alice"])) == 2
    await store.delete_scoped(["f1"], ["alice"])
    assert await store.get_all_ids(["alice"]) == []
    assert sorted(await store.get_all_ids(["bob"])) == ["f1", "f2"]


async def test_list_files_and_health(store: PgVectorScopedStore) -> None:
    files = {f.file_id: f for f in await store.list_files(["bob"])}
    assert files["f2"].filename == "f2.txt"
    assert files["f2"].chunks == 1
    assert await store.health()


async def test_reopening_existing_table(pg_url: str) -> None:
    table = f"t_{uuid.uuid4().hex[:10]}"
    first = await PgVectorScopedStore.create(pg_url, EMB, table=table)
    await first.aadd_documents([_doc("f9", "carol", 0)])
    await first.aclose()
    second = await PgVectorScopedStore.create(pg_url, EMB, table=table)
    assert await second.get_all_ids(["carol"]) == ["f9"]
    await second.aclose()


async def test_postgres_checkpointer_isolates_threads(pg_url: str) -> None:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from ragflow.graphs.chat import ChatContext, build_chat_graph, turn_input
    from ragflow.graphs.retrieve import QueryVectorCache, RetrieveContext, Retriever
    from ragflow.stores.memory import InMemoryScopedStore
    from tests.unit.graphs.test_chat import ScriptedModel

    dsn = pg_url.replace("postgresql+psycopg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
        await saver.setup()
        graph = build_chat_graph(saver)
        store = InMemoryScopedStore(EMB)
        model = ScriptedModel(messages=iter([AIMessage("hi alice"), AIMessage("hi bob")]))
        ctx = ChatContext(
            retriever=Retriever(RetrieveContext(store, QueryVectorCache(EMB))),
            model=model,
            grader=model,
            owners=["x"],
            file_ids=[],
        )
        cfg_a: Any = {"configurable": {"thread_id": "alice:t"}}
        cfg_b: Any = {"configurable": {"thread_id": "bob:t"}}
        await graph.ainvoke(turn_input("a"), cfg_a, context=ctx)
        await graph.ainvoke(turn_input("b"), cfg_b, context=ctx)
        a = (await graph.aget_state(cfg_a)).values["messages"]
        assert [m.content for m in a] == ["a", "hi alice"]


async def test_high_dimension_embeddings_skip_hnsw(pg_url: str) -> None:
    """e.g. qwen3-embedding-8b (4096 dims) exceeds pgvector's 2000-dim index limit."""
    emb = DeterministicFakeEmbedding(size=4096)
    s = await PgVectorScopedStore.create(pg_url, emb, table=f"t_{uuid.uuid4().hex[:10]}")
    await s.aadd_documents([_doc("f1", "alice", 0)])
    hits = await s.asearch(await emb.aembed_query("x"), 1, ["f1"], ["alice"])
    assert len(hits) == 1
    await s.aclose()
