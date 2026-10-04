"""One behavioural suite run against every store that works without a server."""

import uuid
from collections.abc import AsyncIterator

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragflow.stores.base import ScopedVectorStore, chunk_uuid
from ragflow.stores.chroma import ChromaScopedStore
from ragflow.stores.memory import InMemoryScopedStore

EMB = DeterministicFakeEmbedding(size=16)


def _doc(file_id: str, owner: str, idx: int, attempt: str = "a1", text: str = "") -> Document:
    return Document(
        id=chunk_uuid(owner, file_id, attempt, idx),
        page_content=text or f"{file_id} chunk {idx} by {owner}",
        metadata={
            "file_id": file_id,
            "user_id": owner,
            "chunk_index": idx,
            "ingestion_attempt_id": attempt,
            "source": f"/uploads/{owner}/{file_id}.txt",
        },
    )


def _chroma() -> ChromaScopedStore:
    import chromadb
    from langchain_chroma import Chroma

    store = Chroma(
        collection_name=f"t{uuid.uuid4().hex}",
        embedding_function=EMB,
        client=chromadb.EphemeralClient(),
        collection_metadata={"hnsw:space": "cosine"},
    )
    return ChromaScopedStore(store)


@pytest.fixture(params=["memory", "chroma"])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[ScopedVectorStore]:
    s: ScopedVectorStore = InMemoryScopedStore(EMB) if request.param == "memory" else _chroma()
    await s.aadd_documents(
        [_doc("f1", "alice", 0), _doc("f1", "alice", 1), _doc("f2", "bob", 0), _doc("f1", "bob", 0)]
    )
    yield s
    await s.aclose()


async def test_ids_are_owner_scoped(store: ScopedVectorStore) -> None:
    assert sorted(await store.get_all_ids(["alice"])) == ["f1"]
    assert sorted(await store.get_all_ids(["bob"])) == ["f1", "f2"]
    assert await store.get_all_ids([]) == []


async def test_filtered_ids_is_not_an_oracle(store: ScopedVectorStore) -> None:
    assert await store.get_filtered_ids(["f2"], ["alice"]) == []
    assert await store.get_filtered_ids(["f2"], ["bob"]) == ["f2"]


async def test_documents_by_ids_respect_owner(store: ScopedVectorStore) -> None:
    docs = await store.get_documents_by_ids(["f1"], ["alice"])
    assert len(docs) == 2
    assert {d.metadata["user_id"] for d in docs} == {"alice"}


async def test_search_never_crosses_owner(store: ScopedVectorStore) -> None:
    hits = await store.asearch(await EMB.aembed_query("f1 chunk 0 by bob"), 10, ["f1"], ["alice"])
    assert hits
    assert all(d.metadata["user_id"] == "alice" for d, _ in hits)
    assert await store.asearch([0.0] * 16, 4, [], ["alice"]) == []


async def test_delete_scoped_leaves_other_owners(store: ScopedVectorStore) -> None:
    await store.delete_scoped(["f1"], ["alice"])
    assert await store.get_all_ids(["alice"]) == []
    assert sorted(await store.get_all_ids(["bob"])) == ["f1", "f2"]


async def test_delete_attempt_is_precise(store: ScopedVectorStore) -> None:
    await store.aadd_documents([_doc("f1", "alice", 0, attempt="a2")])
    await store.delete_attempt("f1", "alice", "a2")
    assert len(await store.get_documents_by_ids(["f1"], ["alice"])) == 2


async def test_list_files(store: ScopedVectorStore) -> None:
    files = {f.file_id: f for f in await store.list_files(["bob"])}
    assert set(files) == {"f1", "f2"}
    assert files["f2"].filename == "f2.txt"
    assert files["f2"].chunks == 1
