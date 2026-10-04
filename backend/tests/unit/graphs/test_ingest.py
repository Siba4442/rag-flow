from collections.abc import Sequence
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragflow.graphs.ingest import IngestContext, run_ingest
from ragflow.settings import Settings
from ragflow.stores.memory import InMemoryScopedStore

EMB = DeterministicFakeEmbedding(size=8)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"chunk_size": 40, "chunk_overlap": 0, "embedding_batch_size": 2}
    return Settings(_env_file=None, **(base | overrides))  # type: ignore[call-arg, arg-type]


@pytest.fixture
def text_file(tmp_path: Path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_text("\n\n".join(f"Paragraph number {i} with some words." for i in range(7)))
    return path


def _inputs(path: Path, owner: str = "alice") -> dict[str, str | None]:
    return {
        "file_id": "f1",
        "owner": owner,
        "filepath": str(path),
        "filename": path.name,
        "content_type": "text/plain",
    }


async def test_ingest_stores_tagged_chunks(text_file: Path) -> None:
    store = InMemoryScopedStore(EMB)
    result = await run_ingest(_inputs(text_file), IngestContext(store, _settings()))  # type: ignore[arg-type]

    assert result["status"] == "stored"
    assert result["known_type"] is True
    docs = await store.get_documents_by_ids(["f1"], ["alice"])
    assert len(docs) == result["stored"] == 7
    meta = docs[0].metadata
    assert meta["user_id"] == "alice"
    assert meta["ingestion_attempt_id"] == result["attempt_id"]
    assert sorted(d.metadata["chunk_index"] for d in docs) == list(range(7))


class FlakyStore(InMemoryScopedStore):
    """Fails on the second batch to exercise rollback."""

    def __init__(self) -> None:
        super().__init__(EMB)
        self.calls = 0

    async def aadd_documents(self, docs: Sequence[Document]) -> list[str]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("embedding provider down")
        return await super().aadd_documents(docs)


async def test_failed_batch_rolls_back_whole_attempt(text_file: Path) -> None:
    store = FlakyStore()
    ctx = IngestContext(store, _settings(parallel_execution=1))
    result = await run_ingest(_inputs(text_file), ctx)  # type: ignore[arg-type]

    assert result["status"] == "rolled_back"
    assert "embedding provider down" in result["errors"][0]
    assert await store.get_all_ids(["alice"]) == []


async def test_rollback_keeps_earlier_attempts(text_file: Path) -> None:
    store = FlakyStore()
    store.calls = -10  # first ingest succeeds fully
    ctx = IngestContext(store, _settings(embedding_batch_size=0))
    first = await run_ingest(_inputs(text_file), ctx)  # type: ignore[arg-type]
    assert first["status"] == "stored"

    store.calls = 1  # next batch fails
    second = await run_ingest(_inputs(text_file), ctx)  # type: ignore[arg-type]
    assert second["status"] == "rolled_back"
    docs = await store.get_documents_by_ids(["f1"], ["alice"])
    assert {d.metadata["ingestion_attempt_id"] for d in docs} == {first["attempt_id"]}


async def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   ")
    store = InMemoryScopedStore(EMB)
    result = await run_ingest(_inputs(path), IngestContext(store, _settings()))  # type: ignore[arg-type]
    assert result["status"] == "empty"
