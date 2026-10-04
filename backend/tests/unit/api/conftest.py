from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragflow.api.deps import Services
from ragflow.graphs.retrieve import QueryVectorCache, RetrieveContext, Retriever
from ragflow.main import create_app
from ragflow.settings import Settings, get_settings
from ragflow.stores.memory import InMemoryScopedStore

SECRET = "unit-test-secret-that-is-long-enough!"


@dataclass
class Api:
    client: TestClient
    services: Services
    app: FastAPI

    def token(self, user_id: str, **claims: object) -> dict[str, str]:
        tok = jwt.encode({"id": user_id, **claims}, SECRET, algorithm="HS256")
        return {"Authorization": f"Bearer {tok}"}


def _make(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **env: str) -> Api:
    monkeypatch.setenv("RAG_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("CHUNK_SIZE", "60")
    monkeypatch.setenv("CHUNK_OVERLAP", "0")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    settings.rag_upload_dir.mkdir(parents=True, exist_ok=True)
    emb = DeterministicFakeEmbedding(size=16)
    store = InMemoryScopedStore(emb)
    services = Services(
        settings,
        emb,
        store,
        Retriever(RetrieveContext(store, QueryVectorCache(emb), settings.rag_distance_threshold)),
    )
    app = create_app(settings)
    app.state.services = services  # TestClient without `with` skips the lifespan
    return Api(TestClient(app), services, app)


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Api]:
    monkeypatch.delenv("JWT_SECRET", raising=False)
    yield _make(monkeypatch, tmp_path)
    get_settings.cache_clear()


@pytest.fixture
def auth_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Api]:
    yield _make(monkeypatch, tmp_path, JWT_SECRET=SECRET, ENTITY_ID_POLICY="trust")
    get_settings.cache_clear()
