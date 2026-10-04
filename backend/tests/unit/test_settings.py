import pytest

from ragflow.settings import EmbeddingsProvider, Settings, VectorDBType


def test_rag_api_env_names_and_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fallback")
    monkeypatch.setenv("EMBEDDINGS_PROVIDER", "OLLAMA")
    monkeypatch.setenv("VECTOR_DB_TYPE", "chroma")
    monkeypatch.setenv("CORS_ORIGINS", "http://a, http://b")
    monkeypatch.setenv("RAG_DISTANCE_THRESHOLD", "")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.rag_openai_api_key == "fallback"
    assert s.embeddings_provider is EmbeddingsProvider.OLLAMA
    assert s.resolved_embeddings_model == "nomic-embed-text"
    assert s.vector_db_type is VectorDBType.CHROMA
    assert s.cors_origins == ["http://a", "http://b"]
    assert s.rag_distance_threshold is None


def test_connection_urls_quote_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/word")
    monkeypatch.setenv("DB_HOST", "localhost")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.sqlalchemy_url == "postgresql+psycopg://myuser:p%40ss%2Fword@localhost:5432/mydatabase"
    assert s.psycopg_dsn.startswith("postgresql://myuser:")


def test_mongo_deprecated_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONGO_VECTOR_COLLECTION", "legacy")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.resolved_collection_name == "legacy"
    assert s.resolved_search_index == "legacy"
