"""Runtime configuration.

Environment variable names match LibreChat's ``rag-api/app/config.py`` so an existing
``.env`` keeps working. Unlike the original, nothing is built at import time: call
:func:`get_settings` and construct clients from the returned object.
"""

from __future__ import annotations

import urllib.parse
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class VectorDBType(StrEnum):
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    CHROMA = "chroma"
    ATLAS_MONGO = "atlas-mongo"
    MEMORY = "memory"  # in-process, non-persistent; for tests and quick local trials


class EmbeddingsProvider(StrEnum):
    OPENAI = "openai"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    HUGGINGFACETEI = "huggingfacetei"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    GOOGLE_GENAI = "google_genai"
    GOOGLE_VERTEXAI = "vertexai"


class EntityIdPolicy(StrEnum):
    """How a caller-supplied ``entity_id`` is treated when JWT auth is on."""

    TRUST = "trust"  # rag-api behaviour: caller-asserted (authorize upstream!)
    CLAIM = "claim"  # only entities listed in the token's ``entities`` claim
    DENY = "deny"  # ignore entity_id; owner is always the token user


class PdfParser(StrEnum):
    PYPDF = "pypdf"
    LLAMAPARSE = "llamaparse"
    DOCLING = "docling"


DEFAULT_EMBEDDING_MODELS: dict[EmbeddingsProvider, str] = {
    EmbeddingsProvider.OPENAI: "text-embedding-3-small",
    EmbeddingsProvider.AZURE: "text-embedding-3-small",
    EmbeddingsProvider.HUGGINGFACE: "sentence-transformers/all-MiniLM-L6-v2",
    EmbeddingsProvider.HUGGINGFACETEI: "http://huggingfacetei:3000",
    EmbeddingsProvider.OLLAMA: "nomic-embed-text",
    EmbeddingsProvider.BEDROCK: "amazon.titan-embed-text-v1",
    EmbeddingsProvider.GOOGLE_GENAI: "gemini-embedding-001",
    EmbeddingsProvider.GOOGLE_VERTEXAI: "gemini-embedding-001",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- server
    rag_host: str = "0.0.0.0"
    rag_port: int = 8000
    rag_upload_dir: Path = Path("./uploads/")
    #: /local/embed access: "owner" = only {upload_dir}/{owner}/..., "any" = rag-api behaviour
    local_embed_scope: Literal["owner", "any"] = "owner"
    rag_thread_pool_size: int | None = None
    debug_rag_api: bool = False
    console_json: bool = False
    jwt_secret: str | None = None
    entity_id_policy: EntityIdPolicy = EntityIdPolicy.CLAIM
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    serve_frontend: bool = False
    frontend_dist: Path = Path("../frontend/dist")

    # --- ingestion
    chunk_size: int = 1500
    chunk_overlap: int = 100
    parallel_execution: int = 2
    embedding_batch_size: int = 500
    pdf_extract_images: bool = False
    pdf_parser: PdfParser = PdfParser.PYPDF
    llama_cloud_api_key: str | None = None

    # --- vector store
    vector_db_type: VectorDBType = VectorDBType.PGVECTOR
    collection_name: str = "testcollection"
    rag_distance_threshold: float | None = None

    # postgres
    postgres_use_unix_socket: bool = False
    postgres_db: str = "mydatabase"
    postgres_user: str = "myuser"
    postgres_password: str = ""
    db_host: str = "db"
    db_port: int = 5432
    postgres_schema: str | None = None
    pgvector_table: str = "rag_chunks"
    pgvector_create_extension: bool = True

    # pinecone
    pinecone_api_key: str | None = None
    pinecone_index: str = "ragflow"
    pinecone_namespace: str | None = None

    # chroma
    chroma_host: str | None = None
    chroma_port: int = 8000
    chroma_persist_dir: Path = Path("./chroma")

    # atlas mongo
    atlas_mongo_db_uri: str = "mongodb://127.0.0.1:27018/LibreChat"
    atlas_search_index: str = "vector_index"
    mongo_vector_collection: str | None = None  # deprecated alias, as in rag-api

    # --- embeddings
    embeddings_provider: EmbeddingsProvider = EmbeddingsProvider.OPENAI
    embeddings_model: str | None = None
    embeddings_dimensions: int | None = None
    embeddings_chunk_size: int = 200
    rag_check_embedding_ctx_length: bool = True

    rag_openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("RAG_OPENAI_API_KEY", "OPENAI_API_KEY")
    )
    rag_openai_baseurl: str | None = None
    rag_openai_proxy: str | None = None
    rag_azure_openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
    )
    rag_azure_openai_endpoint: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_ENDPOINT"),
    )
    rag_azure_openai_api_version: str | None = None
    ollama_base_url: str = "http://ollama:11434"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_default_region: str = "us-east-1"
    rag_google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RAG_GOOGLE_API_KEY", "GOOGLE_KEY", "GOOGLE_API_KEY"),
    )
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"

    # --- chat graph
    chat_model: str = "openai:gpt-5-mini"
    grader_model: str | None = None
    chat_retrieval_k: int = 6
    chat_max_rewrites: int = 2

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

    @field_validator("embeddings_provider", mode="before")
    @classmethod
    def _lower_provider(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator(
        "rag_distance_threshold", "embeddings_dimensions", "postgres_schema", mode="before"
    )
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @property
    def resolved_embeddings_model(self) -> str:
        return self.embeddings_model or DEFAULT_EMBEDDING_MODELS[self.embeddings_provider]

    @property
    def resolved_collection_name(self) -> str:
        return self.mongo_vector_collection or self.collection_name

    @property
    def resolved_search_index(self) -> str:
        return self.mongo_vector_collection or self.atlas_search_index

    @property
    def resolved_grader_model(self) -> str:
        return self.grader_model or self.chat_model

    def _pg_suffix(self) -> str:
        q = urllib.parse.quote_plus
        creds = f"{q(self.postgres_user)}:{q(self.postgres_password)}"
        if self.postgres_use_unix_socket:
            return f"{creds}@/{q(self.postgres_db)}?host={q(self.db_host)}"
        return f"{creds}@{self.db_host}:{self.db_port}/{q(self.postgres_db)}"

    @property
    def sqlalchemy_url(self) -> str:
        """Async SQLAlchemy URL (psycopg3) for ``langchain_postgres.PGEngine``."""
        return f"postgresql+psycopg://{self._pg_suffix()}"

    @property
    def psycopg_dsn(self) -> str:
        """Plain libpq DSN for psycopg pools (checkpointer)."""
        return f"postgresql://{self._pg_suffix()}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


KNOWN_SOURCE_EXT: frozenset[str] = frozenset(
    {
        "go",
        "py",
        "java",
        "sh",
        "bat",
        "ps1",
        "cmd",
        "js",
        "ts",
        "css",
        "cpp",
        "hpp",
        "h",
        "c",
        "cs",
        "sql",
        "log",
        "ini",
        "pl",
        "pm",
        "r",
        "dart",
        "dockerfile",
        "env",
        "php",
        "hs",
        "hsc",
        "lua",
        "nginxconf",
        "conf",
        "m",
        "mm",
        "plsql",
        "perl",
        "rb",
        "rs",
        "db2",
        "scala",
        "bash",
        "swift",
        "vue",
        "svelte",
        "yml",
        "yaml",
        "eml",
        "ex",
        "exs",
        "erl",
        "tsx",
        "jsx",
        "lhs",
    }
)
