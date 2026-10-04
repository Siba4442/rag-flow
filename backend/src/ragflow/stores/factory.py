"""Build the configured :class:`ScopedVectorStore` (``VECTOR_DB_TYPE``)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from ragflow.settings import Settings, VectorDBType
from ragflow.stores.base import ScopedVectorStore


async def create_vector_store(settings: Settings, embeddings: Embeddings) -> ScopedVectorStore:
    match settings.vector_db_type:
        case VectorDBType.PGVECTOR:
            from ragflow.stores.pgvector import PgVectorScopedStore

            schema = (settings.postgres_schema or "public").split(",")[0].strip()
            return await PgVectorScopedStore.create(
                settings.sqlalchemy_url,
                embeddings,
                table=settings.pgvector_table,
                schema=schema,
            )
        case VectorDBType.CHROMA:
            from ragflow.stores.chroma import ChromaScopedStore

            return ChromaScopedStore.create(
                embeddings,
                collection_name=settings.collection_name,
                host=settings.chroma_host,
                port=settings.chroma_port,
                persist_dir=settings.chroma_persist_dir,
            )
        case VectorDBType.PINECONE:
            from ragflow.stores.pinecone import PineconeScopedStore

            return PineconeScopedStore.create(
                embeddings,
                api_key=settings.pinecone_api_key,
                index_name=settings.pinecone_index,
                namespace=settings.pinecone_namespace,
            )
        case VectorDBType.ATLAS_MONGO:
            from ragflow.stores.atlas_mongo import AtlasMongoScopedStore

            return AtlasMongoScopedStore.create(
                embeddings,
                uri=settings.atlas_mongo_db_uri,
                collection_name=settings.resolved_collection_name,
                index_name=settings.resolved_search_index,
            )
        case VectorDBType.MEMORY:
            from ragflow.stores.memory import InMemoryScopedStore

            return InMemoryScopedStore(embeddings)
