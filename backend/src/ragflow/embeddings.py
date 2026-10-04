"""Embeddings factory, ported from ``rag-api/app/config.py::init_embeddings``.

Provider packages are optional extras, so each branch imports lazily.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings

from ragflow.settings import EmbeddingsProvider, Settings


def init_embeddings(settings: Settings) -> Embeddings:
    provider = settings.embeddings_provider
    model = settings.resolved_embeddings_model

    match provider:
        case EmbeddingsProvider.OPENAI:
            from langchain_openai import OpenAIEmbeddings

            openai_kwargs: dict[str, Any] = {
                "model": model,
                "api_key": settings.rag_openai_api_key,
                "base_url": settings.rag_openai_baseurl,
                "openai_proxy": settings.rag_openai_proxy,
                "chunk_size": settings.embeddings_chunk_size,
                "check_embedding_ctx_length": settings.rag_check_embedding_ctx_length,
            }
            if settings.embeddings_dimensions is not None:
                openai_kwargs["dimensions"] = settings.embeddings_dimensions
            return OpenAIEmbeddings(**openai_kwargs)

        case EmbeddingsProvider.AZURE:
            from langchain_openai import AzureOpenAIEmbeddings

            azure_kwargs: dict[str, Any] = {
                "azure_deployment": model,
                "api_key": settings.rag_azure_openai_api_key,
                "azure_endpoint": (settings.rag_azure_openai_endpoint or "").rstrip("/"),
                "api_version": settings.rag_azure_openai_api_version,
                "chunk_size": settings.embeddings_chunk_size,
                "check_embedding_ctx_length": settings.rag_check_embedding_ctx_length,
            }
            if settings.embeddings_dimensions is not None:
                azure_kwargs["dimensions"] = settings.embeddings_dimensions
            return AzureOpenAIEmbeddings(**azure_kwargs)

        case EmbeddingsProvider.HUGGINGFACE:
            from langchain_huggingface import (
                HuggingFaceEmbeddings,
            )

            hf: Embeddings = HuggingFaceEmbeddings(
                model_name=model, encode_kwargs={"normalize_embeddings": True}
            )
            return hf

        case EmbeddingsProvider.HUGGINGFACETEI:
            from langchain_huggingface import (
                HuggingFaceEndpointEmbeddings,
            )

            tei: Embeddings = HuggingFaceEndpointEmbeddings(model=model)
            return tei

        case EmbeddingsProvider.OLLAMA:
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=model, base_url=settings.ollama_base_url)

        case EmbeddingsProvider.GOOGLE_GENAI | EmbeddingsProvider.GOOGLE_VERTEXAI:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            google_kwargs: dict[str, Any] = {
                "model": model,
                "google_api_key": settings.rag_google_api_key or None,
            }
            if provider is EmbeddingsProvider.GOOGLE_VERTEXAI:
                google_kwargs |= {
                    "vertexai": True,
                    "project": settings.google_cloud_project,
                    "location": settings.google_cloud_location,
                }
            return GoogleGenerativeAIEmbeddings(**google_kwargs)

        case EmbeddingsProvider.BEDROCK:
            import boto3
            from langchain_aws import BedrockEmbeddings

            session_kwargs: dict[str, Any] = {
                "aws_access_key_id": settings.aws_access_key_id,
                "aws_secret_access_key": settings.aws_secret_access_key,
                "region_name": settings.aws_default_region,
            }
            if settings.aws_session_token:
                session_kwargs["aws_session_token"] = settings.aws_session_token
            session = boto3.Session(**session_kwargs)
            return BedrockEmbeddings(
                client=session.client("bedrock-runtime"),
                model_id=model,
                region_name=settings.aws_default_region,
            )

    raise ValueError(f"Unsupported embeddings provider: {provider}")
