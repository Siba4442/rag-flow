"""Application services shared by the routers (built once in the lifespan)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request
from langchain_core.embeddings import Embeddings

from ragflow.graphs.retrieve import Retriever
from ragflow.settings import Settings
from ragflow.stores.base import ScopedVectorStore


@dataclass
class Services:
    settings: Settings
    embeddings: Embeddings
    store: ScopedVectorStore
    retriever: Retriever
    #: Compiled chat graph; None when CHAT_MODEL can't be initialised.
    chat_graph: Any = None
    chat_model: Any = None
    grader_model: Any = None


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services
