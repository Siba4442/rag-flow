"""FastAPI application factory and entry point (``uv run ragflow``)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ragflow.api import documents
from ragflow.api.deps import Services
from ragflow.auth import security_middleware
from ragflow.embeddings import init_embeddings
from ragflow.graphs.retrieve import QueryVectorCache, RetrieveContext, Retriever
from ragflow.logging_setup import LogMiddleware, configure_logging
from ragflow.settings import Settings, get_settings
from ragflow.stores.factory import create_vector_store

logger = logging.getLogger("ragflow")


async def build_services(settings: Settings, stack: AsyncExitStack) -> Services:
    embeddings = init_embeddings(settings)
    store = await create_vector_store(settings, embeddings)
    stack.push_async_callback(store.aclose)
    retriever = Retriever(
        RetrieveContext(
            store=store,
            query_cache=QueryVectorCache(embeddings),
            distance_threshold=settings.rag_distance_threshold,
        )
    )
    return Services(settings, embeddings, store, retriever)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.rag_upload_dir.mkdir(parents=True, exist_ok=True)
    async with AsyncExitStack() as stack:
        app.state.services = await build_services(settings, stack)
        logger.info(
            "ragflow ready: store=%s embeddings=%s",
            settings.vector_db_type,
            settings.embeddings_provider,
        )
        yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(title="RAG Flow", version="0.1.0", lifespan=lifespan)

    # Starlette runs the last-added middleware first: CORS must wrap auth so that
    # 401 responses still carry CORS headers the browser can read.
    app.add_middleware(LogMiddleware)
    app.middleware("http")(security_middleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials="*" not in settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors(), "message": "Request validation failed"},
        )

    app.include_router(documents.router)

    if settings.serve_frontend and settings.frontend_dist.is_dir():
        app.mount("/ui", StaticFiles(directory=settings.frontend_dist, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/ui/")

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("ragflow.main:app", host=settings.rag_host, port=settings.rag_port)
