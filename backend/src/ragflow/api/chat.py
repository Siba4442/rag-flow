"""Chat endpoints: ``POST /chat`` (Server-Sent Events) and thread history.

Thread ids are namespaced by the caller's identity (``{user}:{thread_id}``) before they
reach the checkpointer, so one user can never resume or read another user's thread.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from ragflow.api.deps import Services, get_services
from ragflow.graphs.chat import ChatContext, build_chat_graph, turn_input, visible_history
from ragflow.scope import PUBLIC_OWNER, request_user, resolve_scope
from ragflow.settings import Settings, VectorDBType

if TYPE_CHECKING:
    from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

ServicesDep = Annotated[Services, Depends(get_services)]

STREAM_NODES = frozenset({"generate_answer", "generate_query_or_respond"})


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    file_ids: list[str] = Field(default_factory=list, max_length=50)
    entity_id: str | None = None
    k: int | None = Field(default=None, ge=1, le=20)


# ------------------------------------------------------------------------------ setup
async def build_chat_runtime(settings: Settings, services: Services, stack: AsyncExitStack) -> Any:
    """Compile the chat graph with a checkpointer; None if the model can't be built."""
    from langchain.chat_models import init_chat_model

    try:
        services.chat_model = init_chat_model(settings.chat_model)
        services.grader_model = (
            init_chat_model(settings.resolved_grader_model)
            if settings.grader_model
            else services.chat_model
        )
    except Exception as exc:  # missing provider package or credentials
        logger.warning("Chat disabled: cannot init CHAT_MODEL=%r: %s", settings.chat_model, exc)
        return None

    checkpointer: Checkpointer
    if settings.vector_db_type is VectorDBType.PGVECTOR:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
            settings.psycopg_dsn,
            open=False,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await pool.open()
        stack.push_async_callback(pool.close)
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        checkpointer = saver
    else:
        from langgraph.checkpoint.memory import InMemorySaver

        logger.info("Chat memory is in-process (non-persistent) for %s", settings.vector_db_type)
        checkpointer = InMemorySaver()
    return build_chat_graph(checkpointer)


def _thread_key(request: Request, thread_id: str) -> str:
    user = request_user(request)
    owner = str(user.get("id") or PUBLIC_OWNER) if user else PUBLIC_OWNER
    return f"{owner}:{thread_id}"


def _graph(services: Services) -> Any:
    if services.chat_graph is None:
        raise HTTPException(
            status_code=503, detail="Chat is not configured (check CHAT_MODEL and credentials)"
        )
    return services.chat_graph


# ------------------------------------------------------------------------------ routes
@router.post("")
async def chat(body: ChatRequest, request: Request, services: ServicesDep) -> EventSourceResponse:
    graph = _graph(services)
    scope = resolve_scope(request, body.entity_id)
    settings = services.settings
    context = ChatContext(
        retriever=services.retriever,
        model=services.chat_model,
        grader=services.grader_model,
        owners=list(scope.owners),
        file_ids=body.file_ids,
        k=body.k or settings.chat_retrieval_k,
        max_rewrites=settings.chat_max_rewrites,
    )
    config = {"configurable": {"thread_id": _thread_key(request, body.thread_id)}}

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            async for part in graph.astream(
                turn_input(body.message),
                config,
                context=context,
                stream_mode=["messages", "updates"],
                version="v2",
            ):
                if part["type"] == "messages":
                    chunk, meta = part["data"]
                    node = meta.get("langgraph_node")
                    if (
                        node in STREAM_NODES
                        and isinstance(chunk, AIMessageChunk)
                        and not chunk.tool_call_chunks
                        and isinstance(chunk.content, str)
                        and chunk.content
                    ):
                        yield {"event": "token", "data": json.dumps({"text": chunk.content})}
                elif part["type"] == "updates":
                    for node, update in (part["data"] or {}).items():
                        yield {"event": "step", "data": json.dumps({"node": node})}
                        if node == "generate_answer" and update:
                            sources = update.get("sources", [])
                            yield {"event": "sources", "data": json.dumps(sources, default=str)}
            yield {"event": "done", "data": json.dumps({"thread_id": body.thread_id})}
        except Exception as exc:
            logger.exception("chat stream failed")
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}

    return EventSourceResponse(events())


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str, request: Request, services: ServicesDep) -> dict[str, Any]:
    graph = _graph(services)
    snapshot = await graph.aget_state(
        {"configurable": {"thread_id": _thread_key(request, thread_id)}}
    )
    values = snapshot.values or {}
    view = visible_history(values.get("messages", []), values.get("sources", []))
    return {"thread_id": thread_id, "messages": view.messages}


@router.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str, request: Request, services: ServicesDep) -> dict[str, str]:
    graph = _graph(services)
    await graph.checkpointer.adelete_thread(_thread_key(request, thread_id))
    return {"message": "Thread deleted"}
