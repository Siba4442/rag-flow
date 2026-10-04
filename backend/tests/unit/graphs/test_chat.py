"""Chat graph behaviour with scripted fake models (no network)."""

from collections.abc import Iterator, Sequence
from typing import Any, Literal

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from ragflow.graphs.chat import ChatContext, GradeDocuments, build_chat_graph, turn_input
from ragflow.graphs.retrieve import QueryVectorCache, RetrieveContext, Retriever
from ragflow.stores.base import chunk_uuid
from ragflow.stores.memory import InMemoryScopedStore

EMB = DeterministicFakeEmbedding(size=16)


class ScriptedModel(GenericFakeChatModel):
    """Fake chat model that tolerates bind_tools / with_structured_output."""

    grades: list[Literal["yes", "no"]] = []  # noqa: RUF012 - pydantic field

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedModel":
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return RunnableLambda(lambda _: GradeDocuments(binary_score=self.grades.pop(0)))


def _tool_call(query: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        "", tool_calls=[{"id": call_id, "name": "search_documents", "args": {"query": query}}]
    )


async def _store() -> InMemoryScopedStore:
    store = InMemoryScopedStore(EMB)
    docs = [
        Document(
            id=chunk_uuid(owner, "f1", "a", i),
            page_content=f"{owner} secret fact {i}",
            metadata={"file_id": "f1", "user_id": owner, "chunk_index": i, "source": "doc.txt"},
        )
        for owner in ("alice", "mallory")
        for i in range(3)
    ]
    await store.aadd_documents(docs)
    return store


def _ctx(store: InMemoryScopedStore, model: ScriptedModel, **kw: Any) -> ChatContext:
    retriever = Retriever(RetrieveContext(store, QueryVectorCache(EMB)))
    return ChatContext(
        retriever=retriever, model=model, grader=model, owners=["alice"], file_ids=["f1"], k=3, **kw
    )


def _model(
    messages: list[AIMessage], grades: list[Literal["yes", "no"]] | None = None
) -> ScriptedModel:
    it: Iterator[AIMessage] = iter(messages)
    return ScriptedModel(messages=it, grades=grades or [])


async def test_retrieves_only_callers_chunks_and_answers() -> None:
    store = await _store()
    model = _model([_tool_call("secret fact"), AIMessage("Alice's fact [1].")], ["yes"])
    graph = build_chat_graph()
    out = await graph.ainvoke(turn_input("what is the fact?"), context=_ctx(store, model))

    tool_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs
    assert "mallory" not in tool_msgs[0].content
    assert out["messages"][-1].content == "Alice's fact [1]."
    assert out["sources"]
    assert {s["file_id"] for s in out["sources"]} == {"f1"}
    assert "text" not in out["sources"][0]


async def test_irrelevant_results_trigger_rewrite_then_answer() -> None:
    store = await _store()
    model = _model(
        [
            _tool_call("bad query", "c1"),
            AIMessage("better question"),  # rewrite_question
            _tool_call("better question", "c2"),
            AIMessage("final answer"),
        ],
        grades=["no", "yes"],
    )
    out = await build_chat_graph().ainvoke(turn_input("q?"), context=_ctx(store, model))
    assert out["rewrites"] == 1
    assert out["messages"][-1].content == "final answer"


async def test_rewrite_loop_is_capped() -> None:
    store = await _store()
    model = _model(
        [
            _tool_call("q1", "c1"),
            AIMessage("rewrite"),
            _tool_call("q2", "c2"),
            AIMessage("best effort"),
        ],
        grades=["no", "no", "no"],
    )
    out = await build_chat_graph().ainvoke(
        turn_input("q?"), context=_ctx(store, model, max_rewrites=1)
    )
    assert out["rewrites"] == 1
    assert out["messages"][-1].content == "best effort"


async def test_direct_answer_without_files() -> None:
    store = await _store()
    model = _model([AIMessage("Hello!")])
    ctx = _ctx(store, model)
    ctx = ChatContext(**{**ctx.__dict__, "file_ids": []})
    out = await build_chat_graph().ainvoke(turn_input("hi"), context=ctx)
    assert out["messages"][-1].content == "Hello!"
    assert not [m for m in out["messages"] if isinstance(m, ToolMessage)]


async def test_memory_persists_across_turns() -> None:
    store = await _store()
    model = _model([AIMessage("first"), AIMessage("second")])
    ctx = ChatContext(**{**_ctx(store, model).__dict__, "file_ids": []})
    graph = build_chat_graph(InMemorySaver())
    cfg: Any = {"configurable": {"thread_id": "alice:t1"}}
    await graph.ainvoke(turn_input("one"), cfg, context=ctx)
    out = await graph.ainvoke(turn_input("two"), cfg, context=ctx)
    contents = [m.content for m in out["messages"]]
    assert contents == ["one", "first", "two", "second"]


@pytest.mark.parametrize("bad", ["", "a b", "x/../y"])
def test_thread_id_validation(bad: str) -> None:
    from pydantic import ValidationError

    from ragflow.api.chat import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest(message="hi", thread_id=bad)


async def test_sources_survive_checkpoint_serialisation() -> None:
    """numpy scores/pages from stores must not break the Postgres/msgpack checkpointer."""
    import numpy as np

    from ragflow.graphs.chat import _source

    src = _source(
        1, {"file_id": "f", "page": np.int64(2), "chunk_index": np.int64(0)}, "t", np.float64(0.25)
    )
    assert type(src["score"]) is float
    assert type(src["page"]) is int

    store = await _store()
    model = _model([_tool_call("secret"), AIMessage("ok [1]")], ["yes"])
    graph = build_chat_graph(InMemorySaver())
    cfg: Any = {"configurable": {"thread_id": "alice:ser"}}
    await graph.ainvoke(turn_input("q"), cfg, context=_ctx(store, model))
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    state = await graph.aget_state(cfg)
    JsonPlusSerializer().dumps_typed(state.values)  # raises on numpy scalars
