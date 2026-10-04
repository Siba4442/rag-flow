import json

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from ragflow.graphs.chat import build_chat_graph
from tests.unit.api.conftest import Api
from tests.unit.graphs.test_chat import ScriptedModel


def _events(raw: str) -> list[tuple[str, object]]:
    out = []
    for block in raw.replace("\r\n", "\n").strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((fields.get("event", ""), json.loads(fields.get("data", "null"))))
    return out


def _enable_chat(api: Api, replies: list[AIMessage]) -> None:
    model = ScriptedModel(messages=iter(replies), grades=["yes"])
    api.services.chat_model = api.services.grader_model = model
    api.services.chat_graph = build_chat_graph(InMemorySaver())


def test_chat_disabled_is_503(api: Api) -> None:
    r = api.client.post("/chat", json={"message": "hi", "thread_id": "t1"})
    assert r.status_code == 503


def test_chat_streams_tokens_and_keeps_history(api: Api) -> None:
    _enable_chat(api, [AIMessage("Hello there friend")])
    r = api.client.post("/chat", json={"message": "hi", "thread_id": "t1"})
    events = _events(r.text)
    text = "".join(d["text"] for e, d in events if e == "token")  # type: ignore[index]
    assert text == "Hello there friend"
    assert events[-1] == ("done", {"thread_id": "t1"})

    history = api.client.get("/chat/threads/t1").json()["messages"]
    assert [m["role"] for m in history] == ["user", "assistant"]

    api.client.delete("/chat/threads/t1")
    assert api.client.get("/chat/threads/t1").json()["messages"] == []


def test_threads_are_namespaced_per_user(auth_api: Api) -> None:
    _enable_chat(auth_api, [AIMessage("private")])
    alice, bob = auth_api.token("alice"), auth_api.token("bob")
    auth_api.client.post("/chat", json={"message": "hi", "thread_id": "t1"}, headers=alice)
    assert auth_api.client.get("/chat/threads/t1", headers=bob).json()["messages"] == []
    assert len(auth_api.client.get("/chat/threads/t1", headers=alice).json()["messages"]) == 2


def test_stream_errors_do_not_leak_internals(api: Api) -> None:
    class Boom(ScriptedModel):
        async def ainvoke(self, *_a: object, **_k: object) -> AIMessage:  # type: ignore[override]
            raise RuntimeError("api key sk-secret rejected")

    _enable_chat(api, [])
    api.services.chat_model = Boom(messages=iter([]))
    r = api.client.post("/chat", json={"message": "hi", "thread_id": "t9"})
    events = _events(r.text)
    assert events[-1][0] == "error"
    assert "sk-secret" not in r.text


def test_preamble_before_tool_call_is_not_streamed(api: Api) -> None:
    """Real providers stream text first, then tool-call chunks; only the answer is shown."""
    from collections.abc import AsyncIterator
    from typing import Any

    from langchain_core.messages import AIMessage as AI
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk

    from tests.unit.graphs.test_chat import ScriptedModel

    class Streaming(ScriptedModel):
        calls: int = 0

        async def _astream(self, *a: Any, **k: Any) -> AsyncIterator[ChatGenerationChunk]:
            self.calls += 1
            if self.calls == 1:  # router: preamble text, then a tool call
                yield ChatGenerationChunk(message=AIMessageChunk(content="Let me look that up. "))
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "id": "c1",
                                "name": "search_documents",
                                "args": '{"query":"x"}',
                                "index": 0,
                                "type": "tool_call_chunk",
                            }
                        ],
                    )
                )
            else:  # answer
                for word in ("Final ", "answer"):
                    yield ChatGenerationChunk(message=AIMessageChunk(content=word))

    model = Streaming(messages=iter([AI("unused")]), grades=["yes"])
    api.services.chat_model = api.services.grader_model = model
    api.services.chat_graph = build_chat_graph(InMemorySaver())
    r = api.client.post("/chat", json={"message": "q", "thread_id": "p1", "file_ids": ["f"]})
    events = _events(r.text)
    text = "".join(d["text"] for e, d in events if e == "token")  # type: ignore[index]
    assert "retrieve" in [d["node"] for e, d in events if e == "step"]  # type: ignore[index]
    assert "Let me look" not in text
    assert text == "Final answer"
