"""Adapter testlari: DeepSeek HTTP qatlami (mock transport) va konvertorlar."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from app.llm.base import Capability, LLMError, Msg, ToolCall, ToolSpec, sanitize_json_schema
from app.llm.deepseek import DeepSeekProvider
from app.llm.gemini import _as_json_object, _split_messages

SEARCH_TOOL = ToolSpec(
    name="search_messages",
    description="Chatdan xabar qidiradi",
    parameters={
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
        "additionalProperties": False,
    },
)


def make_provider(handler: Any) -> DeepSeekProvider:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.deepseek.test"
    )
    return DeepSeekProvider(api_key="k", client=client)


# ─── schema tozalash ─────────────────────────────────────────────────────────


def test_sanitize_strips_unsupported_keys() -> None:
    out = sanitize_json_schema(SEARCH_TOOL.parameters)
    assert "$schema" not in out
    assert "additionalProperties" not in out
    assert out["properties"]["query"] == {"type": "string"}
    assert out["required"] == ["query"]


def test_sanitize_is_recursive() -> None:
    schema = {
        "type": "object",
        "properties": {
            "filter": {"type": "object", "additionalProperties": False, "const": 1},
            "tags": {"type": "array", "items": {"type": "string", "$id": "x"}},
        },
    }
    out = sanitize_json_schema(schema)
    assert out["properties"]["filter"] == {"type": "object"}
    assert out["properties"]["tags"]["items"] == {"type": "string"}


# ─── DeepSeek: chat ──────────────────────────────────────────────────────────


async def test_chat_parses_text_and_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "deepseek-chat"
        assert body["messages"][0] == {"role": "system", "content": "sen agentsan"}
        return httpx.Response(
            200,
            json={
                "model": "deepseek-chat",
                "choices": [
                    {"message": {"role": "assistant", "content": "salom"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "prompt_cache_hit_tokens": 8,
                },
            },
        )

    p = make_provider(handler)
    res = await p.chat(
        [Msg.system("sen agentsan"), Msg.user("salom")], model="deepseek-chat", temperature=0.2
    )

    assert res.text == "salom"
    assert res.provider == "deepseek"
    assert (res.usage.tokens_in, res.usage.tokens_out, res.usage.cached_tokens) == (12, 3, 8)
    assert res.finish_reason == "stop"
    await p.aclose()


async def test_chat_parses_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        fn = body["tools"][0]["function"]
        assert fn["name"] == "search_messages"
        assert "$schema" not in fn["parameters"]  # tozalangan
        assert body["tool_choice"] == "auto"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "search_messages",
                                        "arguments": '{"query":"promo","limit":5}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    p = make_provider(handler)
    res = await p.chat([Msg.user("promo postni top")], model="deepseek-chat", tools=[SEARCH_TOOL])

    assert res.text == ""
    assert len(res.tool_calls) == 1
    call = res.tool_calls[0]
    assert (call.id, call.name) == ("c1", "search_messages")
    assert call.arguments == {"query": "promo", "limit": 5}
    await p.aclose()


async def test_broken_tool_arguments_do_not_crash() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "function": {"name": "f", "arguments": "{buzuq json"},
                                }
                            ]
                        }
                    }
                ]
            },
        )

    p = make_provider(handler)
    res = await p.chat([Msg.user("x")], model="deepseek-chat", tools=[SEARCH_TOOL])
    assert res.tool_calls[0].arguments == {}
    await p.aclose()


async def test_reasoner_rejects_tools_before_http() -> None:
    """Capability tekshiruvi so'rov yuborilishidan oldin ishlashi kerak."""
    called = False

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(200, json={})

    p = make_provider(handler)
    with pytest.raises(LLMError, match="quvvatlamaydi"):
        await p.chat([Msg.user("x")], model="deepseek-reasoner", tools=[SEARCH_TOOL])
    assert called is False
    await p.aclose()


async def test_reasoner_drops_temperature() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok", "reasoning_content": "o'yladim"}}]}
        )

    p = make_provider(handler)
    res = await p.chat([Msg.user("x")], model="deepseek-reasoner", temperature=0.9)
    assert "temperature" not in seen
    assert res.reasoning == "o'yladim"
    await p.aclose()


async def test_http_error_is_wrapped() -> None:
    p = make_provider(lambda _: httpx.Response(400, text="bad request"))
    p.max_retries = 1
    with pytest.raises(LLMError, match="HTTP 400"):
        await p.chat([Msg.user("x")], model="deepseek-chat")
    await p.aclose()


async def test_retries_on_429_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="rate limited")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    p = make_provider(handler)
    res = await p.chat([Msg.user("x")], model="deepseek-chat")
    assert res.text == "ok"
    assert calls["n"] == 2
    await p.aclose()


async def test_tool_result_roundtrip_shape() -> None:
    """Tool javobi OpenAI formatida qaytib ketishi kerak."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})

    p = make_provider(handler)
    await p.chat(
        [
            Msg.user("promo postni top"),
            Msg.assistant(tool_calls=[ToolCall("c1", "search_messages", {"query": "promo"})]),
            Msg.tool_result("c1", "search_messages", '{"found":2}'),
        ],
        model="deepseek-chat",
    )

    assistant, tool = seen["messages"][1], seen["messages"][2]
    assert assistant["tool_calls"][0]["function"]["name"] == "search_messages"
    assert assistant["content"] == ""  # None emas
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": '{"found":2}'}
    await p.aclose()


def test_deepseek_capabilities() -> None:
    p = DeepSeekProvider(api_key="k")
    assert Capability.TOOLS in p.capabilities("deepseek-chat")
    assert Capability.EMBED not in p.capabilities("deepseek-chat")


# ─── Gemini konvertorlari ────────────────────────────────────────────────────


def test_split_messages_extracts_system() -> None:
    system, contents = _split_messages(
        [Msg.system("qoida-1"), Msg.system("qoida-2"), Msg.user("salom")]
    )
    assert system == "qoida-1\n\nqoida-2"
    assert len(contents) == 1
    assert contents[0].role == "user"


def test_split_messages_maps_assistant_to_model() -> None:
    _, contents = _split_messages(
        [Msg.assistant("javob", tool_calls=[ToolCall("c1", "f", {"a": 1})])]
    )
    assert contents[0].role == "model"
    parts = contents[0].parts or []
    assert parts[0].text == "javob"
    assert parts[1].function_call is not None
    assert parts[1].function_call.name == "f"
    assert parts[1].function_call.args == {"a": 1}


def test_split_messages_tool_result_becomes_function_response() -> None:
    _, contents = _split_messages([Msg.tool_result("c1", "search_messages", '{"found":2}')])
    part = (contents[0].parts or [])[0]
    assert part.function_response is not None
    assert part.function_response.name == "search_messages"
    assert part.function_response.response == {"found": 2}


def test_as_json_object_wraps_non_objects() -> None:
    assert _as_json_object('{"a":1}') == {"a": 1}
    assert _as_json_object("oddiy matn") == {"result": "oddiy matn"}
    assert _as_json_object("[1,2]") == {"result": [1, 2]}
