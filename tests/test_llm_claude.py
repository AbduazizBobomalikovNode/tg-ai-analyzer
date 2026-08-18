"""Claude adapter testlari — tarmoqsiz: soxta `messages.create` orqali.

Nimalarni qo'riqlaydi:
  * rollar konvertatsiyasi (system ajratiladi, tool natijalari user'ga yig'iladi)
  * Opus 5 va boshqa yangi modellarga `temperature` yuborilmaydi (400 beradi)
  * OAuth token bilan `oauth-2025-04-20` beta header'i qo'yiladi
  * tool_use bloklari ToolCall'ga aylanadi, refusal xato emas
  * json_mode'da ``` fence'lar olib tashlanadi
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.llm import claude as C
from app.llm.base import Capability, LLMError, Msg, ToolCall, ToolSpec


class _FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.messages = _FakeMessages(response, error)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _resp(
    *blocks: Any,
    stop: str = "end_turn",
    model: str = "claude-opus-5",
    usage: tuple[int, int, int] = (10, 5, 0),
) -> Any:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop,
        model=model,
        usage=SimpleNamespace(
            input_tokens=usage[0], output_tokens=usage[1], cache_read_input_tokens=usage[2]
        ),
    )


def _text(t: str) -> Any:
    return SimpleNamespace(type="text", text=t)


def _tool_use(id_: str, name: str, inp: dict[str, Any]) -> Any:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inp)


def _provider(client: _FakeClient) -> C.ClaudeProvider:
    return C.ClaudeProvider(auth=C.ClaudeAuth("api_key", "ANTHROPIC_API_KEY", "k"), client=client)


# ─── konvertorlar ────────────────────────────────────────────────────────────


def test_split_messages_extracts_system_and_maps_roles() -> None:
    system, msgs = C.split_messages(
        [
            Msg.system("A"),
            Msg.user("hi"),
            Msg.assistant("hello", [ToolCall("t1", "search", {"q": "x"})]),
            Msg.tool_result("t1", "search", '{"ok": true}'),
            Msg.tool_result("t2", "search", "second"),
            Msg.system("B"),
            Msg.user("next"),
        ]
    )
    assert system == "A\n\nB"
    assert msgs[0] == {"role": "user", "content": "hi"}
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == [
        {"type": "text", "text": "hello"},
        {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
    ]
    # ikkita ketma-ket tool natijasi bitta user xabariga yig'iladi
    assert msgs[2]["role"] == "user"
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["t1", "t2"]
    assert all(b["type"] == "tool_result" for b in msgs[2]["content"])
    assert msgs[3] == {"role": "user", "content": "next"}


def test_build_request_omits_temperature_for_opus5_but_keeps_for_haiku() -> None:
    msgs = [Msg.user("hi")]
    opus = C.build_request(
        msgs, model="claude-opus-5", tools=None, temperature=0.2, max_tokens=None, json_mode=False
    )
    haiku = C.build_request(
        msgs, model="claude-haiku-4-5", tools=None, temperature=0.2, max_tokens=64, json_mode=False
    )
    assert "temperature" not in opus
    assert haiku["temperature"] == 0.2
    assert opus["max_tokens"] == C._DEFAULT_MAX_TOKENS  # majburiy parametr
    assert haiku["max_tokens"] == 64
    assert "system" not in opus


def test_build_request_json_mode_adds_instruction_and_tools_schema_untouched() -> None:
    schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "additionalProperties": False,
    }
    req = C.build_request(
        [Msg.system("S"), Msg.user("u")],
        model="claude-opus-5",
        tools=[ToolSpec("search", "Search", schema)],
        temperature=0.7,
        max_tokens=None,
        json_mode=True,
    )
    assert req["system"].startswith("S\n\n")
    assert "JSON" in req["system"]
    assert req["tools"] == [{"name": "search", "description": "Search", "input_schema": schema}]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('  {"a": 1}  ', '{"a": 1}'),
        ("plain", "plain"),
    ],
)
def test_strip_json_fences(raw: str, expected: str) -> None:
    assert C.strip_json_fences(raw) == expected


# ─── auth ────────────────────────────────────────────────────────────────────


def test_capabilities() -> None:
    p = C.ClaudeProvider(auth=C.ClaudeAuth("api_key", "ANTHROPIC_API_KEY", "k"), client=object())
    caps = p.capabilities("claude-opus-5")
    assert {
        Capability.CHAT,
        Capability.TOOLS,
        Capability.JSON_MODE,
        Capability.LONG_CONTEXT,
    } <= caps
    assert Capability.EMBED not in caps and Capability.IMAGE_GEN not in caps
    assert p.capabilities("gpt-4") == frozenset()


def test_oauth_token_client_gets_beta_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Anthropic:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)
            self.messages = None

    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _Anthropic)
    C.ClaudeProvider(auth=C.ClaudeAuth("oauth_token", "CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x"))
    assert captured["auth_token"] == "sk-ant-oat01-x"  # noqa: S105
    assert captured["default_headers"] == {"anthropic-beta": C.OAUTH_BETA_HEADER}
    assert captured["max_retries"] == 0
    assert "api_key" not in captured


def test_api_key_client_has_no_oauth_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Anthropic:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _Anthropic)
    C.ClaudeProvider(auth=C.ClaudeAuth("api_key", "ANTHROPIC_API_KEY", "sk-ant-api"))
    assert captured["api_key"] == "sk-ant-api"
    assert "default_headers" not in captured and "auth_token" not in captured


def test_profile_client_is_zero_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Anthropic:
        def __init__(self, **kw: Any) -> None:
            captured.update(kw)

    import anthropic

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _Anthropic)
    C.ClaudeProvider(auth=C.ClaudeAuth("profile", "profile"))
    assert set(captured) == {"max_retries", "timeout"}


# ─── chat ────────────────────────────────────────────────────────────────────


async def test_chat_parses_text_usage_and_finish() -> None:
    client = _FakeClient(_resp(_text("Salom"), usage=(12, 3, 7)))
    res = await _provider(client).chat([Msg.system("S"), Msg.user("hi")], model="claude-opus-5")
    assert res.text == "Salom"
    assert res.provider == "claude"
    assert res.model == "claude-opus-5"
    assert (res.usage.tokens_in, res.usage.tokens_out, res.usage.cached_tokens) == (12, 3, 7)
    assert res.finish_reason == "end_turn"
    sent = client.messages.calls[0]
    assert sent["system"] == "S"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


async def test_chat_parses_tool_calls() -> None:
    client = _FakeClient(
        _resp(_text("Qidiryapman"), _tool_use("toolu_1", "search", {"q": "x"}), stop="tool_use")
    )
    res = await _provider(client).chat(
        [Msg.user("find x")],
        model="claude-opus-5",
        tools=[ToolSpec("search", "d", {"type": "object", "properties": {}})],
    )
    assert res.text == "Qidiryapman"
    assert res.tool_calls == [ToolCall("toolu_1", "search", {"q": "x"})]
    assert res.finish_reason == "tool_use"


async def test_refusal_is_not_an_error() -> None:
    client = _FakeClient(_resp(stop="refusal"))
    res = await _provider(client).chat([Msg.user("x")], model="claude-opus-5")
    assert res.finish_reason == "refusal"
    assert res.text == "" and res.tool_calls == []


async def test_json_mode_strips_fences() -> None:
    client = _FakeClient(_resp(_text('```json\n{"intent": "search"}\n```')))
    res = await _provider(client).chat([Msg.user("x")], model="claude-haiku-4-5", json_mode=True)
    assert res.text == '{"intent": "search"}'


async def test_http_error_is_wrapped_with_status() -> None:
    class _Err(Exception):
        status_code = 400
        message = "bad"

    client = _FakeClient(error=_Err("bad"))
    with pytest.raises(LLMError) as ei:
        await _provider(client).chat([Msg.user("x")], model="claude-opus-5")
    assert ei.value.status == 400
    assert len(client.messages.calls) == 1  # 400 qayta urinilmaydi


async def test_missing_credentials_fails_before_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C, "detect_claude_auth", lambda *_a, **_k: None)
    p = C.ClaudeProvider()
    assert p.auth is None
    with pytest.raises(LLMError, match="kredensial topilmadi"):
        await p.chat([Msg.user("x")], model="claude-opus-5")


async def test_aclose_closes_client() -> None:
    client = _FakeClient()
    p = _provider(client)
    await p.aclose()
    assert client.closed
