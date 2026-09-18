"""Adapter testlari: DeepSeek HTTP qatlami (mock transport) va konvertorlar."""

from __future__ import annotations

import asyncio
import json
import math
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import types

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


def test_deepseek_strips_leaked_tool_markup() -> None:
    """V4 uzun transkriptdan keyin ichki razmetkani matn kanalida chiqaradi.

    Razmetka `finish_reason="stop"` bilan ham keladi, shuning uchun adapter uni
    kesadi — aks holda foydalanuvchi javob o'rniga xom markup ko'radi.
    """
    from app.llm.deepseek import _strip_tool_markup

    bar = "｜"  # noqa: RUF001  (FULLWIDTH VERTICAL LINE — razmetka aynan shunday keladi)
    leaked = (
        "Serverdan qidiraman:\n\n"
        f"<{bar}{bar}DSML{bar}{bar}tool_calls>\n"
        f'<{bar}{bar}DSML{bar}{bar}invoke name="search_telegram">'
    )
    assert _strip_tool_markup(leaked) == "Serverdan qidiraman:"

    clean = "Javob: #843 da kv puli haqida yozilgan."
    assert _strip_tool_markup(clean) == clean
    assert _strip_tool_markup("") == ""


# ─── phase4/B: Gemini embedding — o'lcham va normallashtirish ────────────────


class _FakeEmbeddings:
    """`client.aio.models` o'rnini bosadi: chaqiruv konfiguratsiyasini yozib oladi."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[dict[str, Any]] = []

    async def embed_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=v) for v in self.vectors],
            metadata=SimpleNamespace(billable_character_count=7),
        )


def _gemini_with(vectors: list[list[float]]) -> tuple[Any, _FakeEmbeddings]:
    from app.llm.gemini import GeminiProvider

    provider = GeminiProvider(api_key="test-key")  # soxta kalit — tarmoqqa chiqmaydi
    fake = _FakeEmbeddings(vectors)
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=fake))
    return provider, fake


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def test_embed_passes_output_dimensionality_and_task_type() -> None:
    provider, fake = _gemini_with([[3.0, 4.0]])
    res = asyncio.run(
        provider.embed(
            ["salom"], model="gemini-embedding-001", dim=768, task_type="RETRIEVAL_QUERY"
        )
    )
    cfg = fake.calls[0]["config"]
    assert cfg.output_dimensionality == 768
    assert cfg.task_type == "RETRIEVAL_QUERY"
    assert res.model == "gemini-embedding-001"


def test_embedding_001_vectors_are_normalized_client_side() -> None:
    """Kesilgan vektor birlik uzunlikda emas — 001 uchun biz normallashtiramiz."""
    provider, _ = _gemini_with([[3.0, 4.0], [0.0, 0.0]])
    res = asyncio.run(provider.embed(["a", "b"], model="gemini-embedding-001", dim=2))
    assert _norm(res.vectors[0]) == pytest.approx(1.0)
    assert res.vectors[0] == pytest.approx([0.6, 0.8])
    assert res.vectors[1] == [0.0, 0.0]  # nol vektor — bo'lish yo'q


def test_embedding_2_vectors_are_left_untouched() -> None:
    """embedding-2 o'zi normallashtiradi — ikkinchi marta bo'lish keraksiz."""
    provider, _ = _gemini_with([[3.0, 4.0]])
    res = asyncio.run(provider.embed(["a"], model="gemini-embedding-2", dim=2))
    assert res.vectors[0] == [3.0, 4.0]


def test_normalize_vectors_handles_model_prefix() -> None:
    from app.llm.gemini import NORMALIZE_CLIENT_SIDE, normalize_vectors

    assert NORMALIZE_CLIENT_SIDE == frozenset({"gemini-embedding-001"})
    out = normalize_vectors("models/gemini-embedding-001", [[3.0, 4.0]])
    assert _norm(out[0]) == pytest.approx(1.0)


# ─── 2026-09-16: embed usage yo'q bo'lsa token taxmini (byudjet qopqog'i) ────


class _FakeEmbeddingsNoUsage:
    """`embed_content` javobida usage/metadata umuman yo'q — jonli holat."""

    def __init__(self, vectors: list[list[float]], *, metadata: Any = None) -> None:
        self.vectors = vectors
        self.metadata = metadata
        self.calls: list[dict[str, Any]] = []

    async def embed_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=v) for v in self.vectors], metadata=self.metadata
        )


def _gemini_no_usage(
    vectors: list[list[float]], *, metadata: Any = None
) -> tuple[Any, _FakeEmbeddingsNoUsage]:
    from app.llm.gemini import GeminiProvider

    provider = GeminiProvider(api_key="test-key")
    fake = _FakeEmbeddingsNoUsage(vectors, metadata=metadata)
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=fake))
    return provider, fake


def test_embed_estimates_tokens_when_metadata_is_missing() -> None:
    """`resp.metadata` yo'q — `tokens_in` 0 qolib, narx (`estimate_cost`) ham 0
    bo'lib qolmasin (kunlik xarajat limiti jimgina ishlamay qoladi)."""
    provider, _ = _gemini_no_usage([[3.0, 4.0]], metadata=None)
    res = asyncio.run(provider.embed(["a" * 40], model="gemini-embedding-001", dim=2))
    assert res.usage.tokens_in == 10  # ceil(40 / 4)


def test_embed_estimates_tokens_when_billable_count_is_zero() -> None:
    """`resp.metadata` bor, lekin `billable_character_count` yo'q/0 — bir xil taxmin."""
    provider, _ = _gemini_no_usage(
        [[3.0, 4.0], [1.0, 0.0]], metadata=SimpleNamespace(billable_character_count=0)
    )
    res = asyncio.run(provider.embed(["a" * 4, "b" * 6], model="gemini-embedding-001", dim=2))
    assert res.usage.tokens_in == 1 + 2  # ceil(4/4) + ceil(6/4)


def test_embed_prefers_real_usage_over_estimate() -> None:
    """Javobda haqiqiy son bo'lsa taxminga tushilmaydi."""
    provider, _ = _gemini_no_usage(
        [[3.0, 4.0]], metadata=SimpleNamespace(billable_character_count=999)
    )
    res = asyncio.run(provider.embed(["a" * 4000], model="gemini-embedding-001", dim=2))
    assert res.usage.tokens_in == 999


def test_estimate_tokens_heuristic_matches_char_over_four() -> None:
    from app.llm.base import estimate_tokens_heuristic

    assert estimate_tokens_heuristic([]) == 0
    assert estimate_tokens_heuristic(["", "abcd", "abcde"]) == 1 + 2  # bo'sh satr hisobga kirmaydi


# ─── 2026-09-17: Gemini 3 thought_signature (tool sikli 400 bilan yiqilardi) ──
#
# Jonli xato: ikkinchi iteratsiyada
#   400 INVALID_ARGUMENT … Function call is missing a thought_signature in
#   functionCall parts. This is required for tools to work correctly.
# Sabab: `resp.function_calls` tekislangan ro'yxat — imzo part'da qoladi va
# yo'qoladi. Quyidagi testlar imzoning **ikkala** yo'nalishda (javobdan
# ToolCall'ga va ToolCall'dan keyingi so'rov Content'iga) o'tishini qo'riqlaydi.

_SIG_A = b"sig-alpha"
_SIG_TXT = b"sig-text"


class _FakeChatModels:
    """`client.aio.models` o'rnini bosadi: `contents` ni yozib oladi."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.responses.pop(0)


def _gemini_chat(responses: list[Any]) -> tuple[Any, _FakeChatModels]:
    from app.llm.gemini import GeminiProvider

    provider = GeminiProvider(api_key="test-key")  # soxta kalit — tarmoqqa chiqmaydi
    fake = _FakeChatModels(responses)
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=fake))
    return provider, fake


def _resp(parts: list[Any], *, finish: str = "STOP") -> Any:
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts), finish_reason=finish)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10, candidates_token_count=2, cached_content_token_count=0
        ),
        # Ataylab bor: adapter bu tekislangan ro'yxatni **ishlatmasligi** kerak
        # (unda imzo yo'q). Imzosiz nusxa bilan to'ldiramiz.
        function_calls=[types.FunctionCall(id="c1", name="get_chat_stats", args={"days": 7})],
    )


async def test_parse_chat_keeps_thought_signature_from_part() -> None:
    parts = [
        types.Part(text="Statistikani olaman", thought_signature=_SIG_TXT),
        types.Part(
            function_call=types.FunctionCall(id="c1", name="get_chat_stats", args={"days": 7}),
            thought_signature=_SIG_A,
        ),
    ]
    provider, _ = _gemini_chat([_resp(parts)])
    res = await provider.chat(
        [Msg.user("bu hafta nechta post?")], model="gemini-3.5-flash", tools=[SEARCH_TOOL]
    )

    assert len(res.tool_calls) == 1
    call = res.tool_calls[0]
    assert (call.id, call.name, call.arguments) == ("c1", "get_chat_stats", {"days": 7})
    assert call.signature == _SIG_A
    assert res.text == "Statistikani olaman"
    assert res.signature == _SIG_TXT  # oxirgi matn part'idagi imzo ham saqlanadi
    await provider.aclose()


async def test_second_call_echoes_thought_signature_back() -> None:
    """Ikkinchi so'rovning `contents` ida aynan o'sha imzo bo'lishi shart."""
    first = _resp(
        [
            types.Part(
                function_call=types.FunctionCall(id="c1", name="get_chat_stats", args={"days": 7}),
                thought_signature=_SIG_A,
            )
        ]
    )
    second = _resp([types.Part(text="12 ta post.")])
    provider, fake = _gemini_chat([first, second])

    res1 = await provider.chat([Msg.user("q")], model="gemini-3.5-flash", tools=[SEARCH_TOOL])
    await provider.chat(
        [
            Msg.user("q"),
            Msg.assistant(res1.text, res1.tool_calls, signature=res1.signature),
            Msg.tool_result("c1", "get_chat_stats", '{"posts":12}'),
        ],
        model="gemini-3.5-flash",
        tools=[SEARCH_TOOL],
    )

    contents = fake.calls[1]["contents"]
    model_turn = next(c for c in contents if c.role == "model")
    fc_part = next(p for p in (model_turn.parts or []) if p.function_call is not None)
    assert fc_part.thought_signature == _SIG_A
    await provider.aclose()


def test_split_messages_echoes_text_signature() -> None:
    _, contents = _split_messages([Msg.assistant("javob", signature=_SIG_TXT)])
    part = (contents[0].parts or [])[0]
    assert part.text == "javob"
    assert part.thought_signature == _SIG_TXT


def test_split_messages_omits_signature_when_absent() -> None:
    """Imzosiz (boshqa provider / eski tarix) — maydon umuman yuborilmaydi."""
    _, contents = _split_messages(
        [Msg.assistant("javob", tool_calls=[ToolCall("c1", "f", {"a": 1})])]
    )
    parts = contents[0].parts or []
    assert all(p.thought_signature is None for p in parts)


def test_split_messages_groups_tool_results_of_one_turn() -> None:
    """Bir raunddagi parallel javoblar bitta `Content(role="user")` ichida."""
    _, contents = _split_messages(
        [
            Msg.user("q"),
            Msg.assistant(
                tool_calls=[ToolCall("c1", "f", {}), ToolCall("c2", "g", {})],
            ),
            Msg.tool_result("c1", "f", '{"a":1}'),
            Msg.tool_result("c2", "g", '{"b":2}'),
            Msg.user("davom et"),
        ]
    )
    roles = [c.role for c in contents]
    assert roles == ["user", "model", "user", "user"]
    tool_turn = contents[2]
    names = [p.function_response.name for p in (tool_turn.parts or [])]
    assert names == ["f", "g"]


# ─── boshqa adapterlar imzoni jimgina e'tiborsiz qoldiradi ───────────────────


async def test_deepseek_ignores_signature() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    p = make_provider(handler)
    await p.chat(
        [Msg.assistant(tool_calls=[ToolCall("c1", "f", {}, signature=_SIG_A)])],
        model="deepseek-chat",
    )
    assert "sig-alpha" not in json.dumps(seen)  # imzo tashqariga chiqmaydi
    await p.aclose()


def test_claude_ignores_signature() -> None:
    from app.llm.claude import split_messages

    msg = Msg.assistant(tool_calls=[ToolCall("c1", "f", {}, signature=_SIG_A)])
    _, blocks = split_messages([msg])
    assert "sig-alpha" not in json.dumps(blocks, default=str)
    assert blocks[0]["content"][0]["type"] == "tool_use"


# ─── 2026-09-18: model ro'yxatni bitta hujjat deb qabul qilsa ────────────────
#
# Jonli fakt: `embed_content(contents=[3 ta satr])` `gemini-embedding-001` da
# 3 ta, `gemini-embedding-2` da **bitta** vektor qaytaradi (SDK `t_contents`
# satrlar ro'yxatini bitta `UserContent(parts=[…])` ga yig'adi). Bu holatda
# chaqiruvchi hech narsa yozmaydi (`worker.embed_misaligned`), ya'ni indeks
# butunlay to'xtaydi — shuning uchun adapter pog'onadan pastga tushadi.


class _FakeCollapsing:
    """`client.aio.models` — SDK yig'ish xulqini taqlid qiladi.

    `content_aware=True`: har `types.Content` alohida hujjat (embedding-2 ning
    haqiqiy xulqi). `False`: nima yuborilsa ham bitta vektor — "haqiqiy
    single-only" modeli, per-matn chaqiruvdan boshqa yo'l yo'q.
    """

    def __init__(self, *, content_aware: bool = True, tokens: int = 5) -> None:
        self.content_aware = content_aware
        self.tokens = tokens
        self.calls: list[Any] = []
        self.inflight = 0
        self.max_inflight = 0

    def _count(self, contents: Any) -> int:
        if not isinstance(contents, list):
            return 1
        if all(isinstance(c, types.Content) for c in contents) and self.content_aware:
            return len(contents)
        return 1  # satrlar ro'yxati → bitta hujjat

    async def embed_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.calls.append(contents)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0)  # boshqa korutinlar navbatga tushsin
        self.inflight -= 1
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[1.0, 0.0]) for _ in range(self._count(contents))],
            metadata=SimpleNamespace(billable_character_count=self.tokens),
        )


def _gemini_fake(fake: Any) -> Any:
    from app.llm.gemini import GeminiProvider, reset_embed_mode_cache

    reset_embed_mode_cache()
    provider = GeminiProvider(api_key="test-key")  # soxta kalit — tarmoqqa chiqmaydi
    provider._client = SimpleNamespace(aio=SimpleNamespace(models=fake))
    return provider


def test_embed_batched_path_is_unchanged_for_healthy_models() -> None:
    """Ko'p vektor qaytaradigan modelda hech qanday qo'shimcha chaqiruv yo'q."""
    provider, fake = _gemini_with([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    from app.llm.gemini import reset_embed_mode_cache

    reset_embed_mode_cache()
    res = asyncio.run(provider.embed(["a", "b", "c"], model="gemini-embedding-001", dim=2))
    assert len(res.vectors) == 3
    assert len(fake.calls) == 1
    assert fake.calls[0]["contents"] == ["a", "b", "c"]  # satrlar — eski, arzon yo'l


def test_embed_falls_back_to_per_content_when_the_list_collapses() -> None:
    """Pog'ona 2: har matn alohida `Content` — hamon **bitta** so'rov."""
    fake = _FakeCollapsing(content_aware=True)
    provider = _gemini_fake(fake)
    res = asyncio.run(provider.embed(["a", "b", "c"], model="gemini-embedding-2", dim=2))

    assert len(res.vectors) == 3  # moslik tiklandi
    assert len(fake.calls) == 2  # satrlar (1 vektor) → Content'lar (3 vektor)
    assert all(isinstance(c, types.Content) for c in fake.calls[1])
    assert [c.parts[0].text for c in fake.calls[1]] == ["a", "b", "c"]


def test_embed_mode_is_cached_per_model_for_the_rest_of_the_process() -> None:
    """Aniqlash bir marta: keyingi batch'lar to'g'ridan-to'g'ri to'g'ri yo'ldan."""
    fake = _FakeCollapsing(content_aware=True)
    provider = _gemini_fake(fake)
    asyncio.run(provider.embed(["a", "b"], model="gemini-embedding-2", dim=2))
    fake.calls.clear()

    res = asyncio.run(provider.embed(["c", "d"], model="gemini-embedding-2", dim=2))
    assert len(res.vectors) == 2
    assert len(fake.calls) == 1  # ortiqcha "sinov" chaqiruvi yo'q
    assert all(isinstance(c, types.Content) for c in fake.calls[0])


def test_embed_falls_back_to_per_item_calls_and_sums_usage() -> None:
    """Pog'ona 3: `Content` ham yordam bermasa — har matn alohida so'rov."""
    fake = _FakeCollapsing(content_aware=False, tokens=5)
    provider = _gemini_fake(fake)
    res = asyncio.run(provider.embed(["a", "b", "c"], model="fake-embed-single", dim=2))

    assert len(res.vectors) == 3
    per_item = [c for c in fake.calls if c == ["a"] or c == ["b"] or c == ["c"]]
    assert len(per_item) == 3  # har matn bitta so'rov
    assert res.usage.tokens_in == 15  # 3 x 5 — taxminga tushilmadi


def test_embed_per_item_concurrency_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Semafor: 100 ta parallel RPC 429 (`RESOURCE_EXHAUSTED`) yeb qaytardi."""
    from app.llm import gemini as G

    fake = _FakeCollapsing(content_aware=False)
    provider = _gemini_fake(fake)
    monkeypatch.setattr(G, "get_settings", lambda: SimpleNamespace(embed_single_concurrency=2))
    res = asyncio.run(provider.embed([f"t{i}" for i in range(10)], model="fake-embed-sem", dim=2))

    assert len(res.vectors) == 10
    assert fake.max_inflight <= 2


def test_embed_single_text_needs_no_probing() -> None:
    """Bitta matnda pog'ona ma'nosiz — bitta chaqiruv, keshga yolg'on yozilmaydi."""
    from app.llm.gemini import _EMBED_BATCHED, _EMBED_MODE

    fake = _FakeCollapsing(content_aware=True)
    provider = _gemini_fake(fake)
    res = asyncio.run(provider.embed(["a"], model="gemini-embedding-2", dim=2))
    assert len(res.vectors) == 1 and len(fake.calls) == 1
    assert _EMBED_MODE["gemini-embedding-2"] == _EMBED_BATCHED


def test_embed_rejects_an_empty_text_list() -> None:
    provider, _ = _gemini_with([])
    with pytest.raises(LLMError):
        asyncio.run(provider.embed([], model="gemini-embedding-001", dim=2))
