"""Claude adapter (`anthropic` SDK).

Asosiy maqsad — foydalanuvchi **Claude Code obunasi** (yoki `ant auth login`)
kredensiali bilan alohida API kalit ulamasdan ishlay olishi. Kredensial
manbalari, ustuvorlik tartibida:

  1. `ANTHROPIC_API_KEY`        — oddiy API kalit (x-api-key).
  2. `CLAUDE_CODE_OAUTH_TOKEN`  — `claude setup-token` chiqargan uzoq muddatli
     OAuth token; yoki `ANTHROPIC_AUTH_TOKEN`. Bearer + `oauth-2025-04-20` beta
     header bilan yuboriladi.
  3. Diskdagi profil            — `ant auth login` / Claude Code login yozgan
     `~/.config/anthropic/` (yoki `ANTHROPIC_CONFIG_DIR`). SDK'ning o'zi
     topadi va tokenni yangilab turadi.

Hech biri topilmasa `detect_claude_auth()` `None` qaytaradi — router `auto`
rejimda Gemini'ni tanlaydi. Bu adapter Claude'ni "bor bo'lsa foydalan"
tamoyilida qo'shadi, mavjud Gemini/DeepSeek yo'llari o'zgarmaydi.

Rollarni moslash:
  SYSTEM    → `system` parametri (bir nechta bo'lsa birlashtiriladi)
  USER      → {"role": "user", "content": text}
  ASSISTANT → {"role": "assistant", "content": [text, tool_use...]}
  TOOL      → {"role": "user", "content": [tool_result...]}  (ketma-ketlari
              bitta user xabariga yig'iladi — parallel tool chaqiruvlar uchun)

Imkoniyatlar: chat, tools, json_mode (system ko'rsatma orqali), uzun kontekst,
prompt caching. **Embedding va rasm generatsiya yo'q** → router Gemini'ga
tushiradi (`GEMINI_API_KEY` baribir kerak).

Sirlar: token/kalit hech qachon log'ga tushmaydi — faqat manba nomi
(`ANTHROPIC_API_KEY`, `CLAUDE_CODE_OAUTH_TOKEN`, `profile`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.llm.base import (
    BaseProvider,
    Capability,
    ChatResult,
    LLMError,
    Msg,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
)
from app.logging import get_logger

log = get_logger(__name__)

# OAuth (Bearer) token bilan /v1/messages uchun majburiy beta flag.
OAUTH_BETA_HEADER = "oauth-2025-04-20"

# Bu modellarda `temperature`/`top_p`/`top_k` olib tashlangan — yuborilsa 400.
_NO_SAMPLING_PREFIXES = (
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
)

# max_tokens Claude API'da majburiy. Non-streaming uchun tavsiya etilgan default.
_DEFAULT_MAX_TOKENS = 16000

_JSON_INSTRUCTION = (
    "Respond with a single valid JSON value only. "
    "No markdown code fences, no prose before or after the JSON."
)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


# ─── kredensial aniqlash ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ClaudeAuth:
    """Topilgan kredensial. `secret` hech qachon log'ga chiqmaydi."""

    kind: str  # "api_key" | "oauth_token" | "profile"
    source: str  # log uchun: env nomi yoki "profile"
    secret: str = ""

    def __repr__(self) -> str:  # tasodifan print qilinsa ham sir chiqmasin
        return f"ClaudeAuth(kind={self.kind!r}, source={self.source!r})"


def detect_claude_auth(settings: Settings | None = None) -> ClaudeAuth | None:
    """Claude uchun ishlatsa bo'ladigan kredensial bormi?

    Tarmoqqa chiqmaydi. Profil tekshiruvi SDK'ning o'z zanjiri orqali
    (`anthropic.Anthropic().credentials`) — biz uning ichki formatiga
    bog'lanmaymiz.
    """
    s = settings or get_settings()

    if s.anthropic_api_key:
        return ClaudeAuth("api_key", "ANTHROPIC_API_KEY", s.anthropic_api_key)
    if s.claude_code_oauth_token:
        return ClaudeAuth("oauth_token", "CLAUDE_CODE_OAUTH_TOKEN", s.claude_code_oauth_token)
    if s.anthropic_auth_token:
        return ClaudeAuth("oauth_token", "ANTHROPIC_AUTH_TOKEN", s.anthropic_auth_token)

    try:
        import anthropic

        probe = anthropic.Anthropic(max_retries=0)
        try:
            if probe.credentials is not None or probe.api_key or probe.auth_token:
                return ClaudeAuth("profile", "profile")
        finally:
            probe.close()
    except Exception as exc:  # SDK yo'q / profil buzilgan — Claude yo'q deb hisoblaymiz
        log.debug("llm.claude.probe_failed", error=str(exc))
    return None


# ─── provider ────────────────────────────────────────────────────────────────


class ClaudeProvider(BaseProvider):
    name = "claude"

    def __init__(
        self,
        auth: ClaudeAuth | None = None,
        *,
        timeout: float = 300.0,
        client: Any | None = None,
    ) -> None:
        s = get_settings()
        self._auth = auth or detect_claude_auth(s)
        # `client` — testlarda soxta client ulash uchun
        self._client = client or (self._build_client(s, timeout) if self._auth else None)

    def _build_client(self, s: Settings, timeout: float) -> Any:
        import anthropic

        assert self._auth is not None
        kw: dict[str, Any] = {
            # SDK'ning o'z retry'si o'chiriladi — BaseProvider._with_retry bor,
            # ikkalasi birga retry-storm beradi.
            "max_retries": 0,
            "timeout": timeout,
        }
        if self._auth.kind == "api_key":
            kw["api_key"] = self._auth.secret
        elif self._auth.kind == "oauth_token":
            kw["auth_token"] = self._auth.secret
            # Statik Bearer token bilan SDK beta header'ni o'zi qo'ymaydi.
            kw["default_headers"] = {"anthropic-beta": OAUTH_BETA_HEADER}
        # kind == "profile": argumentsiz — SDK profilni o'zi topadi va yangilaydi
        if s.anthropic_base_url:
            kw["base_url"] = s.anthropic_base_url

        log.info("llm.claude.auth", source=self._auth.source, kind=self._auth.kind)
        return anthropic.AsyncAnthropic(**kw)

    @property
    def auth(self) -> ClaudeAuth | None:
        return self._auth

    # ── imkoniyatlar ─────────────────────────────────────────────────────────

    def capabilities(self, model: str) -> frozenset[Capability]:
        if not model.startswith("claude"):
            return frozenset()
        # EMBED / IMAGE_GEN — Claude'da yo'q, router Gemini'ga tushiradi
        return frozenset(
            {
                Capability.CHAT,
                Capability.TOOLS,
                Capability.JSON_MODE,
                Capability.LONG_CONTEXT,  # barcha joriy modellar >= 200K
                Capability.EXPLICIT_CACHE,  # cache_control
            }
        )

    # ── chat ─────────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Msg],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        needed = [Capability.CHAT]
        if tools:
            needed.append(Capability.TOOLS)
        if json_mode:
            needed.append(Capability.JSON_MODE)
        self.require(model, *needed)

        if self._client is None:
            raise LLMError(
                self.name,
                "kredensial topilmadi. ANTHROPIC_API_KEY yoki CLAUDE_CODE_OAUTH_TOKEN "
                "(`claude setup-token`) bering, yoki `ant auth login` qiling.",
                status=401,
            )

        payload = build_request(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

        client = self._client

        async def _call() -> Any:
            try:
                return await client.messages.create(**payload)
            except Exception as exc:
                raise LLMError(self.name, _describe(exc), status=_status_of(exc)) from exc

        resp: Any = await self._with_retry("chat", _call)
        return parse_response(resp, model=model, json_mode=json_mode)

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()


# ─── konvertorlar ────────────────────────────────────────────────────────────


def build_request(
    messages: list[Msg],
    *,
    model: str,
    tools: list[ToolSpec] | None,
    temperature: float,
    max_tokens: int | None,
    json_mode: bool,
) -> dict[str, Any]:
    """`messages.create(**payload)` uchun tayyor lug'at. Sof funksiya — testlanadi."""
    system_text, converted = split_messages(messages)
    if json_mode:
        system_text = f"{system_text}\n\n{_JSON_INSTRUCTION}".strip()

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
        "messages": converted,
    }
    if system_text:
        payload["system"] = system_text
    if tools:
        payload["tools"] = [_to_tool(t) for t in tools]
    if not model.startswith(_NO_SAMPLING_PREFIXES):
        payload["temperature"] = temperature
    return payload


def split_messages(messages: list[Msg]) -> tuple[str, list[dict[str, Any]]]:
    """SYSTEM'ni ajratadi, qolganini Claude `messages` ro'yxatiga o'giradi."""
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    for m in messages:
        if m.role is Role.SYSTEM:
            if m.content:
                system_parts.append(m.content)
            continue

        if m.role is Role.TOOL:
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id or "",
                "content": m.content,
            }
            # ketma-ket tool natijalari — bitta user xabariga (parallel chaqiruvlar)
            if out and out[-1]["role"] == "user" and _is_tool_result_msg(out[-1]):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue

        if m.role is Role.ASSISTANT:
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments}
                )
            if blocks:
                out.append({"role": "assistant", "content": blocks})
            continue

        if m.content:
            out.append({"role": "user", "content": m.content})

    return "\n\n".join(system_parts), out


def _is_tool_result_msg(msg: dict[str, Any]) -> bool:
    content = msg.get("content")
    return (
        isinstance(content, list)
        and bool(content)
        and all(b.get("type") == "tool_result" for b in content)
    )


def _to_tool(t: ToolSpec) -> dict[str, Any]:
    # Claude to'liq JSON Schema qabul qiladi — Gemini'dagi kabi tozalash shart emas
    return {"name": t.name, "description": t.description, "input_schema": t.parameters}


def parse_response(resp: Any, *, model: str, json_mode: bool = False) -> ChatResult:
    stop = str(getattr(resp, "stop_reason", "") or "")
    if stop == "refusal":
        # Xavfsizlik klassifikatori rad etdi — content bo'sh yoki qisman.
        # Bu xato emas, javob; qayta urinish foydasiz.
        log.warning("llm.claude.refusal", model=model)
        return ChatResult(
            text="",
            tool_calls=[],
            model=str(getattr(resp, "model", "") or model),
            provider="claude",
            usage=_usage_of(resp),
            finish_reason="refusal",
        )

    text_chunks: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in getattr(resp, "content", None) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_chunks.append(getattr(block, "text", "") or "")
        elif btype == "tool_use":
            raw_input = getattr(block, "input", None) or {}
            try:
                args = dict(raw_input) if not isinstance(raw_input, str) else json.loads(raw_input)
            except (json.JSONDecodeError, TypeError, ValueError):
                log.warning("llm.tool_args_invalid", provider="claude", raw=str(raw_input)[:200])
                args = {}
            tool_calls.append(
                ToolCall(
                    id=getattr(block, "id", "") or "",
                    name=getattr(block, "name", "") or "",
                    arguments=args,
                )
            )
        # thinking bloklari — matnga qo'shilmaydi

    text = "".join(text_chunks)
    if json_mode:
        text = strip_json_fences(text)

    return ChatResult(
        text=text,
        tool_calls=tool_calls,
        model=str(getattr(resp, "model", "") or model),
        provider="claude",
        usage=_usage_of(resp),
        finish_reason=stop,
    )


def strip_json_fences(text: str) -> str:
    """Model ko'rsatmaga qaramay ```json ... ``` qo'ysa — ichini oladi."""
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text.strip()


def _usage_of(resp: Any) -> Usage:
    u = getattr(resp, "usage", None)
    return Usage(
        tokens_in=int(getattr(u, "input_tokens", 0) or 0),
        tokens_out=int(getattr(u, "output_tokens", 0) or 0),
        cached_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
    )


def _status_of(exc: Exception) -> int | None:
    val = getattr(exc, "status_code", None)
    return val if isinstance(val, int) else None


def _describe(exc: Exception) -> str:
    status = _status_of(exc)
    msg = getattr(exc, "message", None) or str(exc)
    if status == 401:
        return f"HTTP 401 — kredensial rad etildi ({msg[:200]})"
    if status is not None:
        return f"HTTP {status}: {msg[:300]}"
    return f"tarmoq xatosi: {msg[:300]}"
