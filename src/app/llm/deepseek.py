"""DeepSeek adapter — OpenAI-mos `/chat/completions` endpoint'i.

`openai` SDK o'rniga to'g'ridan-to'g'ri httpx: bizga kerak bo'lgan yuza juda
kichik (chat + tool calling), SDK versiya o'zgarishlariga bog'lanish esa
ortiqcha.

Modellar:
  * `deepseek-chat`     — V3 avlod. Function calling BOR.
  * `deepseek-reasoner` — R1 avlod. Reasoning kuchli, lekin **function calling
    yo'q**. Shuning uchun `capabilities()` da TOOLS berilmaydi va router
    tool kerak bo'lgan vazifada uni tanlamaydi.

Kontekst 64K — Gemini'ning 1M'iga nisbatan kichik. "Butun chat tarixini
kontekstga tashlash" strategiyasi DeepSeek'da ishlamaydi, retrieval majburiy.

Caching: DeepSeek avtomatik disk-cache ishlatadi (`prompt_cache_hit_tokens`),
boshqariladigan cache API yo'q — `EXPLICIT_CACHE` e'lon qilinmaydi.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import get_settings
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
    sanitize_json_schema,
)
from app.logging import get_logger

log = get_logger(__name__)

_TOOL_CAPABLE = frozenset({"deepseek-chat"})
_REASONER = frozenset({"deepseek-reasoner"})


class DeepSeekProvider(BaseProvider):
    name = "deepseek"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        s = get_settings()
        self._api_key = api_key or s.deepseek_api_key
        self._base_url = (base_url or s.deepseek_base_url).rstrip("/")
        # `client` — testlarda MockTransport ulash uchun
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={"Authorization": f"Bearer {self._api_key}"},
        )

    # ── imkoniyatlar ─────────────────────────────────────────────────────────

    def capabilities(self, model: str) -> frozenset[Capability]:
        caps = {Capability.CHAT, Capability.JSON_MODE}
        if model in _TOOL_CAPABLE:
            caps.add(Capability.TOOLS)
        # EMBED / IMAGE_GEN / LONG_CONTEXT / EXPLICIT_CACHE — DeepSeek'da yo'q
        return frozenset(caps)

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

        payload: dict[str, Any] = {
            "model": model,
            "messages": [_to_openai_msg(m) for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [_to_openai_tool(t) for t in tools]
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        # reasoner temperature'ni e'tiborsiz qoldiradi — yubormaymiz
        if model in _REASONER:
            payload.pop("temperature", None)

        data = await self._with_retry("chat", lambda: self._post("/chat/completions", payload))
        return _parse_chat(data, model=model)

    # ── ichki ────────────────────────────────────────────────────────────────

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMError(self.name, f"timeout: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(self.name, f"tarmoq xatosi: {exc}") from exc

        if resp.status_code >= 400:
            raise LLMError(
                self.name,
                f"HTTP {resp.status_code}: {resp.text[:300]}",
                status=resp.status_code,
            )
        try:
            return resp.json()  # type: ignore[no-any-return]
        except ValueError as exc:
            raise LLMError(self.name, f"JSON emas: {resp.text[:200]}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


# ─── konvertorlar ────────────────────────────────────────────────────────────


def _to_openai_msg(m: Msg) -> dict[str, Any]:
    if m.role is Role.TOOL:
        return {"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content}

    out: dict[str, Any] = {"role": m.role.value, "content": m.content}
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in m.tool_calls
        ]
        # tool_calls bo'lganda content bo'sh string bo'lishi kerak, None emas
        out["content"] = m.content or ""
    return out


def _to_openai_tool(t: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": sanitize_json_schema(t.parameters),
        },
    }


def _parse_chat(data: dict[str, Any], *, model: str) -> ChatResult:
    choices = data.get("choices") or []
    if not choices:
        raise LLMError("deepseek", f"bo'sh javob: {str(data)[:200]}")

    choice = choices[0]
    message = choice.get("message") or {}

    tool_calls: list[ToolCall] = []
    for raw in message.get("tool_calls") or []:
        fn = raw.get("function") or {}
        args_raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
        except json.JSONDecodeError:
            log.warning("llm.tool_args_invalid", provider="deepseek", raw=args_raw[:200])
            args = {}
        tool_calls.append(
            ToolCall(id=raw.get("id") or "", name=fn.get("name") or "", arguments=args)
        )

    u = data.get("usage") or {}
    return ChatResult(
        text=message.get("content") or "",
        tool_calls=tool_calls,
        model=data.get("model") or model,
        provider="deepseek",
        usage=Usage(
            tokens_in=int(u.get("prompt_tokens") or 0),
            tokens_out=int(u.get("completion_tokens") or 0),
            cached_tokens=int(u.get("prompt_cache_hit_tokens") or 0),
        ),
        finish_reason=choice.get("finish_reason") or "",
        reasoning=message.get("reasoning_content"),
    )
