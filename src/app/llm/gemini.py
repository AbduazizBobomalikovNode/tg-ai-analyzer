"""Gemini adapter (`google-genai` SDK).

Loyihada Gemini yagona to'liq imkoniyatli provider: chat, tool calling,
embedding, rasm generatsiya, uzun kontekst va boshqariladigan context caching —
hammasi bor. DeepSeek yoqilganda ham embedding/rasm shu yerda qoladi.

Rollarni moslash:
  SYSTEM    → `config.system_instruction` (alohida Content emas)
  USER      → Content(role="user")
  ASSISTANT → Content(role="model")  [+ function_call part'lari]
  TOOL      → Content(role="user")  [Part.from_function_response]
"""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai import types

from app.config import get_settings
from app.llm.base import (
    BaseProvider,
    Capability,
    ChatResult,
    EmbedResult,
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

# 2.5 avlodning barcha modellari >= 1M kontekst
_LONG_CONTEXT_PREFIXES = ("gemini-2.5", "gemini-3", "gemini-1.5")


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        s = get_settings()
        self._client = genai.Client(api_key=api_key or s.gemini_api_key)

    # ── imkoniyatlar ─────────────────────────────────────────────────────────

    def capabilities(self, model: str) -> frozenset[Capability]:
        if "embedding" in model or "embed" in model:
            return frozenset({Capability.EMBED})

        caps = {
            Capability.CHAT,
            Capability.TOOLS,
            Capability.JSON_MODE,
            Capability.EXPLICIT_CACHE,
        }
        if "image" in model:
            caps.add(Capability.IMAGE_GEN)
        if model.startswith(_LONG_CONTEXT_PREFIXES):
            caps.add(Capability.LONG_CONTEXT)
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

        system_text, contents = _split_messages(messages)

        cfg_kwargs: dict[str, Any] = {"temperature": temperature}
        if system_text:
            cfg_kwargs["system_instruction"] = system_text
        if max_tokens:
            cfg_kwargs["max_output_tokens"] = max_tokens
        if json_mode:
            cfg_kwargs["response_mime_type"] = "application/json"
        if tools:
            cfg_kwargs["tools"] = [
                types.Tool(function_declarations=[_to_declaration(t) for t in tools])
            ]
            # Tool'larni biz o'zimiz bajaramiz — SDK avtomatik chaqirmasin.
            cfg_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        config = types.GenerateContentConfig(**cfg_kwargs)

        async def _call() -> Any:
            try:
                return await self._client.aio.models.generate_content(
                    model=model,
                    contents=contents,  # type: ignore[arg-type]
                    config=config,
                )
            except Exception as exc:
                raise LLMError(self.name, str(exc), status=_status_of(exc)) from exc

        resp: Any = await self._with_retry("chat", _call)
        return _parse_chat(resp, model=model)

    # ── embedding ────────────────────────────────────────────────────────────

    async def embed(self, texts: list[str], *, model: str, dim: int) -> EmbedResult:
        self.require(model, Capability.EMBED)

        async def _call() -> Any:
            try:
                return await self._client.aio.models.embed_content(
                    model=model,
                    contents=texts,  # type: ignore[arg-type]
                    config=types.EmbedContentConfig(output_dimensionality=dim),
                )
            except Exception as exc:
                raise LLMError(self.name, str(exc), status=_status_of(exc)) from exc

        resp: Any = await self._with_retry("embed", _call)
        vectors = [list(e.values or []) for e in (resp.embeddings or [])]
        if not vectors:
            raise LLMError(self.name, "embedding bo'sh qaytdi")

        meta = getattr(resp, "metadata", None)
        return EmbedResult(
            vectors=vectors,
            model=model,
            provider=self.name,
            usage=Usage(tokens_in=int(getattr(meta, "billable_character_count", 0) or 0)),
        )

    # ── rasm ─────────────────────────────────────────────────────────────────

    async def generate_image(self, prompt: str, *, model: str) -> bytes:
        self.require(model, Capability.IMAGE_GEN)

        async def _call() -> Any:
            try:
                return await self._client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
                )
            except Exception as exc:
                raise LLMError(self.name, str(exc), status=_status_of(exc)) from exc

        resp: Any = await self._with_retry("image", _call)
        for part in _parts_of(resp):
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return bytes(inline.data)
        raise LLMError(self.name, "javobda rasm topilmadi")


# ─── konvertorlar ────────────────────────────────────────────────────────────


def _to_declaration(t: ToolSpec) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=t.name,
        description=t.description,
        parameters_json_schema=sanitize_json_schema(t.parameters),
    )


def _split_messages(messages: list[Msg]) -> tuple[str, list[types.Content]]:
    """SYSTEM'ni ajratadi, qolganini Gemini Content'lariga o'giradi."""
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for m in messages:
        if m.role is Role.SYSTEM:
            if m.content:
                system_parts.append(m.content)
            continue

        if m.role is Role.TOOL:
            payload = _as_json_object(m.content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(name=m.name or "tool", response=payload)
                    ],
                )
            )
            continue

        parts: list[types.Part] = []
        if m.content:
            parts.append(types.Part.from_text(text=m.content))
        for tc in m.tool_calls:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(id=tc.id, name=tc.name, args=tc.arguments)
                )
            )
        if not parts:
            continue

        contents.append(
            types.Content(role="model" if m.role is Role.ASSISTANT else "user", parts=parts)
        )

    return "\n\n".join(system_parts), contents


def _as_json_object(raw: str) -> dict[str, Any]:
    """Gemini function response'ni obyekt sifatida kutadi."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"result": raw}
    return parsed if isinstance(parsed, dict) else {"result": parsed}


def _parts_of(resp: Any) -> list[Any]:
    for cand in getattr(resp, "candidates", None) or []:
        content = getattr(cand, "content", None)
        if content is not None:
            return list(getattr(content, "parts", None) or [])
    return []


def _parse_chat(resp: Any, *, model: str) -> ChatResult:
    tool_calls = [
        ToolCall(id=fc.id or "", name=fc.name or "", arguments=dict(fc.args or {}))
        for fc in (getattr(resp, "function_calls", None) or [])
    ]

    # `.text` tool_call bo'lganda istisno tashlashi mumkin — qo'lda yig'amiz
    text_chunks = [
        p.text
        for p in _parts_of(resp)
        if getattr(p, "text", None) and not getattr(p, "thought", False)
    ]

    u = getattr(resp, "usage_metadata", None)
    finish = ""
    for cand in getattr(resp, "candidates", None) or []:
        finish = str(getattr(cand, "finish_reason", "") or "")
        break

    return ChatResult(
        text="".join(text_chunks),
        tool_calls=tool_calls,
        model=model,
        provider="gemini",
        usage=Usage(
            tokens_in=int(getattr(u, "prompt_token_count", 0) or 0),
            tokens_out=int(getattr(u, "candidates_token_count", 0) or 0),
            cached_tokens=int(getattr(u, "cached_content_token_count", 0) or 0),
        ),
        finish_reason=finish,
    )


def _status_of(exc: Exception) -> int | None:
    for attr in ("code", "status_code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None
