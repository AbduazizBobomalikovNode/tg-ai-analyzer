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
import math
import warnings
from typing import Any

from google import genai
from google.genai import types

from app.config import get_settings
from app.llm.base import (
    EMBED_DOCUMENT,
    BaseProvider,
    BatchState,
    BatchStatus,
    Capability,
    CapabilityError,
    ChatResult,
    EmbedResult,
    EmbedTaskType,
    LLMError,
    Msg,
    Role,
    ToolCall,
    ToolSpec,
    Usage,
    estimate_tokens_heuristic,
    sanitize_json_schema,
)
from app.logging import get_logger

log = get_logger(__name__)

# 2.5 avlodning barcha modellari >= 1M kontekst
_LONG_CONTEXT_PREFIXES = ("gemini-2.5", "gemini-3", "gemini-1.5")

# phase4/B — `output_dimensionality` bilan **kesilgan** vektor endi birlik
# uzunlikda emas. `gemini-embedding-001` da normallashtirish mijoz tomonida
# qilinadi (Google hujjati shuni talab qiladi); `gemini-embedding-2` o'zi
# normallashtirib qaytaradi, ikkinchi marta bo'lish keraksiz.
#
# Nega muhim: pgvector'da cosine masofa normani hisobga oladi, lekin bizda
# `1 - cosine_distance` ostonalari (alert `threshold`, rerank) birlik vektor
# nazarda tutib sozlangan — normallashtirilmagan qator ular bilan mos kelmaydi.
NORMALIZE_CLIENT_SIDE = frozenset({"gemini-embedding-001"})


def _base_model(model: str) -> str:
    """`models/gemini-embedding-001` → `gemini-embedding-001`."""
    return model.split("/")[-1].strip().lower()


def l2_normalize(vec: list[float]) -> list[float]:
    """Birlik uzunlikka keltiradi; nol vektor o'zgarishsiz qaytadi."""
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


def normalize_vectors(model: str, vectors: list[list[float]]) -> list[list[float]]:
    """`NORMALIZE_CLIENT_SIDE` dagi modellar uchun L2 normallashtirish.

    Batch yo'li ham shu funksiyadan o'tishi kerak — aks holda jonli va batch
    yo'llari bir xil modelda ikki xil fazoga yozardi.
    """
    if _base_model(model) not in NORMALIZE_CLIENT_SIDE:
        return vectors
    return [l2_normalize(v) for v in vectors]


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        s = get_settings()
        self._client = genai.Client(api_key=api_key or s.gemini_api_key)

    # ── imkoniyatlar ─────────────────────────────────────────────────────────

    def capabilities(self, model: str) -> frozenset[Capability]:
        if "embedding" in model or "embed" in model:
            # BATCH_EMBED — `batches.create_embeddings` (4-bosqich/C, quyidagi
            # "Batch" bo'limi). SDK eski bo'lsa metod yo'q: `_batches()` aniq
            # `CapabilityError` beradi, jimgina ishlamay qolish yo'q.
            return frozenset({Capability.EMBED, Capability.BATCH_EMBED})

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

    async def embed(
        self,
        texts: list[str],
        *,
        model: str,
        dim: int,
        task_type: EmbedTaskType = EMBED_DOCUMENT,
    ) -> EmbedResult:
        self.require(model, Capability.EMBED)

        async def _call() -> Any:
            try:
                return await self._client.aio.models.embed_content(
                    model=model,
                    contents=texts,  # type: ignore[arg-type]
                    config=types.EmbedContentConfig(output_dimensionality=dim, task_type=task_type),
                )
            except Exception as exc:
                raise LLMError(self.name, str(exc), status=_status_of(exc)) from exc

        resp: Any = await self._with_retry("embed", _call)
        vectors = normalize_vectors(model, [list(e.values or []) for e in (resp.embeddings or [])])
        if not vectors:
            raise LLMError(self.name, "embedding bo'sh qaytdi")

        meta = getattr(resp, "metadata", None)
        tokens_in = int(getattr(meta, "billable_character_count", 0) or 0)
        if not tokens_in:
            # Javobda usage yo'q — narx 0 bo'lib byudjet qopqog'i jimgina
            # ishlamay qolmasin (`base.estimate_tokens_heuristic` izohiga qarang:
            # 2026-09-16 da tekshirilgan, `embed_content` metadata bermaydi).
            tokens_in = estimate_tokens_heuristic(texts)
        return EmbedResult(
            vectors=vectors,
            model=model,
            provider=self.name,
            usage=Usage(tokens_in=tokens_in),
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

    # ══ Batch embedding (4-bosqich/C) ════════════════════════════════════════
    #
    # SDK: `google-genai` >= 2.23. Uch qism (site-packages yo'llari — sinovda
    # tekshirilgan, `python:3.12-slim` + google-genai 2.23.0):
    #
    #   google/genai/batches.py:2960  AsyncBatches.create_embeddings(model, src, config)
    #       → HTTP `{model}:asyncBatchEmbedContent` (batches.py:~3040)
    #   google/genai/batches.py:827   _EmbedContentBatch_to_mldev — `contents`
    #       ro'yxati `requests[]` massiviga, `config` (task_type /
    #       output_dimensionality) esa **har bir** so'rovga ko'chiriladi.
    #       Ya'ni bitta `EmbedContentBatch` = N ta so'rov, bitta config.
    #   google/genai/types.py         BatchJobDestination.inlined_embed_content_responses:
    #       "The responses will be in the same order as the input requests."
    #
    # Tartib kafolati shu bitta jumlaga tayanadi — shuning uchun chaqiruvchi
    # (`services/embed_batches.py`) uzunlikni **qat'iy** tekshiradi va mos
    # kelmasa hech narsa yozmaydi (0-bosqich qoidasi).
    #
    # `create_embeddings` SDK'da `ExperimentalWarning` beradi — u har submit'da
    # log'ni ifloslantiradi, mazmuni esa bizga ma'lum; shuning uchun bostiriladi.

    def _batches(self) -> Any:
        b = getattr(self._client.aio, "batches", None)
        if b is None or not hasattr(b, "create_embeddings"):
            raise CapabilityError(
                self.name,
                "batch embedding uchun google-genai >= 2.23 kerak "
                "(`batches.create_embeddings` topilmadi)",
            )
        return b

    async def embed_batch_submit(
        self,
        texts: list[str],
        *,
        model: str,
        dim: int,
        task_type: EmbedTaskType = EMBED_DOCUMENT,
        tag: str = "",
    ) -> str:
        self.require(model, Capability.BATCH_EMBED)
        if not texts:
            raise LLMError(self.name, "batch embedding: bo'sh matn ro'yxati")
        batches = self._batches()

        src = types.EmbeddingsBatchJobSource(
            inlined_requests=types.EmbedContentBatch(
                contents=list(texts),  # type: ignore[arg-type]
                config=types.EmbedContentConfig(output_dimensionality=dim, task_type=task_type),
            )
        )
        config = (
            types.CreateEmbeddingsBatchJobConfig(display_name=tag[:120])
            if tag
            else types.CreateEmbeddingsBatchJobConfig()
        )

        async def _call() -> Any:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return await batches.create_embeddings(model=model, src=src, config=config)
            except Exception as exc:
                raise LLMError(self.name, str(exc), status=_status_of(exc)) from exc

        job: Any = await self._with_retry("embed_batch_submit", _call)
        name = str(getattr(job, "name", "") or "")
        if not name:
            raise LLMError(self.name, "batch job yaratildi, lekin nomi qaytmadi")
        log.info("gemini.batch.submitted", job=name, items=len(texts), model=model, tag=tag)
        return name

    async def embed_batch_poll(self, job: str) -> BatchStatus:
        resp = await self._batch_get(job, "embed_batch_poll")
        return _batch_status(resp, job)

    async def embed_batch_fetch(self, job: str, *, model: str = "") -> EmbedResult:
        resp = await self._batch_get(job, "embed_batch_fetch")
        status = _batch_status(resp, job)
        if status.state is not BatchState.SUCCEEDED:
            raise LLMError(
                self.name,
                f"batch {job} tugamagan/muvaffaqiyatsiz: {status.raw_state} {status.error}",
            )

        dest = getattr(resp, "dest", None)
        responses = getattr(dest, "inlined_embed_content_responses", None)
        if responses is None:
            # Faylli chiqish faqat faylli kirishda bo'ladi; biz doim inline
            # yuboramiz, shuning uchun bu yerga tushish = SDK/API shakli
            # o'zgargan. Jimgina bo'sh qaytarish o'rniga aniq xato.
            raise LLMError(
                self.name,
                f"batch {job}: inline javob yo'q "
                f"(dest.file_name={getattr(dest, 'file_name', None)!r})",
            )

        vectors: list[list[float]] = []
        tokens = 0
        for i, item in enumerate(responses):
            err = getattr(item, "error", None)
            if err is not None:
                raise LLMError(
                    self.name,
                    f"batch {job}: #{i} elementda xato: {getattr(err, 'message', err)}",
                )
            single = getattr(item, "response", None)
            emb = getattr(single, "embedding", None)
            values = list(getattr(emb, "values", None) or [])
            if not values:
                raise LLMError(self.name, f"batch {job}: #{i} elementda vektor bo'sh")
            vectors.append(values)
            tokens += int(getattr(single, "token_count", 0) or 0)
        # `tokens` 0 bo'lib qolishi mumkin (javobda usage yo'q — 2026-09-16 da
        # tekshirilgan). Bu yerda **taxmin qilinmaydi**: original matnlar bu
        # funksiyada yo'q (faqat `job` nomi), soxta uzunlik ishlatish noto'g'ri
        # taxminga olib kelardi. Chaqiruvchi (`services/embed_batches.py::_poll_one`)
        # 0 ko'rsa `estimate_tokens_heuristic` bilan **jami** narxni tuzatadi —
        # bu yerda faqat javobdagi haqiqiy son.

        # Sinxron yo'l bilan **bir xil** normallashtirish (phase4/B,
        # `normalize_vectors`) — aks holda bitta modelning batch va jonli
        # qatorlari `message_embeddings` da ikki xil masshtabda aralashardi
        # va cosine ostonalari (alert `threshold`, rerank) buzilardi.
        #
        # H2: javobda `model` maydoni bo'lmasligi mumkin (API uni har doim
        # qaytarmaydi). Unda **so'ralgan** nom ishlatiladi — `status.model`
        # bo'sh bo'lganda `normalize_vectors("")` normallashtirishni jimgina
        # o'tkazib yuborardi va chaqiruvchi `model=''` qatorini yozardi.
        effective = status.model or _base_model(model)
        return EmbedResult(
            vectors=normalize_vectors(effective, vectors),
            model=effective,
            provider=self.name,
            usage=Usage(tokens_in=tokens),
        )

    async def _batch_get(self, job: str, op: str) -> Any:
        batches = self._batches()

        async def _call() -> Any:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return await batches.get(name=job)
            except Exception as exc:
                raise LLMError(self.name, str(exc), status=_status_of(exc)) from exc

        return await self._with_retry(op, _call)


# ─── konvertorlar ────────────────────────────────────────────────────────────


def _to_declaration(t: ToolSpec) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=t.name,
        description=t.description,
        parameters_json_schema=sanitize_json_schema(t.parameters),
    )


def _split_messages(messages: list[Msg]) -> tuple[str, list[types.Content]]:
    """SYSTEM'ni ajratadi, qolganini Gemini Content'lariga o'giradi.

    Ikki nozik joy (Gemini 3):

    1. **thought_signature qaytariladi.** Modelning `function_call` part'i bilan
       kelgan opaque imzo aynan o'sha part'ga qayta biriktiriladi. Usiz ikkinchi
       turn `400 INVALID_ARGUMENT … Function call is missing a thought_signature
       in functionCall parts` bilan yiqiladi — ya'ni tool chaqirgan **har**
       agent yugurishi. Matn part'idagi imzo ham (`Msg.signature`) shu qoida
       bilan qaytariladi.
    2. **Bir turn'dagi tool javoblari bitta Content'ga yig'iladi.** Model bir
       raundda bir nechta funksiya chaqirsa, API javoblarni bitta
       `Content(role="user")` ning parallel `function_response` part'lari
       sifatida kutadi (SDK: `Content.parts` — "ordered parts that constitute
       a single message"). Ilgari har `Msg.tool_result` alohida Content bo'lardi.
    """
    system_parts: list[str] = []
    contents: list[types.Content] = []
    pending_tool: list[types.Part] = []  # bitta turn'dagi function_response'lar

    def flush_tools() -> None:
        if pending_tool:
            contents.append(types.Content(role="user", parts=list(pending_tool)))
            pending_tool.clear()

    for m in messages:
        if m.role is Role.SYSTEM:
            if m.content:
                system_parts.append(m.content)
            continue

        if m.role is Role.TOOL:
            pending_tool.append(
                types.Part.from_function_response(
                    name=m.name or "tool", response=_as_json_object(m.content)
                )
            )
            continue

        flush_tools()

        parts: list[types.Part] = []
        if m.content:
            parts.append(_text_part(m.content, m.signature))
        for tc in m.tool_calls:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(id=tc.id, name=tc.name, args=tc.arguments),
                    **_sig_kwargs(tc.signature),
                )
            )
        if not parts:
            continue

        contents.append(
            types.Content(role="model" if m.role is Role.ASSISTANT else "user", parts=parts)
        )

    flush_tools()
    return "\n\n".join(system_parts), contents


def _sig_kwargs(signature: bytes | str | None) -> dict[str, Any]:
    """`thought_signature=None` ni **yubormaslik** uchun kwargs.

    `types.Part.__init__` faqat `None` bo'lmagan maydonlarni to'playdi, lekin
    aniqlik uchun shu yerda ham kesamiz — signature'siz eski/boshqa provider
    tarixida maydon umuman paydo bo'lmaydi.
    """
    return {"thought_signature": signature} if signature else {}


def _text_part(text: str, signature: bytes | str | None = None) -> types.Part:
    if signature:
        return types.Part(text=text, thought_signature=signature)  # type: ignore[arg-type]
    return types.Part.from_text(text=text)


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
    # Part'lar bo'yicha yuramiz, `resp.function_calls` emas: u tekislangan
    # ro'yxat va `thought_signature` ni **yo'qotadi** (u part'da, FunctionCall
    # ichida emas). Imzosiz keyingi turn 400 bilan yiqiladi — CLAUDE.md,
    # `ToolCall.signature`.
    tool_calls: list[ToolCall] = []
    text_chunks: list[str] = []
    text_signature: bytes | str | None = None

    for p in _parts_of(resp):
        sig = getattr(p, "thought_signature", None)
        fc = getattr(p, "function_call", None)
        if fc is not None:
            tool_calls.append(
                ToolCall(
                    id=getattr(fc, "id", None) or "",
                    name=getattr(fc, "name", None) or "",
                    arguments=dict(getattr(fc, "args", None) or {}),
                    signature=sig,
                )
            )
            continue
        # `.text` tool_call bo'lganda istisno tashlashi mumkin — qo'lda yig'amiz
        if getattr(p, "text", None) and not getattr(p, "thought", False):
            text_chunks.append(p.text)
            if sig:
                text_signature = sig

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
        signature=text_signature,
    )


# `types.JobState` (google/genai/types.py) → normallashtirilgan holat.
# QUEUED/PENDING/UPDATING — hali ishlamagan; PAUSED ham kutish (o'zi tiklanadi).
# PARTIALLY_SUCCEEDED — biz uchun **failed**: yarim to'plamni yozib bo'lmaydi,
# chunki qaysi element tushib qolgani tartibdan bilinmaydi (invariant: qat'iy
# moslik). Bunday job qayta yuboriladi.
_BATCH_STATES: dict[str, BatchState] = {
    "JOB_STATE_UNSPECIFIED": BatchState.PENDING,
    "JOB_STATE_QUEUED": BatchState.PENDING,
    "JOB_STATE_PENDING": BatchState.PENDING,
    "JOB_STATE_UPDATING": BatchState.RUNNING,
    "JOB_STATE_PAUSED": BatchState.RUNNING,
    "JOB_STATE_RUNNING": BatchState.RUNNING,
    "JOB_STATE_SUCCEEDED": BatchState.SUCCEEDED,
    "JOB_STATE_PARTIALLY_SUCCEEDED": BatchState.FAILED,
    "JOB_STATE_FAILED": BatchState.FAILED,
    "JOB_STATE_CANCELLING": BatchState.RUNNING,
    "JOB_STATE_CANCELLED": BatchState.CANCELLED,
    "JOB_STATE_EXPIRED": BatchState.EXPIRED,
}


def _batch_model(resp: Any) -> str:
    """Javobdagi `model` maydoni — **faqat** satr bo'lsa qabul qilinadi.

    SDK bu maydonni umuman bermasligi yoki obyekt berishi mumkin. `str(...)`
    bilan majburlash `<google.genai.types.Model object at 0x…>` kabi axlatni
    `message_embeddings.model` ga yozib qo'yardi — bunday qator qidiruv
    filtridan hech qachon o'tmaydi (H2). Shubhali qiymat = bo'sh satr, ya'ni
    chaqiruvchi so'ralgan model nomiga qaytadi.
    """
    value = getattr(resp, "model", None)
    if not isinstance(value, str):
        return ""
    return _base_model(value)


def _batch_status(resp: Any, job: str) -> BatchStatus:
    raw = getattr(resp, "state", None)
    raw_name = str(getattr(raw, "value", None) or getattr(raw, "name", None) or raw or "")
    err = getattr(resp, "error", None)
    return BatchStatus(
        job=str(getattr(resp, "name", None) or job),
        # Noma'lum holat → RUNNING: "tugadi" deb hisoblash xavfliroq
        # (elementlar qayta yuborilmay muzlab qolardi).
        state=_BATCH_STATES.get(raw_name.upper(), BatchState.RUNNING),
        raw_state=raw_name,
        # Narx qoidasi: API **qaytargan** nom ("models/…" prefiksisiz).
        model=_batch_model(resp),
        error=str(getattr(err, "message", "") or "") if err is not None else "",
    )


def _status_of(exc: Exception) -> int | None:
    for attr in ("code", "status_code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    return None
