"""Provider-agnostik LLM qatlami.

Loyiha Gemini va DeepSeek bilan ishlaydi. Ular teng imkoniyatli emas —
DeepSeek'da embedding ham, rasm generatsiya ham yo'q. Shuning uchun har bir
provider o'z `capabilities()` ini e'lon qiladi va `app.llm.router` vazifaga
mos providerni tanlaydi; imkoniyat yetishmasa fallback'ga o'tadi.

Adapterlar "eng kichik umumiy maxraj" emas: har bir provider o'z kuchli
tomonini saqlaydi (Gemini — uzun kontekst/embedding/rasm; DeepSeek — arzon
reasoning), router esa qaysi biri qayerda ishlashini hal qiladi.
"""

from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from app.logging import get_logger

log = get_logger(__name__)

#: 4 belgi ≈ 1 token — `services/images.py::_CHARS_PER_TOKEN` bilan bir xil
#: taxmin. Ba'zi provider javoblari (Gemini `embed_content`, sinxron ham
#: batch ham) usage/token hisoblagichini umuman bermaydi; shu holatda
#: `tokens_in` 0 bo'lib qolsa `pricing.estimate_cost` ham 0 qaytaradi va
#: kunlik xarajat limiti (`llm_daily_cost_budget_usd`) o'sha sarfni
#: **umuman ko'rmaydi** — jimgina ishlamay qolish. Aniq son uchun alohida
#: `count_tokens` RPC kerak bo'lardi (qo'shimcha kechikish/xarajat har
#: chaqiruvda); o'rniga hujjatlashtirilgan taxmin — provider-agnostik,
#: shuning uchun shu yerda (`app.llm.gemini` va `services.embed_batches`
#: ikkalasi ham shu funksiyadan foydalanadi).
_CHARS_PER_TOKEN = 4


def estimate_tokens_heuristic(texts: list[str]) -> int:
    """`ceil(len(text)/4)` — har matn uchun, yig'indi. Bo'sh matn → 0 token."""
    return sum(-(-len(t) // _CHARS_PER_TOKEN) for t in texts if t)


class Capability(StrEnum):
    CHAT = "chat"
    TOOLS = "tools"  # function calling
    JSON_MODE = "json_mode"
    EMBED = "embed"
    BATCH_EMBED = "batch_embed"  # asinxron (arzon) embedding batch job
    IMAGE_GEN = "image_gen"
    LONG_CONTEXT = "long_context"  # >= 200k token
    EXPLICIT_CACHE = "explicit_cache"  # boshqariladigan context caching


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Msg:
    role: Role
    content: str = ""
    # assistant xabari tool chaqirsa
    tool_calls: list[ToolCall] = field(default_factory=list)
    # role=TOOL bo'lganda: qaysi chaqiruvga javob
    tool_call_id: str | None = None
    name: str | None = None

    @classmethod
    def system(cls, text: str) -> Msg:
        return cls(Role.SYSTEM, text)

    @classmethod
    def user(cls, text: str) -> Msg:
        return cls(Role.USER, text)

    @classmethod
    def assistant(cls, text: str = "", tool_calls: list[ToolCall] | None = None) -> Msg:
        return cls(Role.ASSISTANT, text, tool_calls or [])

    @classmethod
    def tool_result(cls, call_id: str, name: str, content: str) -> Msg:
        return cls(Role.TOOL, content, tool_call_id=call_id, name=name)


@dataclass(slots=True)
class ToolSpec:
    """JSON Schema ko'rinishidagi tool ta'rifi (OpenAI/Gemini uchun umumiy)."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0


@dataclass(slots=True)
class ChatResult:
    text: str
    tool_calls: list[ToolCall]
    model: str
    provider: str
    usage: Usage
    finish_reason: str = ""
    # DeepSeek reasoner'ning `reasoning_content` maydoni — bor bo'lsa
    reasoning: str | None = None


# Embedding vazifasi: indekslashda hujjat, qidiruvda so'rov (asimmetrik retrieval).
# Gemini buni hisobga oladi va vektorlarni boshqa fazoga tushiradi — ikkalasi ham
# default bilan yuborilsa bepul aniqlik yo'qoladi (arxitektura v2, 5.3-band).
EmbedTaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
EMBED_DOCUMENT: EmbedTaskType = "RETRIEVAL_DOCUMENT"
EMBED_QUERY: EmbedTaskType = "RETRIEVAL_QUERY"


@dataclass(slots=True)
class EmbedResult:
    vectors: list[list[float]]
    model: str
    provider: str
    usage: Usage


# ─── batch (asinxron) embedding — 4-bosqich/C ────────────────────────────────
#
# Gemini Batch API sinxron narxning **yarmini** oladi, evaziga javob soatlar
# ichida keladi (SLA 24 soat). Indekslash — fon ishi, kechikish muhim emas;
# shuning uchun `embed_messages` / `embed_nodes` uchun bu to'g'ridan-to'g'ri
# 50% chegirma. Qidiruv tomoni (`RETRIEVAL_QUERY`) **hech qachon** batch
# bo'lmaydi — u user kutib turgan yo'l.


class BatchState(StrEnum):
    """Provider holatlarining normallashtirilgan ko'rinishi."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

    @property
    def done(self) -> bool:
        return self is not BatchState.PENDING and self is not BatchState.RUNNING


@dataclass(slots=True)
class BatchStatus:
    job: str
    state: BatchState
    #: provider qaytargan xom holat (`JOB_STATE_...`) — loglar uchun
    raw_state: str = ""
    #: API qaytargan model nomi (narx qoidasi: so'ralgani emas, qaytgani)
    model: str = ""
    error: str = ""

    @property
    def done(self) -> bool:
        return self.state.done


class LLMError(RuntimeError):
    def __init__(self, provider: str, message: str, *, status: int | None = None) -> None:
        self.provider = provider
        self.status = status
        super().__init__(f"[{provider}] {message}")


class CapabilityError(LLMError):
    """Provider so'ralgan imkoniyatni qo'llab-quvvatlamaydi."""


# HTTP status'lar: qayta urinish mantiqiy bo'lganlari
# 529 — Anthropic "overloaded_error"
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


class BaseProvider(ABC):
    """Umumiy retry/backoff va imkoniyat tekshiruvi."""

    name: str = "base"
    max_retries: int = 3

    @abstractmethod
    def capabilities(self, model: str) -> frozenset[Capability]:
        """Berilgan model nimalarni qo'llab-quvvatlaydi."""

    def require(self, model: str, *needed: Capability) -> None:
        have = self.capabilities(model)
        missing = [c for c in needed if c not in have]
        if missing:
            raise CapabilityError(
                self.name,
                f"model '{model}' quvvatlamaydi: {', '.join(missing)}",
            )

    @abstractmethod
    async def chat(
        self,
        messages: list[Msg],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult: ...

    async def embed(
        self,
        texts: list[str],
        *,
        model: str,
        dim: int,
        task_type: EmbedTaskType = EMBED_DOCUMENT,
    ) -> EmbedResult:
        """Provider `task_type` ni qo'llamasa — e'tiborsiz qoldiradi (xato emas)."""
        raise CapabilityError(self.name, "embedding qo'llab-quvvatlanmaydi")

    # ── batch embedding ──────────────────────────────────────────────────────
    #
    # Uch bosqich ataylab ajratilgan: submit qaytarishi bilan jarayon o'lishi
    # mumkin, job esa serverda qoladi. Shuning uchun `job` nomi **DB'da**
    # saqlanadi (`embed_batches`) va poll/fetch boshqa cron yugurishida,
    # boshqa jarayonda bo'ladi. Provider obyektida hech qanday holat yo'q.

    async def embed_batch_submit(
        self,
        texts: list[str],
        *,
        model: str,
        dim: int,
        task_type: EmbedTaskType = EMBED_DOCUMENT,
        tag: str = "",
    ) -> str:
        """Batch job yaratadi va uning **nomini** qaytaradi (`batches/…`)."""
        raise CapabilityError(self.name, "batch embedding qo'llab-quvvatlanmaydi")

    async def embed_batch_poll(self, job: str) -> BatchStatus:
        raise CapabilityError(self.name, "batch embedding qo'llab-quvvatlanmaydi")

    async def embed_batch_fetch(self, job: str, *, model: str = "") -> EmbedResult:
        """Tugagan jobning vektorlari — **kirish tartibida**.

        `list[list[float]]` emas, `EmbedResult`: chaqiruvchiga API qaytargan
        model nomi ham kerak (narx qoidasi, `pricing.py`).

        `model` — job yuborilganda **so'ralgan** model. API javobida model
        maydoni bo'lmasa (Gemini buni qiladi) normallashtirish va
        `message_embeddings.model` shu nomga tayanadi; usiz qator `model=''`
        bilan yozilib, qidiruv filtridan hech qachon o'tmasdi.

        `usage.tokens_in` javobda yo'q/0 bo'lishi mumkin (usage metadata
        kelmasa) — narx taxminini tiklash chaqiruvchi tomonda
        (`services/embed_batches.py::_poll_one`, `estimate_tokens_heuristic`).
        """
        raise CapabilityError(self.name, "batch embedding qo'llab-quvvatlanmaydi")

    async def generate_image(self, prompt: str, *, model: str) -> bytes:
        raise CapabilityError(self.name, "rasm generatsiya qo'llab-quvvatlanmaydi")

    async def aclose(self) -> None:
        return None

    # ── retry ────────────────────────────────────────────────────────────────

    async def _with_retry[T](self, op: str, fn: Any) -> T:
        delay = 1.0
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return await fn()  # type: ignore[no-any-return]
            except CapabilityError:
                raise  # qayta urinish befoyda
            except LLMError as exc:
                if exc.status is not None and exc.status not in RETRYABLE_STATUS:
                    raise
                last = exc
            except (TimeoutError, ConnectionError) as exc:
                last = exc

            if attempt < self.max_retries:
                sleep_for = delay + random.random() * 0.3  # noqa: S311 — jitter, crypto emas
                log.warning(
                    "llm.retry",
                    provider=self.name,
                    op=op,
                    attempt=attempt,
                    sleep=round(sleep_for, 2),
                    error=str(last),
                )
                await asyncio.sleep(sleep_for)
                delay *= 2

        raise LLMError(self.name, f"{op}: {self.max_retries} urinishdan keyin ham xato: {last}")


def sanitize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini FunctionDeclaration OpenAPI qism-to'plamini qabul qiladi.

    `$schema`, `additionalProperties`, `const` kabi kalitlar 400 xatoga olib
    keladi — rekursiv tozalanadi. OpenAI/DeepSeek uchun zararsiz.
    """
    unsupported = {
        "$schema",
        "$id",
        "$ref",
        "additionalProperties",
        "const",
        "examples",
        "definitions",
        "$defs",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "patternProperties",
    }

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: walk(v) for k, v in node.items() if k not in unsupported}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(schema)  # type: ignore[no-any-return]
