"""Vazifa → provider/model marshrutizatsiyasi.

Gemini va DeepSeek teng imkoniyatli emas. Router har bir vazifa uchun
kerakli imkoniyatlarni biladi va:

  1. konfiguratsiyadagi providerni tanlaydi,
  2. u imkoniyatni qo'llab-quvvatlamasa — fallback'ga o'tadi va ogohlantiradi,
  3. fallback ham qila olmasa — aniq xato tashlaydi (jimgina ishlamay qolish yo'q).

Konfiguratsiya `provider:model` spec ko'rinishida beriladi, masalan:

    LLM_TASK_DEEP=deepseek:deepseek-reasoner
    LLM_TASK_TOOLS=gemini:gemini-2.5-flash

Bo'sh qoldirilsa — `LLM_PROVIDER` va o'sha providerning shu vazifa uchun
standart modeli ishlatiladi.
"""

from __future__ import annotations

from enum import StrEnum

from app.config import get_settings
from app.llm.base import (
    BaseProvider,
    Capability,
    ChatResult,
    EmbedResult,
    LLMError,
    Msg,
    ToolSpec,
)
from app.logging import get_logger

log = get_logger(__name__)


class Task(StrEnum):
    ROUTE = "route"  # intent klassifikatsiya — arzon va tez
    SEARCH = "search"  # qidiruv natijasini sintez qilish, rerank
    TOOLS = "tools"  # agent tool-calling sikli
    DEEP = "deep"  # chuqur tahlil, kanal strategiyasi
    EMBED = "embed"  # vektor indeks
    IMAGE = "image"  # post uchun rasm


# Har bir vazifa uchun majburiy imkoniyatlar.
TASK_REQUIREMENTS: dict[Task, frozenset[Capability]] = {
    Task.ROUTE: frozenset({Capability.CHAT, Capability.JSON_MODE}),
    Task.SEARCH: frozenset({Capability.CHAT, Capability.JSON_MODE}),
    Task.TOOLS: frozenset({Capability.CHAT, Capability.TOOLS}),
    Task.DEEP: frozenset({Capability.CHAT}),
    Task.EMBED: frozenset({Capability.EMBED}),
    Task.IMAGE: frozenset({Capability.IMAGE_GEN}),
}

PROVIDERS = ("gemini", "deepseek")


_instances: dict[str, BaseProvider] = {}


def get_provider(name: str) -> BaseProvider:
    """Provider instansiyasi (jarayon davomida bitta)."""
    if name in _instances:
        return _instances[name]

    if name == "gemini":
        from app.llm.gemini import GeminiProvider

        provider: BaseProvider = GeminiProvider()
    elif name == "deepseek":
        from app.llm.deepseek import DeepSeekProvider

        provider = DeepSeekProvider()
    else:
        raise LLMError("router", f"noma'lum provider: {name!r} (mavjud: {', '.join(PROVIDERS)})")

    _instances[name] = provider
    return provider


def default_model(provider: str, task: Task) -> str:
    s = get_settings()
    if provider == "gemini":
        return {
            Task.ROUTE: s.gemini_model_router,
            Task.SEARCH: s.gemini_model_fast,
            Task.TOOLS: s.gemini_model_fast,
            Task.DEEP: s.gemini_model_deep,
            Task.EMBED: s.gemini_model_embed,
            Task.IMAGE: s.gemini_model_image,
        }[task]
    if provider == "deepseek":
        return {
            Task.ROUTE: s.deepseek_model_chat,
            Task.SEARCH: s.deepseek_model_chat,
            Task.TOOLS: s.deepseek_model_chat,  # reasoner'da function calling yo'q
            Task.DEEP: s.deepseek_model_reasoner,
            # DeepSeek'da bular yo'q — capability tekshiruvi fallback'ga o'tkazadi
            Task.EMBED: "",
            Task.IMAGE: "",
        }[task]
    raise LLMError("router", f"noma'lum provider: {provider!r}")


def _spec_for(task: Task) -> str:
    s = get_settings()
    return {
        Task.ROUTE: s.llm_task_route,
        Task.SEARCH: s.llm_task_search,
        Task.TOOLS: s.llm_task_tools,
        Task.DEEP: s.llm_task_deep,
        Task.EMBED: s.llm_task_embed,
        Task.IMAGE: s.llm_task_image,
    }[task].strip()


def _parse_spec(spec: str, task: Task) -> tuple[str, str]:
    """`provider:model` yoki faqat `provider`."""
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in PROVIDERS:
        raise LLMError("router", f"{task}: noma'lum provider {provider!r}")
    return provider, model or default_model(provider, task)


def resolve(task: Task) -> tuple[BaseProvider, str]:
    """Vazifa uchun (provider, model) juftini qaytaradi. Kerak bo'lsa fallback."""
    s = get_settings()
    needed = TASK_REQUIREMENTS[task]

    spec = _spec_for(task)
    if spec:
        provider_name, model = _parse_spec(spec, task)
    else:
        provider_name = s.llm_provider.lower()
        model = default_model(provider_name, task)

    if model:
        provider = get_provider(provider_name)
        if needed <= provider.capabilities(model):
            return provider, model

    # ── fallback ─────────────────────────────────────────────────────────────
    fb_name = s.llm_fallback_provider.lower()
    fb_model = default_model(fb_name, task)
    fb = get_provider(fb_name)

    if not fb_model or not (needed <= fb.capabilities(fb_model)):
        raise LLMError(
            "router",
            f"'{task}' vazifasini bajaradigan provider yo'q. "
            f"Kerak: {', '.join(sorted(needed))}. "
            f"Urinildi: {provider_name}:{model or '—'}, {fb_name}:{fb_model or '—'}",
        )

    log.warning(
        "llm.fallback",
        task=str(task),
        requested=f"{provider_name}:{model or '—'}",
        used=f"{fb_name}:{fb_model}",
        missing=sorted(
            needed - (get_provider(provider_name).capabilities(model) if model else frozenset())
        ),
    )
    return fb, fb_model


class LLM:
    """Ilova kodi shu fasad bilan ishlaydi — provider nomini bilishi shart emas."""

    async def chat(
        self,
        task: Task,
        messages: list[Msg],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        provider, model = resolve(task)
        result = await provider.chat(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        log.info(
            "llm.chat",
            task=str(task),
            provider=result.provider,
            model=result.model,
            tokens_in=result.usage.tokens_in,
            tokens_out=result.usage.tokens_out,
            cached=result.usage.cached_tokens,
            tool_calls=len(result.tool_calls),
        )
        return result

    async def embed(self, texts: list[str], *, dim: int | None = None) -> EmbedResult:
        provider, model = resolve(Task.EMBED)
        return await provider.embed(texts, model=model, dim=dim or get_settings().embed_dim)

    async def generate_image(self, prompt: str) -> bytes:
        provider, model = resolve(Task.IMAGE)
        return await provider.generate_image(prompt, model=model)


async def close_all() -> None:
    """Graceful shutdown — faqat yaratilgan providerlarni yopadi."""
    for provider in list(_instances.values()):
        await provider.aclose()
    _instances.clear()
