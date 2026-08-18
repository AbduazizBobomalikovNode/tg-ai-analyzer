"""Vazifa → provider/model marshrutizatsiyasi.

Claude, Gemini va DeepSeek teng imkoniyatli emas. Router har bir vazifa uchun
kerakli imkoniyatlarni biladi va:

  1. konfiguratsiyadagi providerni tanlaydi (`LLM_PROVIDER=auto` bo'lsa —
     Claude kredensiali bor-yo'qligiga qarab `claude` yoki `gemini`),
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

PROVIDERS = ("claude", "claude_code", "gemini", "deepseek")

# `LLM_PROVIDER=auto` uchun maxsus qiymat
AUTO = "auto"


_instances: dict[str, BaseProvider] = {}
# `auto` qarori — jarayon davomida bir marta hisoblanadi (kredensial probe arzon,
# lekin har chaqiruvda takrorlash shart emas). None = hali hisoblanmagan.
_auto_choice: str | None = None


def auto_provider() -> str:
    """`auto` rejimda haqiqiy provider. Tartib:

      1. `claude`      — API kredensiali bor (ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN / profil)
      2. `claude_code` — `claude` CLI o'rnatilgan va login qilingan (CLAUDE_CODE_AUTO=true)
      3. `gemini`      — hech biri yo'q

    Claude'da embedding/rasm yo'q — ular baribir fallback (`gemini`) orqali ketadi.
    """
    global _auto_choice
    if _auto_choice is None:
        from app.llm.claude import detect_claude_auth
        from app.llm.claude_code import detect_claude_code

        s = get_settings()
        auth = detect_claude_auth(s)
        cli = detect_claude_code(s) if (auth is None and s.claude_code_auto) else None
        if auth:
            _auto_choice = "claude"
        elif cli and cli.logged_in:
            _auto_choice = "claude_code"
        else:
            _auto_choice = "gemini"
        log.info(
            "llm.auto_provider",
            chosen=_auto_choice,
            claude_auth=auth.source if auth else None,
            claude_cli=(cli.binary if cli else None),
            claude_cli_logged_in=(cli.logged_in if cli else None),
        )
    return _auto_choice


def effective_provider(name: str) -> str:
    """Konfiguratsiya qiymatini haqiqiy provider nomiga aylantiradi (`auto` ni hal qiladi)."""
    name = name.strip().lower()
    return auto_provider() if name == AUTO else name


def get_provider(name: str) -> BaseProvider:
    """Provider instansiyasi (jarayon davomida bitta)."""
    if name in _instances:
        return _instances[name]

    if name == "claude":
        from app.llm.claude import ClaudeProvider

        provider: BaseProvider = ClaudeProvider()
    elif name == "claude_code":
        from app.llm.claude_code import ClaudeCodeProvider

        provider = ClaudeCodeProvider()
    elif name == "gemini":
        from app.llm.gemini import GeminiProvider

        provider = GeminiProvider()
    elif name == "deepseek":
        from app.llm.deepseek import DeepSeekProvider

        provider = DeepSeekProvider()
    else:
        raise LLMError("router", f"noma'lum provider: {name!r} (mavjud: {', '.join(PROVIDERS)})")

    _instances[name] = provider
    return provider


def default_model(provider: str, task: Task) -> str:
    s = get_settings()
    if provider in ("claude", "claude_code"):
        return {
            Task.ROUTE: s.claude_model_router,
            Task.SEARCH: s.claude_model_fast,
            Task.TOOLS: s.claude_model_fast,
            Task.DEEP: s.claude_model_deep,
            # Claude'da bular yo'q — capability tekshiruvi fallback'ga o'tkazadi
            Task.EMBED: "",
            Task.IMAGE: "",
        }[task]
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
    provider = effective_provider(provider)
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
        provider_name = effective_provider(s.llm_provider)
        if provider_name not in PROVIDERS:
            raise LLMError(
                "router",
                f"LLM_PROVIDER noma'lum: {s.llm_provider!r} (mavjud: auto, {', '.join(PROVIDERS)})",
            )
        model = default_model(provider_name, task)

    if model:
        provider = get_provider(provider_name)
        if needed <= provider.capabilities(model):
            return provider, model

    # ── fallback ─────────────────────────────────────────────────────────────
    fb_name = effective_provider(s.llm_fallback_provider)
    if fb_name not in PROVIDERS:
        raise LLMError("router", f"LLM_FALLBACK_PROVIDER noma'lum: {s.llm_fallback_provider!r}")
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
        import time

        from app.observability import record_llm

        provider, model = resolve(task)
        t0 = time.perf_counter()
        try:
            result = await provider.chat(
                messages,
                model=model,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
            )
        except LLMError:
            record_llm(
                provider=provider.name,
                model=model,
                task=str(task),
                ok=False,
                tokens_in=0,
                tokens_out=0,
                seconds=time.perf_counter() - t0,
            )
            raise
        record_llm(
            provider=result.provider,
            model=result.model,
            task=str(task),
            ok=True,
            tokens_in=result.usage.tokens_in,
            tokens_out=result.usage.tokens_out,
            seconds=time.perf_counter() - t0,
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


def reset() -> None:
    """Provider keshini va `auto` qarorini tozalaydi (testlar, qayta konfiguratsiya)."""
    global _auto_choice
    _instances.clear()
    _auto_choice = None


async def close_all() -> None:
    """Graceful shutdown — faqat yaratilgan providerlarni yopadi."""
    for provider in list(_instances.values()):
        await provider.aclose()
    reset()
