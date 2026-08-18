"""Markazlashgan konfiguratsiya. Hamma env o'qish shu yerdan o'tadi."""

from __future__ import annotations

import base64
from enum import StrEnum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WriteMode(StrEnum):
    READ_ONLY = "read_only"
    WRITE_WITH_CONFIRM = "write_with_confirm"
    AUTONOMOUS = "autonomous"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # ─── Bot API ───
    bot_token: str
    control_bot_id: int
    allowed_user_ids: str = ""

    # ─── MTProto ───
    tg_api_id: int
    tg_api_hash: str
    tg_device_model: str = "Desktop"
    tg_system_version: str = "Windows 10"
    tg_app_version: str = "5.7.1"

    # ─── Xavfsizlik ───
    master_key_b64: str
    webapp_base_url: str = "https://example.com"

    # ─── Web UI ───
    web_session_ttl_hours: int = Field(default=24 * 14, ge=1, le=24 * 90)
    # Auth flow: telefon → kod → 2FA. Bitta oqim shuncha daqiqa yashaydi.
    web_auth_flow_ttl_min: int = Field(default=10, ge=2, le=30)
    # Bir IP'dan 10 daqiqada nechta login boshlash mumkin (brute-force/spam)
    web_auth_rate_per_ip: int = Field(default=5, ge=1, le=100)
    # AI chatga kontekst sifatida beriladigan oxirgi xabarlar limiti
    # >200 faqat sinxronlangan (DB) chatlar uchun — jonli GetHistory 200 bilan cheklanadi
    web_context_max_messages: int = Field(default=1000, ge=10, le=5000)
    web_context_default_messages: int = Field(default=50, ge=5, le=1000)
    # Har javobdan keyin arzon model bilan avtomatik baho (relevance/usefulness/grounded)
    web_auto_eval: bool = True
    # Baholanadigan javoblar ulushi (0.0-1.0) — xarajatni kamaytirish uchun sampling
    web_auto_eval_sample: float = Field(default=1.0, ge=0.0, le=1.0)
    # Agent (tool) sikli chegaralari — xarajat qopqog'i
    agent_max_iterations: int = Field(default=5, ge=1, le=12)
    agent_tool_result_tokens: int = Field(default=12_000, ge=1_000, le=60_000)
    # Embedding indeksi (vektor qidiruv). Gemini embed — arzon, lekin o'chirsa bo'ladi
    embed_enabled: bool = True
    embed_batch_per_run: int = Field(default=500, ge=50, le=5000)

    # ─── Infra ───
    database_url: str
    redis_url: str

    # ─── LLM: umumiy ───
    # Standart provider: auto | claude | gemini | deepseek.
    # `auto` — Claude kredensiali (Claude Code / Anthropic login) topilsa
    # `claude`, aks holda `gemini`. Vazifa darajasida `llm_task_*` bilan
    # bekor qilinadi.
    llm_provider: str = "auto"
    # Tanlangan provider vazifani bajara olmasa (masalan DeepSeek'da embedding
    # yo'q) — shu providerga o'tiladi. Gemini'dan boshqasini qo'yish tavsiya
    # etilmaydi: faqat u to'liq imkoniyatli.
    llm_fallback_provider: str = "gemini"

    # Vazifa → `provider:model` spec. Bo'sh = llm_provider + standart model.
    llm_task_route: str = ""
    llm_task_search: str = ""
    llm_task_tools: str = ""
    llm_task_deep: str = ""
    llm_task_embed: str = ""
    llm_task_image: str = ""

    # ─── Claude / Anthropic ───
    # Kredensial ustuvorligi (birinchi topilgani ishlatiladi):
    #   1. ANTHROPIC_API_KEY            — oddiy API kalit
    #   2. CLAUDE_CODE_OAUTH_TOKEN      — `claude setup-token` chiqargan token
    #      (Claude Code obunasi), yoki ANTHROPIC_AUTH_TOKEN
    #   3. diskdagi profil              — `ant auth login` / Claude Code login
    #      (~/.config/anthropic yoki ANTHROPIC_CONFIG_DIR)
    # Hech biri yo'q → `auto` rejimda Gemini ishlatiladi.
    anthropic_api_key: str = ""
    anthropic_auth_token: str = ""
    claude_code_oauth_token: str = ""
    anthropic_base_url: str = ""  # bo'sh = api.anthropic.com
    claude_model_router: str = "claude-haiku-4-5"  # arzon: intent, digest, judge
    claude_model_fast: str = "claude-sonnet-5"  # search/tools — sifat/narx muvozanati
    claude_model_deep: str = "claude-opus-5"  # "chuqur tahlil" tugmasi

    # ─── Claude Code CLI (`claude_code` provider) ───
    # `claude -p` subprocess — Claude Code'ning o'z login sessiyasidan (Keychain /
    # ~/.claude) foydalanadi, token nusxalash shart emas. `auto` tartibida API
    # kredensiali topilmasa va CLI login qilingan bo'lsa tanlanadi.
    claude_code_auto: bool = True  # False = auto rejimda CLI hisobga olinmaydi
    claude_code_bin: str = ""  # bo'sh = PATH'dan `claude`
    claude_code_timeout: int = Field(default=300, ge=10, le=3600)  # soniya, bitta so'rov

    # ─── Gemini ───
    gemini_api_key: str = ""
    gemini_model_router: str = "gemini-2.5-flash-lite"
    gemini_model_fast: str = "gemini-2.5-flash"
    gemini_model_deep: str = "gemini-2.5-pro"
    gemini_model_image: str = "gemini-2.5-flash-image"
    gemini_model_embed: str = "gemini-embedding-001"
    embed_dim: int = 768

    # ─── DeepSeek (OpenAI-mos API) ───
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model_chat: str = "deepseek-chat"  # function calling BOR
    deepseek_model_reasoner: str = "deepseek-reasoner"  # function calling YO'Q

    # ─── Rejim ───
    env: str = "dev"
    log_level: str = "INFO"
    default_locale: str = "uz"
    max_accounts: int = Field(default=3, ge=1, le=10)
    default_write_mode: WriteMode = WriteMode.WRITE_WITH_CONFIRM

    @field_validator("master_key_b64")
    @classmethod
    def _check_master_key(cls, v: str) -> str:
        try:
            raw = base64.b64decode(v, validate=True)
        except Exception as exc:
            raise ValueError("MASTER_KEY_B64 base64 emas") from exc
        if len(raw) != 32:
            raise ValueError("MASTER_KEY_B64 aynan 32 bayt bo'lishi kerak (AES-256)")
        return v

    @property
    def master_key(self) -> bytes:
        return base64.b64decode(self.master_key_b64)

    @property
    def web_secure_cookies(self) -> bool:
        """HTTPS'da bo'lsak `Secure` cookie. Lokal http://localhost'da o'chiq."""
        return self.webapp_base_url.lower().startswith("https://")

    @property
    def allowed_users(self) -> frozenset[int]:
        """Bo'sh bo'lsa — hamma ruxsat (faqat dev uchun)."""
        raw = self.allowed_user_ids.strip()
        if not raw:
            return frozenset()
        return frozenset(int(x) for x in raw.split(",") if x.strip())

    @property
    def is_prod(self) -> bool:
        return self.env.lower() in {"prod", "production"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
