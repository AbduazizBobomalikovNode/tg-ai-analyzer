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

    # ─── Infra ───
    database_url: str
    redis_url: str

    # ─── LLM: umumiy ───
    # Standart provider. Vazifa darajasida `llm_task_*` bilan bekor qilinadi.
    llm_provider: str = "gemini"
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
