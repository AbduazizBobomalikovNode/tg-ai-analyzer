from __future__ import annotations

import base64
import os

# get_settings() import paytida o'qiladi — shuning uchun eng yuqorida.
os.environ.setdefault("BOT_TOKEN", "123456:TEST")
os.environ.setdefault("CONTROL_BOT_ID", "123456")
os.environ.setdefault("TG_API_ID", "1")
os.environ.setdefault("TG_API_HASH", "testhash")
os.environ.setdefault("MASTER_KEY_B64", base64.b64encode(b"\x01" * 32).decode())
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("DEEPSEEK_API_KEY", "test-deepseek-key")

import pytest

from app.config import get_settings


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def master_key(settings) -> bytes:
    return settings.master_key
