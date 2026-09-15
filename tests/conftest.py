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

import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import get_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings():
    return get_settings()


# ─── haqiqiy Postgres ustidagi testlar ───────────────────────────────────────
#
# `TEST_DATABASE_URL` berilmasa — skip. CI'da u `pgvector/pgvector:pg16`
# service'ga ishora qiladi (`.github/workflows/ci.yml`), lokalda esa
# bir martalik konteyner:
#
#   docker run -d --name tgai-testpg -e POSTGRES_PASSWORD=test \
#              -p 55432:5432 pgvector/pgvector:pg16
#   TEST_DATABASE_URL=postgresql+asyncpg://postgres:test@localhost:55432/postgres
#
# Baza UTF-8 lokalda bo'lishi **shart**: `uz_norm` Postgres `lower()` ga
# tayanadi, `C` lokalda esa u kirillni pasaytirmaydi va Python bilan paritet
# buziladi.


@pytest.fixture(scope="session")
def pg_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL berilmagan — DB integratsiya testlari o'tkazib yuborildi")
    return url


@pytest.fixture(scope="session")
def pg_engine(pg_url: str) -> AsyncEngine:
    """Migratsiyalari yugurtirilgan test bazasi uchun AsyncEngine.

    `alembic upgrade head` **bir marta**, alohida jarayonda bajariladi —
    `migrations/env.py` `get_settings().database_url` ni o'qiydi, u esa
    jarayon boshida keshlanadi, shuning uchun uni shu jarayonda qayta
    yo'naltirib bo'lmaydi.

    `NullPool`: har test o'z event loop'ida ishlaydi, asyncpg ulanishi esa
    loop'ga bog'langan — pool'dagi eski ulanishni qayta ishlatish xato beradi.
    """
    env = {**os.environ, "DATABASE_URL": pg_url, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(  # argv qat'iy, shell yo'q
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
    )
    return create_async_engine(pg_url, poolclass=NullPool)


@pytest.fixture(scope="session")
def master_key(settings) -> bytes:
    return settings.master_key
