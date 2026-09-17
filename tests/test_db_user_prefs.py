"""Migratsiya 0018 (`users.prefs`) — haqiqiy Postgres ustida.

`TEST_DATABASE_URL` berilmasa skip (`conftest.pg_engine`).

Nima qo'riqlanadi:
  * ustun **mavjud**, `jsonb`, `NOT NULL` va default `{}` — ya'ni eski
    qatorlar ham, `prefs` bermay INSERT qilingan yangi qator ham `None`
    bo'lib chiqmaydi (o'qish yo'llarida `None` tekshiruvi yo'q);
  * `services/prefs.merge` natijasi yozilib, o'sha holicha qaytadi.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.services import prefs as P

pytestmark = pytest.mark.usefixtures("pg_engine")


async def test_users_prefs_column_shape(pg_engine: AsyncEngine) -> None:
    async with pg_engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_name = 'users' AND column_name = 'prefs'"
                )
            )
        ).first()
    assert row is not None, "0018 migratsiyasi yugurmagan"
    data_type, is_nullable, default = row
    assert data_type == "jsonb" and is_nullable == "NO"
    assert "'{}'" in (default or "")


async def test_prefs_roundtrip(pg_engine: AsyncEngine) -> None:
    merged = P.merge({}, {"theme": "dark", "pinned_chats": [3, 9], "locale": "ru"})
    async with pg_engine.begin() as conn:
        uid = (
            await conn.execute(
                text(
                    "INSERT INTO users (tg_user_id, username) "
                    "VALUES (:tg, 'prefs-test') RETURNING id"
                ),
                {"tg": 900_000_018},
            )
        ).scalar_one()
        # default: `prefs` berilmagan qator ham bo'sh obyekt
        stored = (
            await conn.execute(text("SELECT prefs FROM users WHERE id = :id"), {"id": uid})
        ).scalar_one()
        assert stored == {}

        await conn.execute(
            text("UPDATE users SET prefs = CAST(:p AS jsonb) WHERE id = :id"),
            {"p": json.dumps(merged), "id": uid},
        )
        back = (
            await conn.execute(text("SELECT prefs FROM users WHERE id = :id"), {"id": uid})
        ).scalar_one()
        assert P.normalize(back) == merged
        await conn.execute(text("DELETE FROM users WHERE id = :id"), {"id": uid})
