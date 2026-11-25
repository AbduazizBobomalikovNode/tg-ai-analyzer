"""Migratsiya 0019 (`senders`) + ingestion to'ldirishi — haqiqiy Postgres ustida.

`TEST_DATABASE_URL` berilmasa skip (`conftest.pg_engine`).

Nima qo'riqlanadi:
  * jadval shakli: PK `(account_id, tg_user_id)`, `display_name` NOT NULL,
    va **telefon ustuni yo'q** (qator qiymati LLM promptiga tushadi);
  * `ingestion.upsert_senders` idempotent — qayta sync nomni yangilaydi,
    ikkinchi qator yaratmaydi;
  * `tools.resolve_senders` username va ism bo'lagi bo'yicha topadi, boshqa
    akkauntning jo'natuvchisini esa **topmaydi**.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.services import ingestion as ING
from app.services import tools as T

pytestmark = pytest.mark.usefixtures("pg_engine")

TG_USER = 900_000_019
OTHER_TG_USER = 900_000_020


async def test_senders_table_shape_and_has_no_phone_column(pg_engine: AsyncEngine) -> None:
    async with pg_engine.connect() as conn:
        cols = {
            r[0]: (r[1], r[2])
            for r in (
                await conn.execute(
                    text(
                        "SELECT column_name, data_type, is_nullable "
                        "FROM information_schema.columns WHERE table_name = 'senders'"
                    )
                )
            ).all()
        }
        pk = [
            r[0]
            for r in (
                await conn.execute(
                    text(
                        "SELECT a.attname FROM pg_index i "
                        "JOIN pg_attribute a ON a.attrelid = i.indrelid "
                        "AND a.attnum = ANY(i.indkey) "
                        "WHERE i.indrelid = 'senders'::regclass AND i.indisprimary "
                        "ORDER BY a.attname"
                    )
                )
            ).all()
        ]
    assert cols, "0019 migratsiyasi yugurmagan"
    assert set(cols) == {
        "account_id",
        "tg_user_id",
        "display_name",
        "username",
        "is_bot",
        "updated_at",
    }
    assert cols["display_name"][1] == "NO" and cols["is_bot"][1] == "NO"
    assert pk == ["account_id", "tg_user_id"]
    # Invariant: raqam bu yerga hech qachon qo'shilmaydi.
    assert not any("phone" in c for c in cols)


async def _fixture_account(conn: object, *, label: str, tg: int) -> tuple[int, int]:
    uid = (
        await conn.execute(  # type: ignore[attr-defined]
            text("INSERT INTO users (tg_user_id, username) VALUES (:tg, :u) RETURNING id"),
            {"tg": tg, "u": label},
        )
    ).scalar_one()
    aid = (
        await conn.execute(  # type: ignore[attr-defined]
            text(
                "INSERT INTO accounts (user_id, label, status) "
                "VALUES (:uid, :l, 'active') RETURNING id"
            ),
            {"uid": uid, "l": label},
        )
    ).scalar_one()
    return int(uid), int(aid)


@pytest.fixture
async def accounts(pg_engine: AsyncEngine):
    async with pg_engine.begin() as conn:
        u1, a1 = await _fixture_account(conn, label="senders-a", tg=TG_USER)
        u2, a2 = await _fixture_account(conn, label="senders-b", tg=OTHER_TG_USER)
    yield a1, a2
    async with pg_engine.begin() as conn:
        await conn.execute(text("DELETE FROM users WHERE id IN (:a, :b)"), {"a": u1, "b": u2})


async def test_ingestion_fills_senders_from_the_batch_and_is_idempotent(
    pg_engine: AsyncEngine, accounts: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    account_id, _other = accounts
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)

    class _Scope:
        async def __aenter__(self):
            self.s = maker()
            return self.s

        async def __aexit__(self, *exc):
            if exc[0] is None:
                await self.s.commit()
            await self.s.close()

    monkeypatch.setattr(ING, "session_scope", lambda: _Scope())

    # `GetHistory` javobidagi `m.sender` — qo'shimcha RPC yo'q.
    sender = SimpleNamespace(
        id=555_001, first_name="Nurbek", last_name="Aliyev", username="nurbek", bot=False
    )
    rows = [r for r in (ING.sender_to_row(account_id, sender),) if r]
    assert await ING.upsert_senders(rows) == 1

    # qayta sync: ism o'zgargan → yangilanadi, qator ko'paymaydi
    sender.last_name = "Aliev"
    sender.username = "nurbek_a"
    await ING.upsert_senders([ING.sender_to_row(account_id, sender)])  # type: ignore[list-item]

    async with pg_engine.connect() as conn:
        got = (
            await conn.execute(
                text(
                    "SELECT display_name, username, is_bot FROM senders "
                    "WHERE account_id = :a AND tg_user_id = 555001"
                ),
                {"a": account_id},
            )
        ).all()
    assert got == [("Nurbek Aliev", "nurbek_a", False)]


async def test_resolve_senders_matches_by_username_and_name_within_the_account(
    pg_engine: AsyncEngine, accounts: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    account_id, other_id = accounts
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)

    class _Scope:
        async def __aenter__(self):
            self.s = maker()
            return self.s

        async def __aexit__(self, *exc):
            if exc[0] is None:
                await self.s.commit()
            await self.s.close()

    monkeypatch.setattr(ING, "session_scope", lambda: _Scope())
    await ING.upsert_senders(
        [
            {
                "account_id": account_id,
                "tg_user_id": 556_001,
                "display_name": "Nurbek Aliyev",
                "username": "nurbek",
                "is_bot": False,
            },
            {
                "account_id": other_id,
                "tg_user_id": 556_002,
                "display_name": "Nurbek Boshqa",
                "username": "nurbek2",
                "is_bot": False,
            },
        ]
    )
    async with maker() as session:
        assert await T.resolve_senders(session, account_id, "@nurbek") == [556_001]
        assert await T.resolve_senders(session, account_id, "Aliyev") == [556_001]
        # boshqa akkauntning jo'natuvchisi ko'rinmaydi
        assert await T.resolve_senders(session, account_id, "Boshqa") == []
        assert await T.resolve_senders(session, account_id, "x") == []
