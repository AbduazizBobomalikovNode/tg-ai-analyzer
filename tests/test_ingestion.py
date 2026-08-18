"""Ingestion (2-bosqich) — sof funksiyalar va worker retry mantiqi, DB/Telegram'siz.

Qo'riqlaydi:
  * Telethon Message → `messages` qatori (media saqlanmaydi, faqat turi)
  * ramp-up limitlari (yangi akkaunt < 24 soat)
  * snapshot tier'lari (har soat / 6 soat / kuniga — 4.6-band)
  * FloodWait → SyncPaused → worker o'zini defer bilan qayta navbatga qo'yadi
  * chat konteksti: kichik limit jonli, katta limit faqat DB
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from telethon.tl.types import Message as TlMessage
from telethon.tl.types import PeerChannel, PeerUser

from app.services import chat_service as cs
from app.services import ingestion as ing
from app.worker import tasks

# ─── message_to_row ──────────────────────────────────────────────────────────


def _tl_message(**kw: Any) -> TlMessage:
    base: dict[str, Any] = {
        "id": 42,
        "peer_id": PeerChannel(channel_id=100),
        "date": datetime(2026, 8, 18, 10, 0, tzinfo=UTC),
        "message": "salom",
    }
    base.update(kw)
    return TlMessage(**base)


def test_message_to_row_basic() -> None:
    m = _tl_message(
        from_id=PeerUser(user_id=7),
        views=120,
        forwards=3,
        pinned=True,
        grouped_id=99,
        edit_date=datetime(2026, 8, 18, 11, 0, tzinfo=UTC),
        replies=SimpleNamespace(replies=5),
        reactions=SimpleNamespace(
            results=[SimpleNamespace(count=2, reaction=SimpleNamespace(emoticon="👍"))]
        ),
    )
    row = ing.message_to_row(5, m)
    assert row is not None
    assert row["chat_id"] == 5 and row["tg_msg_id"] == 42
    assert row["sender_id"] == 7
    assert row["text"] == "salom" and row["media_type"] is None
    assert row["views"] == 120 and row["forwards"] == 3 and row["is_pinned"] is True
    assert row["reactions_total"] == 2 and row["replies_count"] == 5
    assert row["grouped_id"] == 99 and row["edited_at"] is not None
    assert "raw" not in row  # xom JSON saqlanmaydi


def test_message_to_row_media_only_keeps_type_not_bytes() -> None:
    class MessageMediaPhoto:  # Telethon nomi bilan bir xil
        photo = b"\x00" * 1000

    m = _tl_message(message="", media=MessageMediaPhoto())
    row = ing.message_to_row(1, m)
    assert row is not None and row["media_type"] == "photo" and row["text"] == ""
    assert not any(isinstance(v, bytes) for v in row.values())


def test_message_to_row_skips_empty_and_service() -> None:
    assert ing.message_to_row(1, _tl_message(message="")) is None
    assert ing.message_to_row(1, SimpleNamespace(id=1, action="join")) is None


def test_reactions_map() -> None:
    reactions = SimpleNamespace(
        results=[
            SimpleNamespace(count=3, reaction=SimpleNamespace(emoticon="🔥")),
            SimpleNamespace(count=1, reaction=SimpleNamespace(document_id=555)),
        ]
    )
    assert ing._reactions_map(reactions) == {"🔥": 3, "custom:555": 1}
    assert ing._reactions_total(reactions) == 4
    assert ing._reactions_map(None) is None


# ─── ramp-up ─────────────────────────────────────────────────────────────────


def test_limits_ramp_up_for_new_account() -> None:
    now = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
    fresh = ing.limits_for(now - timedelta(hours=2), now=now)
    assert fresh.ramp_up and fresh.max_messages == ing.RAMP_UP_MAX_MESSAGES
    assert fresh.max_chats == ing.RAMP_UP_MAX_CHATS
    old = ing.limits_for(now - timedelta(days=3), now=now)
    assert not old.ramp_up and old.max_messages == ing.DEFAULT_MAX_MESSAGES
    assert not ing.limits_for(None, now=now).ramp_up


# ─── snapshot tiers ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("hour", "n"),
    [(1, 1), (6, 2), (12, 2), (3, 2), (0, 2), (7, 1)],
)
def test_snapshot_tiers(hour: int, n: int) -> None:
    tiers = ing.snapshot_tiers(hour)
    assert len(tiers) == n
    assert tiers[0] == (timedelta(0), timedelta(days=1))  # yosh postlar har soat


def test_snapshot_tiers_daily_at_3() -> None:
    assert (timedelta(days=7), timedelta(days=90)) in ing.snapshot_tiers(3)
    assert (timedelta(days=7), timedelta(days=90)) not in ing.snapshot_tiers(15)


# ─── worker: FloodWait → defer ───────────────────────────────────────────────


async def test_sync_account_requeues_on_flood(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_enqueue(
        fn: str, *args: Any, job_id: str | None = None, defer_by: Any = None
    ) -> str:
        calls.append({"fn": fn, "args": args, "job_id": job_id, "defer_by": defer_by})
        return "j1"

    async def fake_refresh(account_id: int) -> int:
        raise ing.SyncPaused(120)

    class _Scope:
        async def __aenter__(self) -> Any:
            class _DB:
                async def get(self, *a: Any) -> Any:
                    return SimpleNamespace(created_at=datetime.now(UTC))

            return _DB()

        async def __aexit__(self, *a: Any) -> None: ...

    import app.db.base as dbbase

    monkeypatch.setattr(dbbase, "session_scope", lambda: _Scope())
    monkeypatch.setattr(tasks, "enqueue", fake_enqueue)
    monkeypatch.setattr(ing, "refresh_chats", fake_refresh)

    res = await tasks.sync_account({}, 9, True)
    assert res == {"paused": 120}
    assert calls and calls[0]["fn"] == "sync_account"
    assert calls[0]["args"] == (9, True)
    assert calls[0]["job_id"] == "sync:acc:9"
    assert calls[0]["defer_by"] == 125  # retry_after + 5


async def test_sync_chat_task_requeues_on_flood(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    async def fake_enqueue(fn: str, *args: Any, **kw: Any) -> str:
        calls.append((fn, args, kw))
        return "j"

    async def fake_sync(chat_id: int, max_messages: int | None = None) -> Any:
        raise ing.SyncPaused(5)

    monkeypatch.setattr(tasks, "enqueue", fake_enqueue)
    monkeypatch.setattr(ing, "sync_chat", fake_sync)
    res = await tasks.sync_chat({}, 3)
    assert res == {"paused": 5}
    assert calls[0][0] == "sync_chat" and calls[0][2]["defer_by"] == 35  # min 30 + 5


# ─── kontekst manbai ─────────────────────────────────────────────────────────


async def test_fetch_context_small_limit_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    async def live(account_id: int, peer_id: int, *, limit: int) -> Any:
        return "Chan", [SimpleNamespace(msg_id=1)]

    monkeypatch.setattr(cs.pool, "recent_messages", live)
    title, msgs, source = await cs.fetch_context(object(), 1, 2, 50)  # type: ignore[arg-type]
    assert (title, source, len(msgs)) == ("Chan", "live", 1)


async def test_fetch_context_large_limit_requires_db(monkeypatch: pytest.MonkeyPatch) -> None:
    async def none(*a: Any, **k: Any) -> None:
        return None

    monkeypatch.setattr(cs, "db_recent_messages", none)
    with pytest.raises(cs.ChatError) as ei:
        await cs.fetch_context(object(), 1, 2, 500)  # type: ignore[arg-type]
    assert ei.value.detail == "not_synced"


async def test_fetch_context_live_failure_falls_back_to_db(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.mtproto.pool import PoolError

    async def live(*a: Any, **k: Any) -> Any:
        raise PoolError("session_revoked")

    async def db(*a: Any, **k: Any) -> Any:
        return "Chan", [SimpleNamespace(msg_id=1), SimpleNamespace(msg_id=2)]

    monkeypatch.setattr(cs.pool, "recent_messages", live)
    monkeypatch.setattr(cs, "db_recent_messages", db)
    _title, msgs, source = await cs.fetch_context(object(), 1, 2, 50)  # type: ignore[arg-type]
    assert source == "db" and len(msgs) == 2
