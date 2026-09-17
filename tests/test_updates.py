"""Real-time updates listener (phase2/C) — soxta klient/eventlar bilan.

Haqiqiy Telegram YO'Q: klient ham, eventlar ham soxta. DB chaqiruvlari
monkeypatch qilinadi — testlar Postgres talab qilmaydi.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from telethon import errors as tg_errors

from app.worker import updates as up

ROOT = Path(__file__).resolve().parents[1]


# ─── soxta klient / eventlar ─────────────────────────────────────────────────


class FakeClient:
    def __init__(self, *, catch_up_exc: Exception | None = None, delay: float = 0.0) -> None:
        self.handlers: list[tuple[object, object]] = []
        self.catch_up_calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._exc = catch_up_exc
        self._delay = delay

    def add_event_handler(self, cb: object, event: object) -> None:
        self.handlers.append((cb, event))

    def remove_event_handler(self, cb: object, event: object = None) -> None:
        self.handlers = [(c, e) for c, e in self.handlers if c is not cb]

    async def catch_up(self) -> None:
        self.catch_up_calls += 1
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._exc is not None:
                raise self._exc
        finally:
            self.concurrent -= 1


def fake_message(msg_id: int, peer_raw: int, *, text: str = "salom", edit: bool = False):
    return SimpleNamespace(
        id=msg_id,
        peer_id=SimpleNamespace(channel_id=peer_raw),
        message=text,
        date=datetime(2026, 1, 1, tzinfo=UTC),
        edit_date=datetime(2026, 1, 2, tzinfo=UTC) if edit else None,
        media=None,
        from_id=None,
        reply_to=None,
        fwd_from=None,
        replies=None,
        reactions=None,
        grouped_id=None,
        pinned=False,
        views=None,
        forwards=None,
    )


@pytest.fixture
def listener(monkeypatch: pytest.MonkeyPatch) -> up.AccountUpdatesListener:
    monkeypatch.setattr(up, "load_chat_map", _fake_load_chat_map)
    lst = up.AccountUpdatesListener(account_id=1, chat_cache_s=3600)
    lst.client = FakeClient()
    lst._chat_map = {777: 42}
    lst._chat_map_at = 10**9  # kesh "yangi" — miss'da qayta yuklanmaydi
    return lst


async def _fake_load_chat_map(account_id: int) -> dict[int, int]:
    return {777: 42}


# ─── handler → DB yozuvi ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_new_message_upserts(
    listener: up.AccountUpdatesListener, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[int, object]] = []

    async def fake_upsert(chat_id: int, m: object) -> int:
        seen.append((chat_id, m))
        return 1

    monkeypatch.setattr(up, "upsert_message_from_update", fake_upsert)
    await listener._on_new(SimpleNamespace(message=fake_message(5, 777), chat_id=-1000000000777))
    assert len(seen) == 1
    assert seen[0][0] == 42  # chats.id
    assert listener.counters.new == 1


@pytest.mark.asyncio
async def test_edited_sets_edited_at(
    listener: up.AccountUpdatesListener, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`edited_at` `message.edit_date` dan keladi — ingestion mapping orqali."""
    from app.services.ingestion import message_to_row

    rows: list[dict[str, object]] = []

    async def fake_upsert(chat_id: int, m: object) -> int:
        row = message_to_row(chat_id, m)
        assert row is not None
        rows.append(row)
        return 1

    monkeypatch.setattr(up, "upsert_message_from_update", fake_upsert)
    msg = fake_message(6, 777, text="tuzatildi", edit=True)
    await listener._on_edited(SimpleNamespace(message=msg, chat_id=-1000000000777))
    assert rows[0]["edited_at"] == datetime(2026, 1, 2, tzinfo=UTC)
    assert rows[0]["tg_msg_id"] == 6
    assert listener.counters.edited == 1


@pytest.mark.asyncio
async def test_unknown_chat_ignored(
    listener: up.AccountUpdatesListener, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    async def fake_upsert(chat_id: int, m: object) -> int:
        calls.append(chat_id)
        return 1

    monkeypatch.setattr(up, "upsert_message_from_update", fake_upsert)
    await listener._on_new(SimpleNamespace(message=fake_message(7, 999), chat_id=-1000000000999))
    assert calls == []
    assert listener.counters.new == 0
    assert listener.counters.ignored == 1


@pytest.mark.asyncio
async def test_listener_never_fetches_dialogs(
    listener: up.AccountUpdatesListener, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Noma'lum chat uchun ham hech qanday MTProto so'rov ketmaydi."""

    async def boom(*a: object, **k: object) -> None:
        raise AssertionError("listener dialog tortmasligi kerak")

    monkeypatch.setattr(up.pool, "dialogs", boom, raising=False)
    monkeypatch.setattr(up, "upsert_message_from_update", boom)
    await listener._on_new(SimpleNamespace(message=fake_message(8, 12345), chat_id=None))


# ─── o'chirish: faqat yumshoq ────────────────────────────────────────────────


def test_soft_delete_stmt_is_update_not_delete() -> None:
    stmt = up.soft_delete_stmt([42], [1, 2, 3])
    sql = str(stmt.compile(compile_kwargs={"literal_binds": False})).upper()
    assert sql.startswith("UPDATE MESSAGES")
    assert "DELETED_AT" in sql
    assert "DELETE FROM" not in sql


def test_updates_module_has_no_delete_statement() -> None:
    src = (ROOT / "src" / "app" / "worker" / "updates.py").read_text(encoding="utf-8")
    assert "sa_delete" not in src
    assert "sqlalchemy import delete" not in src


@pytest.mark.asyncio
async def test_deleted_with_peer(
    listener: up.AccountUpdatesListener, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[list[int], list[int]]] = []

    async def fake_mark(chat_ids: list[int], ids: list[int]) -> int:
        seen.append((list(chat_ids), list(ids)))
        return len(ids)

    monkeypatch.setattr(up, "mark_deleted", fake_mark)
    await listener._on_deleted(SimpleNamespace(deleted_ids=[1, 2], chat_id=-1000000000777))
    assert seen == [([42], [1, 2])]
    assert listener.counters.deleted == 2


@pytest.mark.asyncio
async def test_deleted_without_peer_uses_private_chats(
    listener: up.AccountUpdatesListener, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_private(account_id: int) -> list[int]:
        return [11, 12]

    seen: list[list[int]] = []

    async def fake_mark(chat_ids: list[int], ids: list[int]) -> int:
        seen.append(list(chat_ids))
        return 1

    monkeypatch.setattr(up, "private_chat_ids", fake_private)
    monkeypatch.setattr(up, "mark_deleted", fake_mark)
    await listener._on_deleted(SimpleNamespace(deleted_ids=[9], chat_id=None))
    assert seen == [[11, 12]]


def test_raw_from_marked() -> None:
    assert up.raw_from_marked(-1000000000777) == 777
    assert up.raw_from_marked(-123456) == 123456
    assert up.raw_from_marked(555) == 555
    assert up.raw_from_marked(None) is None


# ─── catch_up ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_catch_up_floodwait_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    async def noop_hb(name: str, *, extra: object = None) -> None:
        return None

    monkeypatch.setattr(up, "heartbeat", noop_hb)
    exc = tg_errors.FloodWaitError(request=None)
    exc.seconds = 42
    client = FakeClient(catch_up_exc=exc)
    lst = up.AccountUpdatesListener(account_id=1)
    lst.client = client
    assert await lst.catch_up() is False
    assert client.catch_up_calls == 1  # qayta urinish YO'Q
    assert lst.counters.catchup == 0
    assert lst._catchup_busy is False  # keyingi slot ochiq qoladi


@pytest.mark.asyncio
async def test_only_one_catch_up_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    async def noop_hb(name: str, *, extra: object = None) -> None:
        return None

    monkeypatch.setattr(up, "heartbeat", noop_hb)
    client = FakeClient(delay=0.05)
    lst = up.AccountUpdatesListener(account_id=1)
    lst.client = client
    results = await asyncio.gather(*(lst.catch_up() for _ in range(4)))
    assert client.max_concurrent == 1
    assert client.catch_up_calls == 1
    assert results.count(True) == 1
    assert results.count(False) == 3


# ─── hayot sikli ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disabled_by_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    s = get_settings()
    assert s.tg_updates_listener is False  # default: o'chiq

    async def boom() -> list[int]:
        raise AssertionError("bayroq o'chiq — akkauntlar so'ralmasligi kerak")

    monkeypatch.setattr(up, "active_account_ids", boom)
    assert await up.start_listeners() == []
    assert up.active_listeners() == {}


@pytest.mark.asyncio
async def test_start_and_stop_registers_and_removes_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()

    async def fake_get(account_id: int) -> FakeClient:
        return client

    async def noop_hb(name: str, *, extra: object = None) -> None:
        return None

    monkeypatch.setattr(up.pool, "get", fake_get)
    monkeypatch.setattr(up, "load_chat_map", _fake_load_chat_map)
    monkeypatch.setattr(up, "heartbeat", noop_hb)

    lst = up.AccountUpdatesListener(account_id=1)
    await lst.start()
    assert len(client.handlers) == 3
    assert lst._chat_map == {777: 42}
    await lst.stop()
    assert client.handlers == []
    assert lst._task is None


@pytest.mark.asyncio
async def test_start_listeners_uses_flag_and_registers(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient()

    async def fake_get(account_id: int) -> FakeClient:
        return client

    async def fake_accounts() -> list[int]:
        return [7]

    async def noop_hb(name: str, *, extra: object = None) -> None:
        return None

    monkeypatch.setattr(up.pool, "get", fake_get)
    monkeypatch.setattr(up, "load_chat_map", _fake_load_chat_map)
    monkeypatch.setattr(up, "active_account_ids", fake_accounts)
    monkeypatch.setattr(up, "heartbeat", noop_hb)

    s = up.get_settings()
    monkeypatch.setattr(s, "tg_updates_listener", True)
    try:
        assert await up.start_listeners() == [7]
        assert 7 in up.active_listeners()
    finally:
        await up.stop_listeners()
    assert up.active_listeners() == {}


# ─── api jarayoni listener'ga tegmaydi ───────────────────────────────────────


def test_api_never_starts_listener() -> None:
    """`api` update oqimini iste'mol qilmaydi — bitta sessiya, bitta iste'molchi."""
    web = ROOT / "src" / "app" / "web"
    for path in web.rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "worker.updates" not in src, path
        assert "start_listeners" not in src, path


def test_worker_settings_wires_lifecycle() -> None:
    src = (ROOT / "src" / "app" / "worker" / "settings.py").read_text(encoding="utf-8")
    assert "start_listeners" in src
    assert "stop_listeners" in src
