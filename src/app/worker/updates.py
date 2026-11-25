"""Real-time updates listener (2-bosqich, arxitektura v2 §8.3).

Telethon update oqimini tinglaydi va `messages` jadvalini **jonli** yangilab
turadi: `NewMessage` → upsert, `MessageEdited` → matn + `edited_at`,
`MessageDeleted` → `deleted_at` (yumshoq o'chirish).

Qat'iy chegaralar
-----------------
* **Faqat worker jarayonida.** Bitta akkaunt = bitta MTProto sessiya =
  update oqimining bitta iste'molchisi. `api` bu modulni import ham qilmaydi
  (`tests/test_updates.py` buni tekshiradi) — aks holda ikki jarayon bir xil
  sessiyada update kutadi va `AuthKeyDuplicatedError` xavfi paydo bo'ladi.
* **Yangi TL metod yo'q.** Handler'lar faqat event bilan **kelgan** ma'lumotni
  ishlatadi: `event.message`, `event.deleted_ids`, `event.chat_id`.
  `get_sender()` / `get_chat()` **chaqirilmaydi** — ular allowlist'da yo'q
  metodlarga borishi mumkin. `client.catch_up()` esa `updates.GetDifference` /
  `updates.GetChannelDifference` ishlatadi — ular `allowlist.INTERNAL` da.
* **Retry-storm yo'q** (CLAUDE.md 4.2). `FloodWaitError` da qayta urinilmaydi:
  `mtproto.floodwait` logga tushadi va keyingi catch_up slotigacha kutiladi.
* **O'chirish faqat yumshoq.** Hech qanday `DELETE` SQL yo'q — faqat
  `UPDATE messages SET deleted_at = now()`. Shu sababli bu mantiq
  `services/` da emas, `worker/` da yashaydi (CI `services/` ichida
  `delete_*` yordamchilarini grep bilan taqiqlaydi).
* **Dialog tortilmaydi.** `chats` jadvalida yo'q (yoki `sync_state=failed`)
  peer'dan kelgan update **e'tiborsiz qoldiriladi** — listener hech qachon
  `GetDialogs` chaqirmaydi. Yangi chat 10 daqiqalik `refresh_chats` /
  `incremental_sync_all` cron'i orqali paydo bo'ladi (zaxira yo'l).

Downstream invalidatsiya — **avtomatik**, bu yerdan hech narsa chaqirilmaydi:
digest'lar `chat_digests.src_fingerprint` orqali (xabar to'plami o'zgarsa
fingerprint o'zgaradi va digest qayta hisoblanadi), epizod/tugunlar esa
`content_hash` bo'yicha qayta hisoblanadi. Listener faqat `messages` ni
yangilaydi; qolganini mavjud keshdan-chiqarish mexanizmlari bajaradi.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy import update as sa_update
from telethon import errors as tg_errors
from telethon import events

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Chat, ChatType, Message, SyncState
from app.logging import get_logger
from app.mtproto.pool import PoolError, pool
from app.observability import FLOODWAIT, heartbeat
from app.services.ingestion import active_account_ids, raw_peer_id, upsert_message_from_update

log = get_logger(__name__)

HEARTBEAT_NAME = "updates_listener"
CATCHUP_JITTER = 0.2  # ±20% — hamma akkaunt bir vaqtda GetDifference qilmasin
START_DELAY_MAX = 5.0  # soniya: startup'da birinchi catch_up ozgina kechiktiriladi


# ─── chat xaritasi (xom peer id → chats.id) ──────────────────────────────────


async def load_chat_map(account_id: int) -> dict[int, int]:
    """`{xom tg_peer_id: chats.id}` — `sync_state != failed` chatlar uchun."""
    async with session_scope() as db:
        rows = await db.execute(
            select(Chat.tg_peer_id, Chat.id).where(
                Chat.account_id == account_id,
                Chat.sync_state != SyncState.FAILED,
            )
        )
        return {int(peer): int(cid) for peer, cid in rows.all()}


async def private_chat_ids(account_id: int) -> list[int]:
    """Akkauntning shaxsiy chatlari (`chats.id`) — peer'siz o'chirish uchun."""
    async with session_scope() as db:
        rows = await db.execute(
            select(Chat.id).where(
                Chat.account_id == account_id,
                Chat.type == ChatType.PRIVATE,
                Chat.sync_state != SyncState.FAILED,
            )
        )
        return [int(r[0]) for r in rows.all()]


# ─── yumshoq o'chirish ───────────────────────────────────────────────────────


def soft_delete_stmt(
    chat_ids: Sequence[int], tg_msg_ids: Sequence[int], *, now: datetime | None = None
) -> Any:
    """`UPDATE messages SET deleted_at = ...` — **hech qachon** `DELETE`.

    Telegram xabarni o'chirganda bizdagi nusxa saqlanib qoladi, faqat
    belgilanadi: tarixiy tahlil (kim nima yozgan edi) buzilmaydi, retrieval
    esa `deleted_at IS NULL` bilan filtrlaydi.
    """
    return (
        sa_update(Message)
        .where(Message.chat_id.in_(list(chat_ids)), Message.tg_msg_id.in_(list(tg_msg_ids)))
        .values(deleted_at=now or datetime.now(UTC))
    )


async def mark_deleted(chat_ids: Sequence[int], tg_msg_ids: Sequence[int]) -> int:
    """Xabarlarni o'chirilgan deb belgilaydi. Qaytaradi: nechta qator tegildi."""
    if not chat_ids or not tg_msg_ids:
        return 0
    async with session_scope() as db:
        res = await db.execute(soft_delete_stmt(chat_ids, tg_msg_ids))
    return int(getattr(res, "rowcount", 0) or 0)


def raw_from_marked(chat_id: int | None) -> int | None:
    """Bot-API uslubidagi (`-100…`) id → xom id (`chats.tg_peer_id` fazosi).

    `link_prefix_for` bilan bir xil mantiq: `-100` faqat **manfiy** qiymatdan
    qirqiladi, musbat xom id'ga tegilmaydi.
    """
    if chat_id is None:
        return None
    if chat_id >= 0:
        return int(chat_id)
    raw = str(-int(chat_id))
    if raw.startswith("100") and len(raw) > 3:
        raw = raw[3:]
    return int(raw)


# ─── akkaunt listener'i ──────────────────────────────────────────────────────


@dataclass(slots=True)
class Counters:
    new: int = 0
    edited: int = 0
    deleted: int = 0
    catchup: int = 0
    ignored: int = 0


@dataclass(slots=True)
class AccountUpdatesListener:
    """Bitta akkauntning update oqimi."""

    account_id: int
    catchup_minutes: float = 30.0
    chat_cache_s: float = 60.0

    client: Any = None
    counters: Counters = field(default_factory=Counters)
    _handlers: list[tuple[Any, Any]] = field(default_factory=list)
    _task: asyncio.Task[None] | None = None
    _chat_map: dict[int, int] = field(default_factory=dict)
    _chat_map_at: float = 0.0
    _catchup_busy: bool = False
    _stopped: bool = False

    # ── hayot sikli ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        self.client = await pool.get(self.account_id)
        self._handlers = [
            (self._on_new, events.NewMessage()),
            (self._on_edited, events.MessageEdited()),
            (self._on_deleted, events.MessageDeleted()),
        ]
        for cb, ev in self._handlers:
            self.client.add_event_handler(cb, ev)
        self._chat_map = await load_chat_map(self.account_id)
        self._chat_map_at = _now()
        self._stopped = False
        self._task = asyncio.create_task(self._catchup_loop())
        log.info(
            "updates.started",
            account_id=self.account_id,
            chats=len(self._chat_map),
            catchup_minutes=self.catchup_minutes,
        )

    async def stop(self) -> None:
        self._stopped = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        for cb, ev in self._handlers:
            with contextlib.suppress(Exception):
                self.client.remove_event_handler(cb, ev)
        self._handlers = []
        log.info(
            "updates.stopped",
            account_id=self.account_id,
            new=self.counters.new,
            edited=self.counters.edited,
            deleted=self.counters.deleted,
        )

    # ── catch_up ─────────────────────────────────────────────────────────────

    async def _catchup_loop(self) -> None:
        """Startda bir marta, keyin `catchup_minutes` da bir — jitter bilan."""
        try:
            await asyncio.sleep(random.uniform(0.0, START_DELAY_MAX))  # noqa: S311
            await self.catch_up()
            while not self._stopped:
                base = max(60.0, self.catchup_minutes * 60.0)
                await asyncio.sleep(base * random.uniform(1 - CATCHUP_JITTER, 1 + CATCHUP_JITTER))  # noqa: S311
                await self.catch_up()
        except asyncio.CancelledError:  # pragma: no cover — shutdown yo'li
            raise

    async def catch_up(self) -> bool:
        """`updates.GetDifference` — uzilishda o'tkazib yuborilgan update'lar.

        Bir akkauntda bir vaqtda **bittadan ko'p emas**: oldingisi tugamagan
        bo'lsa chaqiruv jimgina o'tkazib yuboriladi (qaytadi `False`).
        """
        if self._catchup_busy or self.client is None:
            return False
        self._catchup_busy = True
        try:
            await self.client.catch_up()
        except tg_errors.FloodWaitError as exc:
            # Retry YO'Q — keyingi slotgacha kutamiz (4.2-band).
            FLOODWAIT.inc()
            log.warning(
                "mtproto.floodwait",
                account_id=self.account_id,
                op="catch_up",
                seconds=int(getattr(exc, "seconds", 0) or 0),
            )
            return False
        except (PoolError, tg_errors.RPCError, OSError, ConnectionError, TimeoutError) as exc:
            log.warning(
                "updates.catchup_failed",
                account_id=self.account_id,
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            return False
        else:
            self.counters.catchup += 1
            log.info("updates.catchup", account_id=self.account_id, count=self.counters.catchup)
            await heartbeat(
                HEARTBEAT_NAME,
                extra={
                    "account_id": self.account_id,
                    "new": self.counters.new,
                    "edited": self.counters.edited,
                    "deleted": self.counters.deleted,
                    "catchup": self.counters.catchup,
                },
            )
            return True
        finally:
            self._catchup_busy = False

    # ── chat xaritasi ────────────────────────────────────────────────────────

    async def chat_db_id(self, peer: int | None) -> int | None:
        """Xom peer id → `chats.id`. Noma'lum chat → `None` (dialog tortilmaydi)."""
        if peer is None:
            return None
        cid = self._chat_map.get(peer)
        if cid is not None:
            return cid
        if _now() - self._chat_map_at < self.chat_cache_s:
            return None  # kesh yangi — chat haqiqatan ham bizda yo'q
        self._chat_map = await load_chat_map(self.account_id)
        self._chat_map_at = _now()
        return self._chat_map.get(peer)

    # ── handler'lar ──────────────────────────────────────────────────────────

    async def _on_new(self, event: Any) -> None:
        await self._upsert(event, kind="new")

    async def _on_edited(self, event: Any) -> None:
        await self._upsert(event, kind="edited")

    async def _upsert(self, event: Any, *, kind: str) -> None:
        message = getattr(event, "message", None)
        if message is None:
            return
        peer = raw_peer_id(getattr(message, "peer_id", None))
        if peer is None:
            peer = raw_from_marked(getattr(event, "chat_id", None))
        chat_id = await self.chat_db_id(peer)
        if chat_id is None:
            self.counters.ignored += 1
            log.debug("updates.unknown_chat", account_id=self.account_id, peer=peer, kind=kind)
            return
        try:
            # `message_to_row` `edited_at` ni `message.edit_date` dan oladi.
            written = await upsert_message_from_update(chat_id, message)
        except Exception as exc:  # bitta xabar butun oqimni to'xtatmasin
            log.warning(
                "updates.upsert_failed",
                account_id=self.account_id,
                chat_id=chat_id,
                kind=kind,
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            return
        if not written:
            return
        if kind == "new":
            self.counters.new += 1
            log.info(
                "updates.new",
                account_id=self.account_id,
                chat_id=chat_id,
                count=self.counters.new,
            )
        else:
            self.counters.edited += 1
            log.info(
                "updates.edited",
                account_id=self.account_id,
                chat_id=chat_id,
                count=self.counters.edited,
            )

    async def _on_deleted(self, event: Any) -> None:
        ids = [int(i) for i in (getattr(event, "deleted_ids", None) or [])]
        if not ids:
            return
        peer = raw_from_marked(getattr(event, "chat_id", None))
        if peer is not None:
            chat_id = await self.chat_db_id(peer)
            if chat_id is None:
                self.counters.ignored += 1
                log.debug("updates.unknown_chat", account_id=self.account_id, peer=peer, kind="del")
                return
            chat_ids = [chat_id]
        else:
            # Shaxsiy chatlarda `updateDeleteMessages` peer bermaydi — id'lar
            # shu akkauntning **faqat private** chatlari ichida qidiriladi.
            chat_ids = await private_chat_ids(self.account_id)
            if not chat_ids:
                self.counters.ignored += 1
                return
        try:
            touched = await mark_deleted(chat_ids, ids)
        except Exception as exc:
            log.warning(
                "updates.delete_failed",
                account_id=self.account_id,
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            return
        self.counters.deleted += touched
        log.info(
            "updates.deleted",
            account_id=self.account_id,
            ids=len(ids),
            rows=touched,
            count=self.counters.deleted,
        )


def _now() -> float:
    return asyncio.get_running_loop().time()


# ─── supervisor (worker on_startup / on_shutdown) ────────────────────────────

_listeners: dict[int, AccountUpdatesListener] = {}


def active_listeners() -> dict[int, AccountUpdatesListener]:
    return _listeners


async def start_listeners() -> list[int]:
    """Barcha faol akkauntlar uchun listener. Bayroq o'chiq bo'lsa — hech narsa.

    Faqat **worker** `on_startup` dan chaqiriladi.
    """
    s = get_settings()
    if not s.tg_updates_listener:
        log.info("updates.disabled")
        return []
    started: list[int] = []
    for account_id in await active_account_ids():
        if account_id in _listeners:
            continue
        listener = AccountUpdatesListener(
            account_id=account_id,
            catchup_minutes=float(s.tg_updates_catchup_minutes),
            chat_cache_s=float(s.tg_updates_chat_cache_s),
        )
        try:
            await listener.start()
        except (PoolError, tg_errors.RPCError, OSError, ConnectionError, TimeoutError) as exc:
            log.warning(
                "updates.start_failed",
                account_id=account_id,
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
            )
            continue
        _listeners[account_id] = listener
        started.append(account_id)
    if started:
        await heartbeat(HEARTBEAT_NAME, extra={"accounts": started})
    return started


async def stop_listeners() -> None:
    for account_id in list(_listeners):
        listener = _listeners.pop(account_id)
        with contextlib.suppress(Exception):
            await listener.stop()
