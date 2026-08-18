"""Ulangan akkauntlar uchun klient pool.

Bitta akkaunt = bitta ulangan `GuardedTelegramClient` (lazy connect, qayta
foydalanish). Session DB'dan `session_store.load_session()` orqali ochiladi.

Bu modul faqat **o'qish** yordamchilarini beradi (dialoglar, oxirgi xabarlar).
Yozish yo'li 6-bosqichda `assert_writable()` bilan qo'shiladi.

Entity keshi: `StringSession` entity'larni saqlamaydi, shuning uchun peer'ga
murojaat qilishdan oldin `get_dialogs()` chaqirilib, InputPeer'lar xotirada
saqlanadi (`messages.GetDialogsRequest` — allowlist'da).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from telethon import errors as tg_errors
from telethon.tl.types import Channel, Chat, User

from app.db.base import session_scope
from app.db.models import Account, AccountStatus
from app.logging import get_logger
from app.mtproto.client import GuardedTelegramClient, build_client
from app.services.session_store import load_session, revoke_session

log = get_logger(__name__)

DIALOGS_TTL = 60.0  # soniya


class PoolError(RuntimeError):
    def __init__(self, code: str, message: str = "") -> None:
        self.code = code  # i18n: pool.err.<code>
        super().__init__(message or code)


@dataclass(frozen=True, slots=True)
class DialogInfo:
    peer_id: int
    title: str
    kind: str  # user | group | channel
    username: str | None
    unread: int
    last_message_at: datetime | None
    last_message_text: str


@dataclass(frozen=True, slots=True)
class MessageInfo:
    msg_id: int
    date: datetime | None
    sender: str
    text: str
    media_type: str | None
    views: int | None


@dataclass(slots=True)
class _Entry:
    client: GuardedTelegramClient
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    dialogs: list[DialogInfo] = field(default_factory=list)
    dialogs_at: float = 0.0
    input_peers: dict[int, Any] = field(default_factory=dict)
    titles: dict[int, str] = field(default_factory=dict)


class ClientPool:
    def __init__(self) -> None:
        self._entries: dict[int, _Entry] = {}
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock_for(self, account_id: int) -> asyncio.Lock:
        return self._locks.setdefault(account_id, asyncio.Lock())

    async def get(self, account_id: int) -> GuardedTelegramClient:
        entry = await self._entry(account_id)
        return entry.client

    async def _entry(self, account_id: int) -> _Entry:
        async with self._lock_for(account_id):
            entry = self._entries.get(account_id)
            if entry is not None and entry.client.is_connected():
                return entry

            async with session_scope() as db:
                account = await db.get(Account, account_id)
                if account is None:
                    raise PoolError("no_account")
                if account.status != AccountStatus.ACTIVE:
                    raise PoolError("account_inactive", account.status)
                session_string = load_session(account)
            if not session_string:
                raise PoolError("no_session")

            client = build_client(session_string, account_id=account_id)
            del session_string
            try:
                await client.connect()
                authorized = await client.is_user_authorized()
            except (OSError, ConnectionError, TimeoutError) as exc:
                raise PoolError("network", str(exc)[:100]) from exc

            if not authorized:
                await self._mark_revoked(account_id, client)
                raise PoolError("session_revoked")

            entry = _Entry(client=client)
            self._entries[account_id] = entry
            log.info("pool.connected", account_id=account_id)
            return entry

    async def _mark_revoked(self, account_id: int, client: GuardedTelegramClient) -> None:
        try:
            await client.disconnect()
        finally:
            async with session_scope() as db:
                account = await db.get(Account, account_id)
                if account is not None:
                    await revoke_session(db, account, status=AccountStatus.REVOKED)
            self._entries.pop(account_id, None)

    async def close(self, account_id: int) -> None:
        entry = self._entries.pop(account_id, None)
        if entry is not None:
            await entry.client.disconnect()

    async def close_all(self) -> None:
        for account_id in list(self._entries):
            await self.close(account_id)

    # ── o'qish yordamchilari ─────────────────────────────────────────────────

    async def dialogs(
        self, account_id: int, *, limit: int = 100, force: bool = False
    ) -> list[DialogInfo]:
        entry = await self._entry(account_id)
        async with entry.lock:
            if not force and entry.dialogs and time.monotonic() - entry.dialogs_at < DIALOGS_TTL:
                return entry.dialogs
            try:
                raw = await entry.client.get_dialogs(limit=limit)
            except tg_errors.AuthKeyUnregisteredError as exc:
                await self._mark_revoked(account_id, entry.client)
                raise PoolError("session_revoked") from exc
            except tg_errors.FloodWaitError as exc:
                raise PoolError("flood", str(exc.seconds)) from exc

            out: list[DialogInfo] = []
            for d in raw:
                ent = d.entity
                if ent is None:
                    continue
                pid = int(getattr(ent, "id", 0) or 0)
                info = DialogInfo(
                    peer_id=pid,
                    title=_title_of(ent),
                    kind=_kind_of(ent),
                    username=getattr(ent, "username", None),
                    unread=int(getattr(d, "unread_count", 0) or 0),
                    last_message_at=getattr(d.message, "date", None) if d.message else None,
                    last_message_text=(getattr(d.message, "message", "") or "")[:120]
                    if d.message
                    else "",
                )
                out.append(info)
                entry.input_peers[pid] = d.input_entity
                entry.titles[pid] = info.title

            entry.dialogs = out
            entry.dialogs_at = time.monotonic()
            return out

    async def recent_messages(
        self, account_id: int, peer_id: int, *, limit: int = 50
    ) -> tuple[str, list[MessageInfo]]:
        """(chat sarlavhasi, oxirgi xabarlar — eskidan yangiga)."""
        entry = await self._entry(account_id)
        if peer_id not in entry.input_peers:
            await self.dialogs(account_id, force=True)
        peer = entry.input_peers.get(peer_id)
        if peer is None:
            raise PoolError("no_dialog")

        try:
            msgs = await entry.client.get_messages(peer, limit=limit)
        except tg_errors.AuthKeyUnregisteredError as exc:
            await self._mark_revoked(account_id, entry.client)
            raise PoolError("session_revoked") from exc
        except tg_errors.FloodWaitError as exc:
            raise PoolError("flood", str(exc.seconds)) from exc

        out: list[MessageInfo] = []
        for m in reversed(list(msgs)):
            text = getattr(m, "message", "") or ""
            media = getattr(m, "media", None)
            if not text and media is None:
                continue
            out.append(
                MessageInfo(
                    msg_id=int(m.id),
                    date=getattr(m, "date", None),
                    sender=_sender_name(m),
                    text=text,
                    media_type=type(media).__name__.removeprefix("MessageMedia").lower()
                    if media is not None
                    else None,
                    views=getattr(m, "views", None),
                )
            )
        return entry.titles.get(peer_id, str(peer_id)), out


def _title_of(ent: Any) -> str:
    if isinstance(ent, User):
        name = " ".join(p for p in (ent.first_name, ent.last_name) if p)
        return name or (f"@{ent.username}" if ent.username else str(ent.id))
    return getattr(ent, "title", None) or str(getattr(ent, "id", ""))


def _kind_of(ent: Any) -> str:
    if isinstance(ent, User):
        return "user"
    if isinstance(ent, Channel):
        return "group" if getattr(ent, "megagroup", False) else "channel"
    if isinstance(ent, Chat):
        return "group"
    return "unknown"


def _sender_name(m: Any) -> str:
    # `m.sender` — javob bilan kelgan entity keshi, qo'shimcha so'rov yo'q.
    # (`get_sender()` `messages.GetChatsRequest` chaqirishi mumkin — allowlist'da yo'q.)
    sender = getattr(m, "sender", None)
    if sender is None:
        return "—"
    return _title_of(sender)


# jarayon davomida bitta pool
pool = ClientPool()
