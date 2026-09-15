"""Ingestion — 2-bosqich: chat registry, tarix sync, metrik snapshot.

Uch mustaqil ish (hammasi faqat O'QISH — allowlist READ to'plami):

  refresh_chats(account_id)      dialoglar → `chats` (upsert, sync holatiga tegmaydi)
  sync_chat(chat_id, ...)        tarix → `messages` (yangidan eskiga, batch upsert)
  snapshot_chat_metrics(chat_id) kanal postlari views/forwards/reactions → snapshot

Ban riski (rejaning 4.2-bandi):
  * batch'lar orasida `wait_time` pauza, Telethon flood_sleep_threshold=60;
    undan uzun FloodWait — ish to'xtaydi, `sync_error` yoziladi, worker
    `retry_after` bilan qayta navbatga qo'yadi (retry-storm yo'q).
  * **Ramp-up**: akkaunt yoshi < 24 soat → har chatda ≤ 1000 xabar, faqat
    eng faol 20 dialog. Keyin sekin kengaytiriladi.
  * Media yuklab olinmaydi — faqat `media_type`.

Snapshot (4.6-band): faqat kanallar; yosh postlar tez-tez, eskilari kamroq —
`snapshot_tiers()`. Bu cron'ni o'chirmang: bo'shliq abadiy qoladi.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from telethon import errors as tg_errors
from telethon.tl.types import Channel
from telethon.tl.types import Chat as TlChat
from telethon.tl.types import Message as TlMessage
from telethon.tl.types import User as TlUser

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import (
    Account,
    Chat,
    ChatType,
    Message,
    MessageMetricSnapshot,
    SyncState,
)
from app.logging import get_logger
from app.mtproto.pool import DIALOGS_LIMIT, PoolError, pool
from app.observability import FLOODWAIT, SNAPSHOT_POSTS, SYNC_MESSAGES
from app.services.stopwords import STOP_EN, STOP_RU, STOP_UZ
from app.services.textnorm import strip_invisible, uz_fold

log = get_logger(__name__)

BATCH = 100  # GetHistory maksimal
WAIT_BETWEEN_BATCHES = 0.7  # soniya — flood-wait'siz ~90 so'rov/daqiqa
RAMP_UP_HOURS = 24
RAMP_UP_MAX_MESSAGES = 1000
RAMP_UP_MAX_CHATS = 20
DEFAULT_MAX_MESSAGES = 50_000  # bitta ishda; keyingi ish davom ettiradi
SNAPSHOT_BATCH = 100


class SyncPaused(Exception):
    """FloodWait — worker `retry_after` soniyadan keyin qayta urinadi."""

    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"flood wait {retry_after}s")


@dataclass(slots=True)
class SyncReport:
    chat_id: int
    fetched: int = 0
    inserted_or_updated: int = 0
    newest_id: int | None = None
    oldest_id: int | None = None
    done: bool = False  # tarix boshigacha yetildi
    error: str | None = None


# ─── chat registry ───────────────────────────────────────────────────────────


def _chat_type(ent: Any) -> str:
    if isinstance(ent, TlUser):
        return ChatType.PRIVATE
    if isinstance(ent, Channel):
        return ChatType.SUPERGROUP if getattr(ent, "megagroup", False) else ChatType.CHANNEL
    if isinstance(ent, TlChat):
        return ChatType.GROUP
    return ChatType.GROUP


def _title(ent: Any) -> str:
    if isinstance(ent, TlUser):
        name = " ".join(p for p in (ent.first_name, ent.last_name) if p)
        return name or (f"@{ent.username}" if ent.username else str(ent.id))
    return getattr(ent, "title", None) or str(getattr(ent, "id", ""))


def link_prefix_for(chat_type: str, tg_peer_id: int, username: str | None) -> str | None:
    """Sitata havolasi prefiksi: `<prefix><tg_msg_id>` → to'liq t.me havolasi.

    Sync vaqtida bir marta hisoblanadi (`chats.link_prefix`) — aks holda har
    sitata uchun runtime'da peer turini qayta tekshirish kerak bo'ladi.

    * public (username bor) → ``https://t.me/<username>/``
    * kanal/supergruh        → ``https://t.me/c/<id>/`` (``-100`` prefiksisiz)
    * private / oddiy guruh  → ``None`` (bunday chatga ommaviy havola yo'q)

    `chats.tg_peer_id` odatda **xom** Telethon id (``ent.id``, musbat) bo'ladi,
    lekin bot API uslubidagi ``-100…`` ko'rinish ham kelishi mumkin. ``-100``
    faqat **manfiy** qiymatdan qirqiladi: musbat xom id'da ``100`` bilan
    boshlanuvchi haqiqiy kanallar bor (masalan ``1001234567``) va ularni
    qirqish noto'g'ri havola beradi.
    """
    if username:
        return f"https://t.me/{username}/"
    if chat_type in (ChatType.CHANNEL, ChatType.SUPERGROUP):
        peer = int(tg_peer_id)
        raw = str(-peer) if peer < 0 else str(peer)
        if peer < 0 and raw.startswith("100") and len(raw) > 3:
            raw = raw[3:]
        return f"https://t.me/c/{raw}/"
    return None


# ─── til aniqlash ────────────────────────────────────────────────────────────

MIN_CHARS_FOR_LANG = 12  # undan qisqa matnda har qanday detektor tanga tashlaydi

#: Faqat o'zbekchaga xos stop-so'zlar (fold fazosida). `lingua` da **o'zbek
#: tili yo'q** (75 tilning ichida emas), shuning uchun uz ↔ ru/en ajratish
#: marker so'zlar + yozuv (script) tahlili bilan qilinadi, lingua esa faqat
#: ru/en ni tasdiqlaydi. Aniqlik mukammal emas — `lang_conf` shuning uchun
#: saqlanadi, filtr emas, **vazn** sifatida ishlatilsin.
_UZ_MARKERS: frozenset[str] = STOP_UZ - STOP_RU - STOP_EN

_TOKEN_RE = re.compile(r"[a-zЀ-ӿ]+")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


@lru_cache(maxsize=1)
def _detector() -> Any:
    """`lingua` detektori (ru/en), low-accuracy rejimida. Yo'q bo'lsa — None."""
    try:
        from lingua import Language, LanguageDetectorBuilder
    except ImportError:  # pragma: no cover — dev muhitda paket bo'lmasa
        log.warning("ingest.lingua_missing")
        return None
    langs = [Language.RUSSIAN, Language.ENGLISH]
    return LanguageDetectorBuilder.from_languages(*langs).with_low_accuracy_mode().build()


def _lingua_conf(text: str) -> dict[str, float]:
    det = _detector()
    if det is None:
        return {}
    try:
        values = det.compute_language_confidence_values(text)
    except Exception:  # pragma: no cover — detektor xatosi ingest'ni to'xtatmasin
        return {}
    return {v.language.iso_code_639_1.name.lower(): float(v.value) for v in values}


def detect_lang(text: str) -> tuple[str | None, float | None]:
    """Xabar tili: ``uz`` | ``ru`` | ``en`` | ``other`` (+ ishonch 0..1).

    < ``MIN_CHARS_FOR_LANG`` belgili matn uchun ``(None, None)`` — aniqlash
    o'tkazib yuboriladi (qisqa matnda natija shovqin).

    Args:
        text: tozalangan xabar matni.

    Returns:
        ``(lang, confidence)`` juftligi; aniqlanmasa ``("other", conf)``.
    """
    body = (text or "").strip()
    if len(body) < MIN_CHARS_FOR_LANG:
        return None, None

    folded = uz_fold(body)
    tokens = _TOKEN_RE.findall(folded)
    if not tokens:
        return "other", 0.0
    uz_hits = sum(1 for t in tokens if t in _UZ_MARKERS)
    uz_ratio = uz_hits / len(tokens)

    conf = _lingua_conf(body)
    ru, en = conf.get("ru", 0.0), conf.get("en", 0.0)
    cyr = len(_CYRILLIC_RE.findall(body))
    lat = len(_LATIN_RE.findall(body))

    # 1) O'zbekcha marker so'zlar — ikkala yozuvda ham ishlaydi (fold tufayli).
    if uz_hits >= 2 or (uz_hits >= 1 and len(tokens) <= 8):
        return "uz", round(min(0.99, 0.55 + uz_ratio), 3)
    # 2) Kirill ustun → rus (lingua tasdiqlasa), aks holda noaniq.
    if cyr > lat:
        return ("ru", round(ru, 3)) if ru >= 0.55 else ("other", round(ru, 3))
    # 3) Lotin ustun → ingliz (lingua tasdiqlasa), aks holda o'zbek lotin.
    if lat > cyr:
        if en >= 0.65:
            return "en", round(en, 3)
        return "uz", round(max(0.3, 1.0 - en), 3)
    return "other", 0.0


def _is_admin(ent: Any) -> bool:
    return bool(getattr(ent, "creator", False) or getattr(ent, "admin_rights", None))


def _is_writable(ent: Any) -> bool:
    if isinstance(ent, Channel) and not getattr(ent, "megagroup", False):
        rights = getattr(ent, "admin_rights", None)
        return bool(
            getattr(ent, "creator", False) or (rights and getattr(rights, "post_messages", False))
        )
    return not bool(getattr(ent, "left", False))


async def refresh_chats(account_id: int, *, limit: int = DIALOGS_LIMIT) -> int:
    """Dialoglarni `chats` jadvaliga upsert qiladi. Qaytaradi: nechta yozuv."""
    try:
        client = await pool.get(account_id)
        dialogs = await client.get_dialogs(limit=limit)
    except tg_errors.FloodWaitError as exc:
        raise SyncPaused(int(exc.seconds)) from exc

    rows: list[dict[str, Any]] = []
    for d in dialogs:
        ent = d.entity
        if ent is None:
            continue
        ctype = _chat_type(ent)
        uname = getattr(ent, "username", None)
        rows.append(
            {
                "account_id": account_id,
                "tg_peer_id": int(ent.id),
                "access_hash": getattr(ent, "access_hash", None),
                "type": ctype,
                "title": _title(ent)[:256],
                "username": uname,
                "link_prefix": link_prefix_for(ctype, int(ent.id), uname),
                "is_writable": _is_writable(ent),
                "is_admin": _is_admin(ent),
                "participants_count": getattr(ent, "participants_count", None),
                "last_message_at": getattr(d.message, "date", None) if d.message else None,
                # yangi chat uchun DEFAULT_WRITE_MODE; mavjudlarga tegilmaydi (upsert set_ da yo'q)
                "write_mode": str(get_settings().default_write_mode),
            }
        )
    if not rows:
        return 0

    async with session_scope() as db:
        stmt = pg_insert(Chat).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_chat_per_account",
            set_={
                "access_hash": stmt.excluded.access_hash,
                "type": stmt.excluded.type,
                "title": stmt.excluded.title,
                "username": stmt.excluded.username,
                "link_prefix": stmt.excluded.link_prefix,
                "is_writable": stmt.excluded.is_writable,
                "is_admin": stmt.excluded.is_admin,
                "participants_count": stmt.excluded.participants_count,
                "last_message_at": stmt.excluded.last_message_at,
            },
        )
        await db.execute(stmt)
    log.info("ingest.chats_refreshed", account_id=account_id, count=len(rows))
    return len(rows)


# ─── xabar → qator ───────────────────────────────────────────────────────────


def _peer_id(peer: Any) -> int | None:
    if peer is None:
        return None
    for attr in ("user_id", "channel_id", "chat_id"):
        val = getattr(peer, attr, None)
        if val is not None:
            return int(val)
    return None


def _media_type(media: Any) -> str | None:
    if media is None:
        return None
    name = type(media).__name__  # MessageMediaPhoto → photo
    return name.removeprefix("MessageMedia").lower()[:24] or None


def _reactions_total(reactions: Any) -> int | None:
    results = getattr(reactions, "results", None)
    if not results:
        return None
    return sum(int(getattr(r, "count", 0) or 0) for r in results)


def _reactions_map(reactions: Any) -> dict[str, int] | None:
    results = getattr(reactions, "results", None)
    if not results:
        return None
    out: dict[str, int] = {}
    for r in results:
        emoji = getattr(getattr(r, "reaction", None), "emoticon", None)
        key = emoji or f"custom:{getattr(getattr(r, 'reaction', None), 'document_id', '?')}"
        out[key] = int(getattr(r, "count", 0) or 0)
    return out


def _fwd_peer_id(fwd: Any) -> int | None:
    """Forward manba peer'i — to'liq (`-100…`) ko'rinishda, yangi RPC'siz.

    `from_name` (maxfiylik sozlamasi yopiq bo'lsa keladigan matn) **e'tiborga
    olinmaydi**: u peer emas, va uni ism sifatida saqlash foydasiz.
    """
    peer = getattr(fwd, "from_id", None)
    if peer is None:
        return None
    try:
        from telethon import utils as tl_utils

        return int(tl_utils.get_peer_id(peer))
    except (TypeError, ValueError, AttributeError):
        return _peer_id(peer)


def _top_msg_id(reply_to: Any) -> int | None:
    """Forum topic / thread ildizi.

    Telethon `MessageReplyHeader`: `reply_to_top_id` — thread ildizi;
    forum topic'da `forum_topic=True` bo'lsa va `reply_to_top_id` bo'sh bo'lsa,
    topic ildizi `reply_to_msg_id` ning o'zi bo'ladi.
    """
    if reply_to is None:
        return None
    top = getattr(reply_to, "reply_to_top_id", None)
    if top is not None:
        return int(top)
    if getattr(reply_to, "forum_topic", False):
        root = getattr(reply_to, "reply_to_msg_id", None)
        return int(root) if root is not None else None
    return None


def message_to_row(chat_id: int, m: Any) -> dict[str, Any] | None:
    """Telethon Message → `messages` qatori. Service/bo'sh xabarlar → None.

    Matn DB'ga **tozalangan** holda yoziladi (`textnorm.strip_invisible`):
    zero-width, bidi va Unicode Tags belgilari saqlanmaydi. Bu
    `services/untrusted.neutralize` ni almashtirmaydi — u prompt
    chegarasida, bu esa saqlash chegarasida ishlaydi.
    """
    if not isinstance(m, TlMessage) and not hasattr(m, "message"):
        return None
    text = strip_invisible(getattr(m, "message", None) or "")
    media = getattr(m, "media", None)
    if not text and media is None:
        return None
    replies = getattr(m, "replies", None)
    fwd = getattr(m, "fwd_from", None)
    reply_to = getattr(m, "reply_to", None)
    lang, lang_conf = detect_lang(text)
    return {
        "chat_id": chat_id,
        "tg_msg_id": int(m.id),
        "sender_id": _peer_id(getattr(m, "from_id", None)),
        "published_at": m.date,
        "edited_at": getattr(m, "edit_date", None),
        "text": text,
        "media_type": _media_type(media),
        "reply_to_msg_id": getattr(reply_to, "reply_to_msg_id", None) if reply_to else None,
        "fwd_from_id": _peer_id(getattr(fwd, "from_id", None)) if fwd else None,
        "fwd_from_peer_id": _fwd_peer_id(fwd),
        "fwd_from_msg_id": getattr(fwd, "channel_post", None) if fwd else None,
        "fwd_date": getattr(fwd, "date", None) if fwd else None,
        "top_msg_id": _top_msg_id(reply_to),
        "lang": lang,
        "lang_conf": lang_conf,
        "grouped_id": getattr(m, "grouped_id", None),
        "is_pinned": bool(getattr(m, "pinned", False)),
        "views": getattr(m, "views", None),
        "forwards": getattr(m, "forwards", None),
        "reactions_total": _reactions_total(getattr(m, "reactions", None)),
        "replies_count": getattr(replies, "replies", None) if replies else None,
    }


async def upsert_messages(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    async with session_scope() as db:
        stmt = pg_insert(Message).values(rows)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_msg_per_chat",
            set_={
                "text": stmt.excluded.text,
                "edited_at": stmt.excluded.edited_at,
                "media_type": stmt.excluded.media_type,
                # matn o'zgargan bo'lsa til ham qayta hisoblanadi (text_fold/tsv_*
                # generated — ularni DB o'zi yangilaydi)
                "lang": stmt.excluded.lang,
                "lang_conf": stmt.excluded.lang_conf,
                "fwd_from_peer_id": stmt.excluded.fwd_from_peer_id,
                "fwd_from_msg_id": stmt.excluded.fwd_from_msg_id,
                "fwd_date": stmt.excluded.fwd_date,
                "top_msg_id": stmt.excluded.top_msg_id,
                "is_pinned": stmt.excluded.is_pinned,
                "views": stmt.excluded.views,
                "forwards": stmt.excluded.forwards,
                "reactions_total": stmt.excluded.reactions_total,
                "replies_count": stmt.excluded.replies_count,
            },
        )
        await db.execute(stmt)
    return len(rows)


# ─── ramp-up ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SyncLimits:
    max_messages: int
    max_chats: int
    ramp_up: bool


def limits_for(account_created_at: datetime | None, *, now: datetime | None = None) -> SyncLimits:
    now = now or datetime.now(UTC)
    if account_created_at is not None:
        created = (
            account_created_at
            if account_created_at.tzinfo
            else account_created_at.replace(tzinfo=UTC)
        )
        if now - created < timedelta(hours=RAMP_UP_HOURS):
            return SyncLimits(RAMP_UP_MAX_MESSAGES, RAMP_UP_MAX_CHATS, True)
    return SyncLimits(DEFAULT_MAX_MESSAGES, 10_000, False)


# ─── tarix sync ──────────────────────────────────────────────────────────────


async def sync_chat(chat_id: int, *, max_messages: int | None = None) -> SyncReport:
    """Bitta chat tarixini sinxronlaydi.

    Strategiya:
      1. yangi xabarlar: `min_id=synced_max_id` (incremental) — har doim.
      2. orqaga to'ldirish: `offset_id=synced_min_id` — hali boshiga yetilmagan bo'lsa.
    Har batch DB'ga darhol yoziladi (jarayon o'lsa progress yo'qolmaydi).
    """
    report = SyncReport(chat_id=chat_id)
    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None:
            report.error = "chat topilmadi"
            return report
        account = await db.get(Account, chat.account_id)
        limits = limits_for(account.created_at if account else None)
        budget = max_messages or limits.max_messages
        account_id, peer_id = chat.account_id, chat.tg_peer_id
        synced_min, synced_max = chat.synced_min_id, chat.synced_max_id
        chat.sync_state = SyncState.RUNNING
        chat.sync_error = None

    try:
        client = await pool.get(account_id)
        peer = await _input_peer(account_id, peer_id)

        # 1) yangi xabarlar (yangidan eskiga; min_id — undan kattalar keladi)
        if synced_max:
            fetched = await _pull(client, peer, chat_id, report, budget, min_id=synced_max)
            budget -= fetched
        # 2) orqaga to'ldirish yoki birinchi to'liq sync
        if budget > 0 and (synced_min is None or synced_min > 1):
            fetched = await _pull(
                client, peer, chat_id, report, budget, offset_id=synced_min or 0, backfill=True
            )
            budget -= fetched
        if not synced_max and not synced_min and report.fetched == 0:
            report.done = True  # bo'sh chat
        await _finish(chat_id, report, state=SyncState.IDLE)
    except tg_errors.FloodWaitError as exc:
        FLOODWAIT.inc()
        report.error = f"flood_wait:{exc.seconds}"
        await _finish(chat_id, report, state=SyncState.IDLE, error=report.error)
        raise SyncPaused(int(exc.seconds)) from exc
    except (PoolError, tg_errors.RPCError, OSError, ConnectionError, TimeoutError) as exc:
        report.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        await _finish(chat_id, report, state=SyncState.FAILED, error=report.error)
        log.warning("ingest.sync_failed", chat_id=chat_id, error=report.error)
    return report


async def _input_peer(account_id: int, peer_id: int) -> Any:
    return await pool.input_peer(account_id, peer_id)


async def _pull(
    client: Any,
    peer: Any,
    chat_id: int,
    report: SyncReport,
    budget: int,
    *,
    min_id: int = 0,
    offset_id: int = 0,
    backfill: bool = False,
) -> int:
    """Telethon iteratori orqali batch'lab tortadi va yozadi. Qaytaradi: nechta xabar."""
    fetched = 0
    batch: list[dict[str, Any]] = []
    total_estimate: int | None = None
    it = client.iter_messages(
        peer, limit=budget, min_id=min_id, offset_id=offset_id, wait_time=WAIT_BETWEEN_BATCHES
    )
    async for m in it:
        if total_estimate is None:
            total_estimate = getattr(it, "total", None)
        fetched += 1
        row = message_to_row(chat_id, m)
        if row is not None:
            batch.append(row)
        report.newest_id = max(report.newest_id or 0, int(m.id))
        report.oldest_id = min(report.oldest_id or int(m.id), int(m.id))
        if len(batch) >= BATCH:
            report.inserted_or_updated += await upsert_messages(batch)
            await _progress(chat_id, report, total_estimate)
            batch = []
    if batch:
        report.inserted_or_updated += await upsert_messages(batch)
    report.fetched += fetched
    if fetched:
        SYNC_MESSAGES.inc(fetched)
    if backfill and fetched < budget:
        report.done = True  # tarix boshiga yetildi (limitgacha yetmay tugadi)
    await _progress(chat_id, report, total_estimate)
    return fetched


async def _progress(chat_id: int, report: SyncReport, total_estimate: int | None) -> None:
    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None:
            return
        if report.newest_id is not None:
            chat.synced_max_id = max(chat.synced_max_id or 0, report.newest_id)
        if report.oldest_id is not None:
            chat.synced_min_id = (
                report.oldest_id
                if chat.synced_min_id is None
                else min(chat.synced_min_id, report.oldest_id)
            )
        if total_estimate:
            chat.total_estimate = int(total_estimate)
        chat.synced_total = (
            await db.execute(select(func.count(Message.id)).where(Message.chat_id == chat_id))
        ).scalar_one()


async def _finish(
    chat_id: int, report: SyncReport, *, state: str, error: str | None = None
) -> None:
    """Yakuniy holat. `synced_min_id == 1` — tarix boshigacha yetilgan (sentinel)."""
    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None:
            return
        if report.done:
            chat.synced_min_id = 1
        if state == SyncState.IDLE and chat.synced_min_id == 1:
            state = SyncState.DONE
        chat.sync_state = state
        chat.sync_error = error
        chat.last_sync_at = datetime.now(UTC)
    log.info(
        "ingest.sync_done",
        chat_id=chat_id,
        fetched=report.fetched,
        upserted=report.inserted_or_updated,
        state=state,
        error=error,
    )


# ─── snapshot ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SnapshotPlan:
    """Qaysi yoshdagi postlar shu soatda snapshot olinadi."""

    hour: int
    max_age_days: int = 90
    ids_by_age: dict[str, list[int]] = field(default_factory=dict)


def snapshot_tiers(hour: int) -> list[tuple[timedelta, timedelta]]:
    """(dan, gacha) yosh oraliqlari — shu soatda snapshot olinadigan postlar.

    < 24 soat  — har soat
    1-7 kun    — har 6 soat (0, 6, 12, 18)
    7-90 kun   — kuniga bir marta (03:00)
    """
    tiers = [(timedelta(0), timedelta(days=1))]
    if hour % 6 == 0:
        tiers.append((timedelta(days=1), timedelta(days=7)))
    if hour == 3:
        tiers.append((timedelta(days=7), timedelta(days=90)))
    return tiers


async def snapshot_chat_metrics(chat_id: int, *, now: datetime | None = None) -> int:
    """Kanal postlari uchun views/forwards/reactions snapshot. Qaytaradi: nechta post."""
    now = now or datetime.now(UTC)
    tiers = snapshot_tiers(now.hour)

    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None or chat.type != ChatType.CHANNEL:
            return 0
        account_id, peer_id = chat.account_id, chat.tg_peer_id
        ids: list[int] = []
        id_map: dict[int, int] = {}  # tg_msg_id → messages.id
        for lo, hi in tiers:
            rows = await db.execute(
                select(Message.id, Message.tg_msg_id).where(
                    Message.chat_id == chat_id,
                    Message.published_at <= now - lo,
                    Message.published_at > now - hi,
                )
            )
            for mid, tg_id in rows.all():
                ids.append(int(tg_id))
                id_map[int(tg_id)] = int(mid)
    if not ids:
        return 0

    try:
        client = await pool.get(account_id)
        peer = await _input_peer(account_id, peer_id)
    except tg_errors.FloodWaitError as exc:
        raise SyncPaused(int(exc.seconds)) from exc

    captured = 0
    for chunk in _chunks(ids, SNAPSHOT_BATCH):
        try:
            msgs = await client.get_messages(peer, ids=chunk)
        except tg_errors.FloodWaitError as exc:
            raise SyncPaused(int(exc.seconds)) from exc
        snaps: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        for m in msgs or []:
            if m is None or not hasattr(m, "id") or int(m.id) not in id_map:
                continue
            reactions = getattr(m, "reactions", None)
            replies = getattr(m, "replies", None)
            snaps.append(
                {
                    "message_id": id_map[int(m.id)],
                    "captured_at": now,
                    "views": getattr(m, "views", None),
                    "forwards": getattr(m, "forwards", None),
                    "replies_count": getattr(replies, "replies", None) if replies else None,
                    "reactions_total": _reactions_total(reactions),
                    "reactions": _reactions_map(reactions),
                }
            )
            updates.append(
                {
                    "b_id": id_map[int(m.id)],
                    "views": getattr(m, "views", None),
                    "forwards": getattr(m, "forwards", None),
                    "reactions_total": _reactions_total(reactions),
                    "replies_count": getattr(replies, "replies", None) if replies else None,
                }
            )
        if snaps:
            async with session_scope() as db:
                await db.execute(pg_insert(MessageMetricSnapshot).values(snaps))
                for u in updates:
                    await db.execute(
                        update(Message)
                        .where(Message.id == u["b_id"])
                        .values(
                            views=u["views"],
                            forwards=u["forwards"],
                            reactions_total=u["reactions_total"],
                            replies_count=u["replies_count"],
                        )
                    )
            captured += len(snaps)
    if captured:
        SNAPSHOT_POSTS.inc(captured)
    log.info("ingest.snapshot", chat_id=chat_id, posts=captured, tiers=len(tiers))
    return captured


def _chunks(items: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ─── akkaunt darajasi ────────────────────────────────────────────────────────


async def chats_to_sync(account_id: int, *, limits: SyncLimits) -> list[int]:
    """Sync tartibi: oxirgi xabari yangi bo'lganlar oldin; ramp-up'da top-N."""
    async with session_scope() as db:
        rows = await db.execute(
            select(Chat.id)
            .where(Chat.account_id == account_id, Chat.sync_state != SyncState.RUNNING)
            .order_by(Chat.last_message_at.desc().nulls_last(), Chat.id)
            .limit(limits.max_chats)
        )
        return [int(r[0]) for r in rows.all()]


async def channel_ids(account_id: int | None = None) -> list[int]:
    async with session_scope() as db:
        q = select(Chat.id).where(Chat.type == ChatType.CHANNEL, Chat.synced_total > 0)
        if account_id is not None:
            q = q.where(Chat.account_id == account_id)
        return [int(r[0]) for r in (await db.execute(q)).all()]


async def active_account_ids() -> list[int]:
    from app.db.models import AccountStatus

    async with session_scope() as db:
        rows = await db.execute(select(Account.id).where(Account.status == AccountStatus.ACTIVE))
        return [int(r[0]) for r in rows.all()]
