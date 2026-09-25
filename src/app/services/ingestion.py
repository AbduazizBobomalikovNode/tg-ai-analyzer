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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy import case, exists, func, null, select, update
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
    Sender,
    SyncState,
)
from app.logging import get_logger
from app.mtproto.pool import DIALOGS_LIMIT, PoolError, pool
from app.observability import FLOODWAIT, SNAPSHOT_POSTS, SYNC_MESSAGES
from app.services import sync_progress as SP
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
    #: Redis'ga batch progressini yozuvchi (`services/sync_progress.ChatTracker`).
    #: `None` — ko'rsatkich o'chiq (test, yoki chat topilmagan); sync baribir ishlaydi.
    tracker: Any = field(default=None, repr=False, compare=False)


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

    * public (username bor), **shaxsiy emas** → ``https://t.me/<username>/``
    * kanal/supergruh        → ``https://t.me/c/<id>/`` (``-100`` prefiksisiz)
    * shaxsiy (`private`) / oddiy guruh → ``None`` — bunday chatga ommaviy
      havola yo'q. **Shaxsiy chat username bilan ham** (jonli regressiya:
      "Otabek Kursdosh" javobida `t.me/JOO571/<msg>` chiqdi) — `@username/<id>`
      shakli faqat ochiq kanal/supergruh postiga ishlaydi, foydalanuvchining
      DM tarixiga emas: bunday havola ochilmaydi.

    `chats.tg_peer_id` odatda **xom** Telethon id (``ent.id``, musbat) bo'ladi,
    lekin bot API uslubidagi ``-100…`` ko'rinish ham kelishi mumkin. ``-100``
    faqat **manfiy** qiymatdan qirqiladi: musbat xom id'da ``100`` bilan
    boshlanuvchi haqiqiy kanallar bor (masalan ``1001234567``) va ularni
    qirqish noto'g'ri havola beradi.
    """
    if chat_type == ChatType.PRIVATE:
        return None
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


#: `chats.last_message_text` ustunining kengligi (ui/R1a, migratsiya 0017).
PREVIEW_CHARS = 200


def _dialog_preview(dialog: Any) -> str | None:
    """Dialogdagi oxirgi xabarning qisqa ko'rinishi (UI ro'yxati uchun).

    Bu **kontent**, ya'ni ishonchsiz ma'lumot: faqat UI'da ko'rsatiladi,
    promptga `<untrusted_data>` konvertisiz hech qachon qo'shilmaydi.
    """
    msg = getattr(dialog, "message", None)
    text_ = strip_invisible(str(getattr(msg, "message", "") or "")).strip() if msg else ""
    return text_[:PREVIEW_CHARS] or None


def _photo_id(ent: Any) -> int | None:
    """Entity'dagi profil rasmi id'si (`ChatPhotoEmpty` → `None`)."""
    pid = getattr(getattr(ent, "photo", None), "photo_id", None)
    return int(pid) if pid else None


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
                # ── ui/R1a: chat profili, **qo'shimcha RPC'siz** ──
                # Uchalasi ham shu `GetDialogs` javobida bor. `avatar_photo_id`
                # faqat **o'zgarishni aniqlash** kaliti: fayl `services/avatars.py`
                # da alohida yuklanadi (xabar mediasi hech qachon emas).
                "unread_count": int(getattr(d, "unread_count", 0) or 0),
                "last_message_text": _dialog_preview(d),
                "avatar_photo_id": _photo_id(ent),
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
                # ── ui/R1a ──
                "unread_count": stmt.excluded.unread_count,
                "last_message_text": stmt.excluded.last_message_text,
                "avatar_photo_id": stmt.excluded.avatar_photo_id,
                # Rasm id'si o'zgargan bo'lsa `avatar_updated_at` NULL bo'ladi —
                # ya'ni qator `refresh_chat_profiles` navbatiga tushadi. O'zgarmasa
                # tegilmaydi: fayl qayta yuklanmaydi (RPC tejash).
                "avatar_updated_at": case(
                    (
                        Chat.avatar_photo_id.is_distinct_from(stmt.excluded.avatar_photo_id),
                        null(),
                    ),
                    else_=Chat.avatar_updated_at,
                ),
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
        # Shaxsiy chatda `from_id` bo'sh keladi — "bu xabarni biz yozdik"
        # ni faqat `m.out` aytadi (alertlar shu ustunga qaraydi).
        "out": bool(getattr(m, "out", False)),
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


SENDER_NAME_CHARS = 96


def sender_to_row(account_id: int, sender: Any) -> dict[str, Any] | None:
    """Telethon `User` → `senders` qatori. Foydalanuvchi bo'lmasa — `None`.

    Manba `m.sender` — `GetHistory` javobi bilan **birga** kelgan entity keshi,
    ya'ni qo'shimcha RPC yo'q va allowlist chegarasi joyida (`get_sender()`
    chaqirilmaydi, u allowlist'da yo'q metodlarga borishi mumkin).

    Kanal/guruh (anonim admin posti) — `None`: `senders` faqat odamlar uchun,
    kanal nomini `chats.title` allaqachon beradi.

    **Telefon raqami olinmaydi.** `User.phone` bor bo'lsa ham o'qilmaydi:
    qator qiymati LLM promptiga tushadi.

    Matn `strip_invisible` dan o'tadi (saqlash chegarasi, `message_to_row`
    bilan bir xil qoida); prompt chegarasidagi `neutralize` — `tools._line` da.
    """
    uid = getattr(sender, "id", None)
    if not isinstance(uid, int) or isinstance(uid, bool):
        return None
    if not hasattr(sender, "first_name"):  # Channel / Chat — odam emas
        return None
    parts = [str(getattr(sender, attr, "") or "").strip() for attr in ("first_name", "last_name")]
    display = strip_invisible(" ".join(x for x in parts if x)).strip()[:SENDER_NAME_CHARS]
    username = strip_invisible(str(getattr(sender, "username", "") or "").strip())[:64]
    if not display and not username:
        return None
    return {
        "account_id": int(account_id),
        "tg_user_id": int(uid),
        "display_name": display,
        "username": username or None,
        "is_bot": bool(getattr(sender, "bot", False)),
    }


async def upsert_senders(rows: list[dict[str, Any]]) -> int:
    """`senders` upsert. Xatosi sync'ni to'xtatmaydi — nom qulaylik kanali."""
    if not rows:
        return 0
    async with session_scope() as db:
        stmt = pg_insert(Sender).values(rows)
        await db.execute(
            stmt.on_conflict_do_update(
                index_elements=["account_id", "tg_user_id"],
                set_={
                    "display_name": stmt.excluded.display_name,
                    "username": stmt.excluded.username,
                    "is_bot": stmt.excluded.is_bot,
                    "updated_at": func.now(),
                },
            )
        )
    return len(rows)


def raw_peer_id(peer: Any) -> int | None:
    """Telethon `Peer*` / `InputPeer*` → **xom** (musbat) id.

    `chats.tg_peer_id` shu fazoda saqlanadi, shuning uchun real-time updates
    listener (`worker/updates.py`) peer'ni chat qatoriga shu orqali bog'laydi.
    """
    return _peer_id(peer)


async def upsert_message_from_update(chat_id: int, m: Any) -> int:
    """Bitta xabarni (update oqimidan kelgan) upsert qiladi.

    `sync_chat` bilan aynan bir xil yo'l — `message_to_row` + `upsert_messages`,
    ya'ni `edited_at` `m.edit_date` dan, matn esa `strip_invisible` dan o'tadi.
    Service/bo'sh xabar → 0.
    """
    row = message_to_row(chat_id, m)
    if row is None:
        return 0
    ents = getattr(m, "entities", None)
    return await upsert_messages([row], tg_entities={(chat_id, int(m.id)): ents} if ents else None)


async def upsert_messages(
    rows: list[dict[str, Any]],
    *,
    tg_entities: dict[tuple[int, int], Any] | None = None,
) -> int:
    """Xabarlarni upsert qiladi va (phase4/A) entity indeksini yangilaydi.

    `tg_entities` — `(chat_id, tg_msg_id)` → Telethon `message.entities`.
    Faqat `mention_name` uchun kerak: ism **erkin matndan taxmin
    qilinmaydi**, faqat Telegram bergan entity'dan olinadi.
    """
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
        res = await db.execute(stmt.returning(Message.id, Message.chat_id, Message.tg_msg_id))
        await _index_entities(db, res.all(), rows, tg_entities)
    return len(rows)


async def _index_entities(
    db: Any,
    written: Sequence[Any],
    rows: list[dict[str, Any]],
    tg_entities: dict[tuple[int, int], Any] | None,
) -> None:
    """Upsert qilingan xabarlar uchun `node_entities` (phase4/A).

    Xatosi **sync'ni to'xtatmaydi**: indeks — qulaylik kanali, ingest esa
    asosiy yo'l. Backfill ishi (`worker.tasks.backfill_entities`) o'tkazib
    yuborilganini keyin to'ldiradi.
    """
    if not written:
        return
    try:
        if not get_settings().entities_enabled:
            return
        from app.services.entities import EntityRef, index_messages_entities

        by_key = {(int(r["chat_id"]), int(r["tg_msg_id"])): r for r in rows}
        chat_ids = {int(w[1]) for w in written}
        accounts = dict(
            (await db.execute(select(Chat.id, Chat.account_id).where(Chat.id.in_(chat_ids)))).all()
        )
        refs: list[EntityRef] = []
        for mid, chat_id, tg_msg_id in written:
            row = by_key.get((int(chat_id), int(tg_msg_id)))
            account_id = accounts.get(int(chat_id))
            if row is None or account_id is None:
                continue
            refs.append(
                EntityRef(
                    message_id=int(mid),
                    account_id=int(account_id),
                    chat_id=int(chat_id),
                    text=row.get("text") or "",
                    seen_at=row.get("published_at"),
                    tg_entities=(tg_entities or {}).get((int(chat_id), int(tg_msg_id))),
                )
            )
        await index_messages_entities(db, refs)
    except Exception as exc:  # entity indeksi ingest'ni hech qachon yiqitmaydi
        log.warning("ingest.entities_failed", error=f"{type(exc).__name__}: {str(exc)[:160]}")


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
        title, synced_now, estimate_now = chat.title, chat.synced_total, chat.total_estimate
        chat.sync_state = SyncState.RUNNING
        chat.sync_error = None

    # UI ko'rsatkichi (Redis, TTL 10 daq) — `chats` jadvali haqiqat manbai
    # bo'lib qoladi, bu yerda faqat "hozir nima bo'lyapti" (services/sync_progress).
    report.tracker = SP.ChatTracker(account_id, chat_id, title, budget=budget)
    await report.tracker.publish(
        fetched=0, synced_total=synced_now, total_estimate=estimate_now, phase="new"
    )

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
        await report.tracker.publish_paused(int(exc.seconds))
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
    # `synced_total` uchun yuguruvchi hisoblagich: ilgari har 100 xabarda
    # butun chat bo'yicha `COUNT(*)` yugurardi (million qatorli chatda har
    # batch = to'liq skan). Bir marta o'qiymiz, keyin upsert natijasini
    # qo'shib boramiz (upsert — taxmin: mavjud qator qayta hisoblanadi, lekin
    # bu faqat progress ko'rsatkichi; `_finish` da aniqlanmaydi ham).
    # bir marta o'qiymiz, keyin har batchda upsert natijasini qo'shamiz.
    # Qayta sinxronlashda upsert mavjud qatorni ham sanaydi, ya'ni raqam
    # biroz oshib ketishi mumkin — u faqat progress ko'rsatkichi va keyingi
    # sync boshidagi COUNT uni to'g'irlaydi.
    counter = await _message_count(chat_id)
    account_id = await _account_of(chat_id)
    phase = "backfill" if backfill else "new"
    batch: list[dict[str, Any]] = []
    # phase4/A: `mention_name` faqat Telegram entity'laridan olinadi — matn
    # bilan birga batch'gacha olib boriladi (`messages` da saqlanmaydi).
    batch_entities: dict[tuple[int, int], Any] = {}
    # Jo'natuvchi nomlari: `m.sender` **shu javob bilan** kelgan entity keshidan
    # (qo'shimcha RPC yo'q). Batch ichida id bo'yicha yig'iladi — bir odam yuz
    # marta yozgan bo'lsa ham bitta qator.
    batch_senders: dict[int, dict[str, Any]] = {}
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
            ents = getattr(m, "entities", None)
            if ents:
                batch_entities[(chat_id, int(m.id))] = ents
            if account_id is not None:
                srow = sender_to_row(account_id, getattr(m, "sender", None))
                if srow is not None:
                    batch_senders[srow["tg_user_id"]] = srow
        report.newest_id = max(report.newest_id or 0, int(m.id))
        report.oldest_id = min(report.oldest_id or int(m.id), int(m.id))
        if len(batch) >= BATCH:
            written = await upsert_messages(batch, tg_entities=batch_entities)
            await _flush_senders(batch_senders)
            report.inserted_or_updated += written
            counter += written
            await _progress(
                chat_id,
                report,
                total_estimate,
                synced_total=counter,
                phase=phase,
                fetched=report.fetched + fetched,
            )
            batch = []
            batch_entities = {}
            batch_senders = {}
    if batch:
        written = await upsert_messages(batch, tg_entities=batch_entities)
        report.inserted_or_updated += written
        counter += written
    await _flush_senders(batch_senders)
    report.fetched += fetched
    if fetched:
        SYNC_MESSAGES.inc(fetched)
    # ── "tarix boshiga yetildi" faqat Telegram **kalta batch** qaytarganda ──
    # Ilgari shart `fetched < budget` edi va byudjet tugashini ham "tugadi"
    # deb o'qirdi: keyingi `_finish` `synced_min_id = 1` sentinelini qo'yib,
    # qolgan tarix abadiy yo'qolardi. Endi faqat oxirgi batch to'lmagan
    # bo'lsa (yoki umuman xabar kelmasa) `done` qo'yiladi — byudjet aynan
    # qolgan tarixga teng chiqqan holatda esa keyingi yugurish 0 xabar olib
    # keladi va sentinel o'sha yerda qo'yiladi (bir ortiqcha so'rov — arzon).
    chunk = min(BATCH, budget) or BATCH
    if backfill and (fetched == 0 or (fetched < budget and fetched % chunk != 0)):
        report.done = True
    await _progress(
        chat_id, report, total_estimate, synced_total=counter, phase=phase, fetched=report.fetched
    )
    return fetched


async def _account_of(chat_id: int) -> int | None:
    """Chat egasi (`chats.account_id`) — `senders` qatori uchun, bir marta.

    Topilmasa (yoki so'rov yiqilsa) `None`: jo'natuvchi nomlari **qulaylik**
    kanali, ingest esa asosiy yo'l — nom uchun sync to'xtamaydi.
    """
    try:
        async with session_scope() as db:
            row = (
                await db.execute(select(Chat.account_id).where(Chat.id == chat_id))
            ).scalar_one_or_none()
        return int(row) if row is not None else None
    except Exception as exc:
        log.warning("ingest.sender_account_lookup_failed", chat_id=chat_id, error=str(exc)[:200])
        return None


async def _flush_senders(rows: dict[int, dict[str, Any]]) -> None:
    """Nomlar sync'ni yiqitmaydi: xato — log, ingest davom etadi."""
    if not rows:
        return
    try:
        await upsert_senders(list(rows.values()))
    except Exception as exc:  # nom qulaylik kanali, asosiy yo'l emas
        log.warning("ingest.senders_failed", error=str(exc)[:200], n=len(rows))


# ─── jo'natuvchi nomlarini keyin to'ldirish (backfill) ───────────────────────
# `senders` jadvali migratsiya 0019 bilan paydo bo'ldi, ya'ni undan **oldin**
# sinxronlangan xabarlarning muallifi nomsiz qoldi va javoblarda
# "Foydalanuvchi 1164475416" ko'rinardi (jonli o'lchov, `people-04`).
# Bu yo'l qayta sync qilmaydi: har chatdan oxirgi `per_chat` xabarni o'qib,
# `m.sender` keshidan nomlarni yig'adi (qo'shimcha RPC yo'q, `history` bucket).

SENDER_BACKFILL_PER_CHAT = 200
SENDER_BACKFILL_CANDIDATES = 60  # nechta chat tekshiriladi (DB, arzon)


async def _has_missing_senders(db: Any, account_id: int, chat_id: int, per_chat: int) -> bool:
    """Chatning oxirgi `per_chat` xabari ichida `senders` da yo'q muallif bormi?"""
    recent = (
        select(Message.sender_id.label("sender_id"))
        .where(Message.chat_id == chat_id, Message.sender_id.isnot(None))
        .order_by(Message.tg_msg_id.desc())
        .limit(per_chat)
        .subquery()
    )
    stmt = (
        select(recent.c.sender_id)
        .where(
            ~exists(
                select(Sender.tg_user_id).where(
                    Sender.account_id == account_id, Sender.tg_user_id == recent.c.sender_id
                )
            )
        )
        .limit(1)
    )
    return (await db.execute(stmt)).first() is not None


async def chats_missing_senders(
    account_id: int, *, limit: int = 30, per_chat: int = SENDER_BACKFILL_PER_CHAT
) -> list[int]:
    """Nomlar yetishmayotgan chatlar (eng yangi faollikdan boshlab, ≤ `limit`)."""
    out: list[int] = []
    async with session_scope() as db:
        found = await db.execute(
            select(Chat.id)
            .where(Chat.account_id == account_id, Chat.synced_total > 0)
            .order_by(Chat.last_message_at.desc().nulls_last(), Chat.id)
            .limit(SENDER_BACKFILL_CANDIDATES)
        )
        rows = found.scalars().all()
        for chat_id in rows:
            if await _has_missing_senders(db, account_id, int(chat_id), per_chat):
                out.append(int(chat_id))
                if len(out) >= limit:
                    break
    return out


async def harvest_senders(chat_id: int, *, per_chat: int = SENDER_BACKFILL_PER_CHAT) -> int:
    """Bitta chatdan jo'natuvchi nomlarini yig'adi. Qaytaradi: yozilgan qatorlar.

    Faqat o'qish: `iter_messages` (GetHistory) va `m.sender` keshi —
    `get_sender()` **chaqirilmaydi** (u allowlist'da yo'q metodlarga borishi
    mumkin). Shaxsiy chatda umuman RPC yo'q: entity dialog keshidan olinadi.
    FloodWait — `SyncPaused`, retry yo'q (4.2-band).
    """
    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None:
            return 0
        account_id, peer_id, ctype = int(chat.account_id), int(chat.tg_peer_id), str(chat.type)

    rows: dict[int, dict[str, Any]] = {}
    if ctype == ChatType.PRIVATE:
        srow = sender_to_row(account_id, await pool.entity(account_id, peer_id))
        if srow is not None:
            rows[srow["tg_user_id"]] = srow
    else:
        try:
            client = await pool.get(account_id)
            peer = await _input_peer(account_id, peer_id)
            async for m in client.iter_messages(
                peer, limit=max(1, int(per_chat)), wait_time=WAIT_BETWEEN_BATCHES
            ):
                srow = sender_to_row(account_id, getattr(m, "sender", None))
                if srow is not None:
                    rows[srow["tg_user_id"]] = srow
        except tg_errors.FloodWaitError as exc:
            FLOODWAIT.inc()
            raise SyncPaused(int(exc.seconds)) from exc
    if not rows:
        return 0
    written = await upsert_senders(list(rows.values()))
    log.info("ingest.senders_backfilled", chat_id=chat_id, senders=written)
    return written


async def _message_count(chat_id: int) -> int:
    async with session_scope() as db:
        return int(
            (
                await db.execute(select(func.count(Message.id)).where(Message.chat_id == chat_id))
            ).scalar_one()
            or 0
        )


async def _progress(
    chat_id: int,
    report: SyncReport,
    total_estimate: int | None,
    *,
    synced_total: int | None = None,
    phase: str | None = None,
    fetched: int | None = None,
) -> None:
    """Oraliq progress. `synced_total` — yuguruvchi hisoblagich (`_pull`); `None`
    bo'lsa aniq `COUNT(*)` olinadi (endi faqat aniqlik kerak bo'lgan joyda).

    Shu yerda **DB yozuvi bilan bir vaqtda** Redis'ga UI ko'rsatkichi ham
    chiqadi (`report.tracker`) — ikkinchisi yiqilsa birinchisi zarar ko'rmaydi.
    """
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
            int(synced_total)
            if synced_total is not None
            else (
                await db.execute(select(func.count(Message.id)).where(Message.chat_id == chat_id))
            ).scalar_one()
        )
        live = (int(chat.synced_total), chat.total_estimate)
    if report.tracker is not None:
        await report.tracker.publish(
            fetched=fetched if fetched is not None else report.fetched,
            synced_total=live[0],
            total_estimate=live[1],
            phase=phase,
        )


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


def snapshot_candidates_stmt(
    chat_id: int, now: datetime, lo: timedelta, hi: timedelta, limit: int
) -> Any:
    """Bitta yosh pog'onasi uchun nomzod postlar — **eng eski snapshot birinchi**.

    `LIMIT` majburiy (`SNAPSHOT_MAX_MESSAGES_PER_RUN`): usiz bitta katta kanal
    butun pog'onani bir yugurishda tortib, qolgan kanallarni ochlikda
    qoldirardi. Tartib `max(captured_at)` bo'yicha (hech olinmagan — birinchi),
    shuning uchun qopqoq hech bir postni abadiy chetda qoldirmaydi: keyingi
    yugurishda o'shalar oldinga chiqadi.
    """
    last_snap = (
        select(
            MessageMetricSnapshot.message_id.label("mid"),
            func.max(MessageMetricSnapshot.captured_at).label("last_at"),
        )
        .group_by(MessageMetricSnapshot.message_id)
        .subquery()
    )
    return (
        select(Message.id, Message.tg_msg_id)
        .outerjoin(last_snap, last_snap.c.mid == Message.id)
        .where(
            Message.chat_id == chat_id,
            Message.published_at <= now - lo,
            Message.published_at > now - hi,
        )
        .order_by(last_snap.c.last_at.asc().nulls_first(), Message.id)
        .limit(limit)
    )


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
        # Bitta yugurishda olinadigan postlar qopqog'i: usiz bitta katta kanal
        # butun pog'onasini (o'n minglab post) bir yugurishda tortib, qolgan
        # kanallarni ochlikda qoldiradi. Tanlov **eng eski snapshot** dan
        # boshlanadi (hech qachon olinmagan — birinchi), shuning uchun qopqoq
        # hech bir postni abadiy chetda qoldirmaydi: navbat aylanadi.
        budget = get_settings().snapshot_max_messages_per_run
        for lo, hi in tiers:
            if budget <= 0:
                break
            rows = await db.execute(snapshot_candidates_stmt(chat_id, now, lo, hi, budget))
            for mid, tg_id in rows.all():
                ids.append(int(tg_id))
                id_map[int(tg_id)] = int(mid)
                budget -= 1
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
