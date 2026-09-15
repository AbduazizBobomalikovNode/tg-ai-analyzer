"""Chat profil rasmi va profil matni (ui/R1a) — `DATA_DIR/avatars` + `chats` ustunlari.

**"Media saqlanmaydi" invariantiga chegaralangan istisno** (migratsiya 0017,
CLAUDE.md "Kontent"): faqat **profil rasmi** yuklab olinadi va saqlanadi —
kichik fayl, chat boshiga bittasi. Xabar mediasi (photo/video/document/voice)
hech qachon yuklab olinmaydi: `upload.GetFileRequest` allowlist'da READ, lekin
uni chaqiradigan **yagona modul shu fayl**, va buni
`tests/test_avatars.py::test_no_other_module_downloads_files` grep bilan
qo'riqlaydi (`download_media` / `download_file` / `GetFileRequest`).

Narx qopqoqlari:
  * har chaqiruv `history` bucket'idan token oladi (`tools.bucket_guard`) —
    `search_*` va profil tool'lari bilan **bitta** akkaunt kvotasi;
  * `FloodWaitError` **retry qilinmaydi** — matn bo'lib `chats.avatar_error`
    ga yoziladi, keyingi cron yugurishi qayta uradi (4.2-band);
  * `avatar_photo_id` o'zgarmagan bo'lsa fayl **qayta yuklanmaydi**;
  * fayl 512 KB dan katta bo'lsa rad etiladi (profil rasmi bunchalik bo'lmaydi).

Rasm faqat egasiga beriladi: `web/routers/tg.py` dagi
`GET /api/accounts/{id}/chats/{chat_id}/avatar` `get_owned_chat()` bilan
tekshiradi (naqsh `services/images.py` dan).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import Account, Chat
from app.logging import get_logger
from app.services.textnorm import strip_invisible

log = get_logger(__name__)

MAX_AVATAR_BYTES = 512 * 1024
MAX_ABOUT_CHARS = 4000
_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"
_EXT = {"image/jpeg": "jpg", "image/png": "png"}
#: Profil rasmi ham serverga boradi — `search_*` bilan bir xil akkaunt kvotasi.
BUCKET = "history"


def avatars_dir() -> Path:
    d = Path(get_settings().data_dir) / "avatars"
    d.mkdir(parents=True, exist_ok=True)
    return d


def path_for(chat_id: int, mime: str | None = "image/jpeg") -> Path:
    ext = _EXT.get(str(mime or ""), "jpg")
    return avatars_dir() / f"{int(chat_id)}.{ext}"


def _sniff(data: bytes) -> str | None:
    if data.startswith(_JPEG):
        return "image/jpeg"
    if data.startswith(_PNG):
        return "image/png"
    return None


def photo_id_of(entity: Any) -> int | None:
    """Entity'dagi profil rasmi id'si (`ChatPhotoEmpty` / rasmsiz → `None`)."""
    photo = getattr(entity, "photo", None)
    pid = getattr(photo, "photo_id", None)
    return int(pid) if pid else None


def _remove_files(chat_id: int) -> None:
    for ext in set(_EXT.values()):
        try:
            (avatars_dir() / f"{int(chat_id)}.{ext}").unlink(missing_ok=True)
        except OSError as exc:  # fayl tizimi — avatar yo'qligi xato emas
            log.warning("avatar.unlink_failed", chat_id=chat_id, error=str(exc)[:200])


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


async def get_owned_chat(session: AsyncSession, user_id: int, chat_id: int) -> Chat | None:
    """Chat → egasi (`chats.account_id` → `accounts.user_id`) mos kelsa qator."""
    row = (
        await session.execute(
            select(Chat)
            .join(Account, Account.id == Chat.account_id)
            .where(Chat.id == int(chat_id), Account.user_id == int(user_id))
            .limit(1)
        )
    ).scalar_one_or_none()
    return row


def _guard(account_id: int) -> str | None:
    """`history` bucket'idan token. Bo'sh bo'lsa — xato matni (kutish yo'q)."""
    from app.services.tools import bucket_guard

    return bucket_guard(account_id, BUCKET)


async def _load(account_id: int, chat_id: int) -> Chat | None:
    async with session_scope() as db:
        row = await db.get(Chat, int(chat_id))
        if row is None or row.account_id != int(account_id):
            return None
        db.expunge(row)
        return row


async def _apply(chat_id: int, values: dict[str, Any]) -> None:
    async with session_scope() as db:
        row = await db.get(Chat, int(chat_id))
        if row is None:
            return
        for key, val in values.items():
            setattr(row, key, val)


# ─── profil rasmi ────────────────────────────────────────────────────────────


async def refresh_avatar(account_id: int, chat_id: int) -> bool:
    """Profil rasmini yangilaydi. `True` — yangi fayl yozildi.

    Yo'l: dialog keshidagi entity (`pool.entity`, qo'shimcha RPC yo'q) →
    `photo_id` → o'zgargan bo'lsa `download_profile_photo(..., file=bytes)`.
    Entity keshda bo'lmasa `chats` qatoridan `InputPeer` quriladi
    (`pool.input_peer_from_row`) — `get_entity()` hech qachon chaqirilmaydi.
    """
    from telethon import errors as tg_errors

    from app.mtproto.pool import PoolError, input_peer_from_row, pool

    if not get_settings().avatars_enabled:
        return False
    row = await _load(account_id, chat_id)
    if row is None:
        return False

    try:
        entity = await pool.entity(account_id, row.tg_peer_id)
    except PoolError as exc:
        await _apply(chat_id, {"avatar_error": f"pool:{exc.code}"[:200]})
        return False

    photo_id = photo_id_of(entity) if entity is not None else row.avatar_photo_id
    if photo_id is None:
        # Rasmi yo'q (yoki olib tashlangan) — maydonlar tozalanadi, fayl o'chadi.
        _remove_files(chat_id)
        await _apply(
            chat_id,
            {
                "avatar_photo_id": None,
                "avatar_mime": None,
                "avatar_error": None,
                "avatar_updated_at": datetime.now(UTC),
            },
        )
        return False

    fresh = (
        photo_id == row.avatar_photo_id
        and row.avatar_updated_at is not None
        and row.avatar_mime
        and path_for(chat_id, row.avatar_mime).exists()
    )
    if fresh:
        return False

    if (err := _guard(account_id)) is not None:
        await _apply(chat_id, {"avatar_error": err[:200]})
        return False

    target = entity if entity is not None else input_peer_from_row(row)
    if target is None:
        await _apply(chat_id, {"avatar_error": "no_peer"})
        return False

    try:
        client = await pool.get(account_id)
        data = await client.download_profile_photo(target, file=bytes)
    except tg_errors.FloodWaitError as exc:
        # Retry yo'q — keyingi cron yugurishi qayta uradi (4.2-band).
        await _apply(chat_id, {"avatar_error": f"flood_wait {exc.seconds}s"[:200]})
        return False
    except PoolError as exc:
        await _apply(chat_id, {"avatar_error": f"pool:{exc.code}"[:200]})
        return False
    except Exception as exc:  # RpcBlocked, ChannelPrivate, tarmoq — holat, xato emas
        await _apply(chat_id, {"avatar_error": f"{type(exc).__name__}: {exc}"[:200]})
        return False

    if not data:
        _remove_files(chat_id)
        await _apply(
            chat_id,
            {
                "avatar_photo_id": None,
                "avatar_mime": None,
                "avatar_error": None,
                "avatar_updated_at": datetime.now(UTC),
            },
        )
        return False
    if len(data) > MAX_AVATAR_BYTES:
        await _apply(chat_id, {"avatar_error": f"too_large {len(data)}b"})
        return False
    mime = _sniff(bytes(data))
    if mime is None:
        await _apply(chat_id, {"avatar_error": "unknown_format"})
        return False

    _remove_files(chat_id)
    _write_atomic(path_for(chat_id, mime), bytes(data))
    await _apply(
        chat_id,
        {
            "avatar_photo_id": int(photo_id),
            "avatar_mime": mime,
            "avatar_error": None,
            "avatar_updated_at": datetime.now(UTC),
        },
    )
    log.info("avatar.saved", chat_id=chat_id, bytes=len(data), mime=mime)
    return True


# ─── profil matni (`GetFull*`) ───────────────────────────────────────────────


async def full_info(client: Any, peer: Any) -> tuple[Any, Any]:
    """Peer turiga mos `GetFull*` → (full, entity).

    **DB'dagi `chats.type` ga ishonilmaydi** — haqiqiy `InputPeer` turi
    tekshiriladi (CLAUDE.md "Profil tool'lari va peer turi"):
    `InputPeerChannel` → `channels.GetFullChannel` (kanal **va** supergruh),
    `InputPeerChat` → `messages.GetFullChat`, `InputPeerUser` →
    `users.GetFullUser`. Noto'g'risi `TypeError` beradi.

    Dispatch `services/tools.get_chat_info` dagi bilan bir xil qoida;
    u yerda natija modelga matn bo'lib ketadi, bu yerda DB'ga yoziladi.
    """
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    if isinstance(peer, tlt.InputPeerChannel):
        res = await client(tlf.channels.GetFullChannelRequest(channel=peer))
        return res.full_chat, _pick(getattr(res, "chats", None), peer.channel_id)
    if isinstance(peer, tlt.InputPeerChat):
        res = await client(tlf.messages.GetFullChatRequest(chat_id=peer.chat_id))
        return res.full_chat, _pick(getattr(res, "chats", None), peer.chat_id)
    if isinstance(peer, tlt.InputPeerUser):
        res = await client(tlf.users.GetFullUserRequest(id=peer))
        return res.full_user, _pick(getattr(res, "users", None), peer.user_id)
    raise TypeError(f"unsupported peer: {type(peer).__name__}")


def _pick(items: Any, ident: int) -> Any:
    for item in items or ():
        if int(getattr(item, "id", 0) or 0) == int(ident):
            return item
    return None


def _participants(full: Any) -> int | None:
    n = getattr(full, "participants_count", None)
    if n is None:  # oddiy guruhda ro'yxat ichida keladi
        parts = getattr(getattr(full, "participants", None), "participants", None)
        n = len(parts) if parts else None
    return int(n) if n else None


async def refresh_about(account_id: int, chat_id: int) -> bool:
    """`about` + verified/scam/fake + a'zolar sonini `GetFull*` dan yangilaydi."""
    from telethon import errors as tg_errors

    from app.mtproto.pool import PoolError, input_peer_from_row, pool

    row = await _load(account_id, chat_id)
    if row is None:
        return False
    if (err := _guard(account_id)) is not None:
        await _apply(chat_id, {"avatar_error": err[:200]})
        return False

    peer = input_peer_from_row(row)
    try:
        if peer is None:
            peer = await pool.input_peer(account_id, row.tg_peer_id)
        client = await pool.get(account_id)
        full, entity = await full_info(client, peer)
    except tg_errors.FloodWaitError as exc:
        await _apply(chat_id, {"avatar_error": f"flood_wait {exc.seconds}s"[:200]})
        return False
    except PoolError as exc:
        await _apply(chat_id, {"avatar_error": f"pool:{exc.code}"[:200]})
        return False
    except Exception as exc:  # cheklov = holat (private chat, forbidden, …)
        await _apply(chat_id, {"avatar_error": f"{type(exc).__name__}: {exc}"[:200]})
        return False

    about = strip_invisible(str(getattr(full, "about", "") or "")).strip()[:MAX_ABOUT_CHARS]
    values: dict[str, Any] = {
        "about": about,
        "is_verified": bool(getattr(entity, "verified", False)),
        "is_scam": bool(getattr(entity, "scam", False)),
        "is_fake": bool(getattr(entity, "fake", False)),
        "avatar_error": None,
    }
    count = _participants(full)
    if count is not None:
        values["participants_count"] = count
    await _apply(chat_id, values)
    return True
