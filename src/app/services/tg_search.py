"""Telegram **server** qidiruvining ichki API'si (3-bosqich, arxitektura v2, 8-bo'lim).

`services/tools.py` dagi qidiruv tool'lari model uchun matn (`ToolResult` +
`<untrusted_data>` konverti) qaytaradi. Deep search quvuriga esa matn emas,
**tipli natija** kerak: hitlar ro'yxati, qaysi bucket to'lgani, nechta RPC
ketgani. Shu sababli xom MTProto mantiqi shu modulga ko'chirildi:

    tools.search_telegram  ─┐
    tools.search_global    ─┼─► tg_search.server_*  ─► MTProto (allowlist READ)
    deep_search.escalate   ─┘

Tool'lar endi shu funksiyalar ustidagi **yupqa o'ramchi**: ular faqat
natijani satrlarga aylantiradi. Ikki yo'l bitta koddan o'tadi — server
qidiruvining xatti-harakati bir joyda.

Invariantlar (CLAUDE.md):

* **Allowlist tegilmaydi** — `messages.Search`, `messages.SearchGlobal`,
  `messages.GetSearchResultsCalendar`, `channels.SearchPosts`,
  `channels.CheckSearchPostsFlood` allaqachon READ.
* **`FloodWaitError` retry qilinmaydi** — `error_text` bo'lib qaytadi,
  chaqiruvchi lokal qidiruvga tushadi (retry-storm yo'q).
* **Stars hech qachon sarflanmaydi** — `server_discover` `allow_paid_stars`
  ni yubormaydi; avval `CheckSearchPostsFlood`, bepul slot bo'lmasa faqat
  hashtag rejimi yoki tushuntiruvchi matn.
* **Premium yo'q bo'lsa kashfiyot to'xtamaydi** — `SearchPosts` Premium
  talab qilsa (`note_code="premium_required"`) tekin yo'l ishlaydi:
  `contacts.Search` (nom) + topilgan ochiq kanallarda `messages.Search`
  (`server_discover_by_name`). Yangi TL metod qo'shilmagan.
* **Telefon raqami qaytarilmaydi** — `ServerHit` da bunday maydon yo'q va
  entity'dan faqat nom/username o'qiladi.
* **`get_sender()` / `get_chat()` chaqirilmaydi** — muallif va chat nomi
  javob bilan birga kelgan `users`/`chats` ro'yxatidan olinadi.

Shift (`MethodBuckets`) `tools.bucket_guard` orqali — hisob **bitta**
`tools._buckets` nusxasida turadi, ya'ni tool yo'li ham, deep search yo'li ham
bitta akkaunt kvotasini bo'lishadi (ikkalasi ham o'sha bitta MTProto
sessiyasini yeydi).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chat, Message
from app.logging import get_logger
from app.services import search as S
from app.services.stopwords import is_stopword
from app.services.textnorm import uz_fold

log = get_logger(__name__)

__all__ = [
    "BUCKET_CONTACTS",
    "BUCKET_DISCOVER",
    "BUCKET_HISTORY",
    "BUCKET_RESOLVE",
    "BUCKET_SEARCH",
    "GLOBAL_SCOPES",
    "SERVER_LIMIT_MAX",
    "ChannelHits",
    "ServerHit",
    "ServerHits",
    "discover_name_queries",
    "persist_server_hits",
    "send_request",
    "server_calendar",
    "server_channel_search",
    "server_discover",
    "server_discover_by_name",
    "server_search_chat",
    "server_search_global",
]

# ─── shift sinflari ──────────────────────────────────────────────────────────

BUCKET_HISTORY = "history"
BUCKET_SEARCH = "search"
BUCKET_RESOLVE = "resolve"
BUCKET_DISCOVER = "discover"
#: `contacts.Search` uchun **alohida** bucket: u `channels.SearchPosts` emas,
#: arzonroq nom qidiruvi. Premium probe'i (`CheckSearchPostsFlood`) bilan bitta
#: `burst=1` bucket'ni bo'lishganda probe yagona tokenni yer va Premium'siz
#: akkauntda nom bo'yicha zaxira yo'l **hech qachon** ishga tushmasdi.
BUCKET_CONTACTS = "contacts"

SERVER_LIMIT_MAX = 20  # server qidiruvi DB'dan qimmatroq — qattiqroq qopqoq
SNIPPET_CHARS = 400
POST_SNIPPET_CHARS = 200

GLOBAL_SCOPES: dict[str, dict[str, Any]] = {
    "all": {},
    "channels": {"broadcasts_only": True},
    "groups": {"groups_only": True},
    "users": {"users_only": True},
}

# `filter` argumenti → `InputMessagesFilter*` klass nomi
FILTERS: dict[str, str] = {
    "url": "InputMessagesFilterUrl",
    "pinned": "InputMessagesFilterPinned",
    "mentions": "InputMessagesFilterMyMentions",
    "poll": "InputMessagesFilterPoll",
    "photo": "InputMessagesFilterPhotos",
    "document": "InputMessagesFilterDocument",
    "voice": "InputMessagesFilterVoice",
    "video": "InputMessagesFilterVideo",
}


def msg_filter(name: str, tlt: Any) -> Any | None:
    cls_name = FILTERS.get(name.strip().lower())
    cls = getattr(tlt, cls_name, None) if cls_name else None
    return cls() if cls is not None else None


# ─── natija turlari ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class ServerHit:
    """Telegram serveridan kelgan bitta topilma — konvertsiz, tipli.

    `chat_id` — lokal `chats.id` (agar akkaunt shu chatni sinxronlagan bo'lsa),
    aks holda `None`: bunday hit **saqlanmaydi** va dalilda `external` deb
    belgilanadi (lokal indeks bilan solishtirilmagan).
    """

    tg_msg_id: int
    tg_peer_id: int | None = None
    date: datetime | None = None
    sender: str = ""
    text: str = ""
    chat_title: str = ""
    chat_id: int | None = None  # lokal `chats.id`
    link: str = ""
    message_id: int | None = None  # lokal `messages.id` (saqlangandan keyin)
    raw: Any = None  # xom TL xabari — faqat `persist_server_hits` uchun
    # `Peer*` sinfidan olingan tur: `user` | `chat` | `channel` | None (M6).
    # Telegram'da raqamli id **tur bo'yicha** unikal: `PeerUser(777)` va
    # `PeerChannel(777)` butunlay boshqa ob'ektlar. Turni tashimasak,
    # `persist_server_hits` xabarni boshqa chatga yozib qo'yishi mumkin.
    peer_kind: str | None = None
    # "Saved Messages" (peer=self) — lokal `chats` da o'z qatori yo'q (M7).
    saved: bool = False


@dataclass(slots=True)
class ServerHits:
    """Bitta rung natijasi. Xato — **istisno emas**, matn (chaqiruvchi hal qiladi)."""

    hits: list[ServerHit] = field(default_factory=list)
    rate_limited: str | None = None  # to'lgan bucket nomi (yoki o'chirilgan sinf)
    error_text: str | None = None
    calls: int = 0
    note: str | None = None  # xato emas, holat (masalan Stars kvotasi tugagani)
    mode: str = ""  # discover: "hashtag" | "free text"
    # `note` sababi mashina o'qiy oladigan shaklda: hozircha faqat
    # `"post_search_flood"` (bepul slot tugagan) — qamrov hisoboti shu
    # bo'yicha halol sabab yozadi, matnni qayta tahlil qilmasdan.
    #: `"post_search_flood"` (bepul slot tugagan) yoki `"premium_required"`
    #: (`channels.SearchPosts` Telegram Premium talab qildi) — ikkalasi ham
    #: xato emas, **holat**: chaqiruvchi nom bo'yicha zaxira yo'lga tushadi.
    note_code: str = ""
    retry_at: str = ""  # `note_code` bilan birga: "YYYY-MM-DD HH:MM UTC" yoki ""
    #: Nom bo'yicha kashfiyot (`server_discover_by_name`): topilgan ochiq
    #: kanallar (`contacts.Search`) va ulardan **haqiqatan** o'qilganlari.
    channels: list[dict[str, Any]] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)
    #: `contacts.Search` ga haqiqatan yuborilgan nom so'rovlari (≤2) —
    #: qamrov/`server_notes` "nima izlandi" ni rost aytsin.
    name_queries: list[str] = field(default_factory=list)


# ─── entity yordamchilari (qo'shimcha RPC yo'q) ──────────────────────────────


def entity_name(obj: Any) -> str:
    """Telethon entity nomi — qo'shimcha so'rovsiz (`m.sender` / `m.chat` keshi)."""
    if obj is None:
        return ""
    title = getattr(obj, "title", None)
    if title:
        return str(title)
    parts = [getattr(obj, "first_name", None), getattr(obj, "last_name", None)]
    name = " ".join(p for p in parts if p)
    return name or (getattr(obj, "username", None) or "")


def peer_id_of(peer: Any) -> int | None:
    """`PeerUser/PeerChat/PeerChannel` (yoki InputPeer*) → raqamli id."""
    for attr in ("user_id", "chat_id", "channel_id"):
        val = getattr(peer, attr, None)
        if val:
            return int(val)
    return None


#: `Peer*` atributi → peer turi. `chats.type` bilan moslashtirish uchun (M6).
PEER_KINDS: tuple[tuple[str, str], ...] = (
    ("user_id", "user"),
    ("chat_id", "chat"),
    ("channel_id", "channel"),
)

#: Peer turi ↔ `chats.type` mosligi. Supergroup Telegram'da ham kanal
#: (`PeerChannel`), shuning uchun `channel` ikkalasini ham qabul qiladi.
PEER_KIND_CHAT_TYPES: dict[str, frozenset[str]] = {
    "user": frozenset({"private"}),
    "chat": frozenset({"group"}),
    "channel": frozenset({"channel", "supergroup"}),
}


def peer_kind_of(peer: Any) -> str | None:
    """`PeerUser/PeerChat/PeerChannel` → `user` / `chat` / `channel`."""
    for attr, kind in PEER_KINDS:
        if getattr(peer, attr, None):
            return kind
    return None


def peer_kind_matches(kind: str | None, chat_type: Any) -> bool:
    """Peer turi lokal `chats.type` bilan mos keladimi? Tur noma'lum — mos emas."""
    if not kind:
        return False
    allowed = PEER_KIND_CHAT_TYPES.get(kind) or frozenset()
    return str(chat_type) in allowed


def entity_index(res: Any) -> dict[int, Any]:
    """Javob bilan **birga kelgan** users/chats — qo'shimcha RPC yo'q.

    `get_sender()` / `get_chat()` chaqirilmaydi: ular allowlist'da yo'q
    metodlarga borishi mumkin (CLAUDE.md).
    """
    index: dict[int, Any] = {}
    for ent in list(getattr(res, "users", None) or []) + list(getattr(res, "chats", None) or []):
        ident = getattr(ent, "id", None)
        if ident:
            index[int(ident)] = ent
    return index


def who_of(m: Any, index: dict[int, Any]) -> str:
    """Muallif nomi; entity kelmagan bo'lsa `user:<id>` (— emas)."""
    pid = peer_id_of(getattr(m, "from_id", None))
    if pid is None:  # kanal nomidan yozilgan post
        pid = peer_id_of(getattr(m, "peer_id", None))
    if pid is None:
        return "—"
    return entity_name(index.get(pid)) or f"user:{pid}"


def chat_of(m: Any, index: dict[int, Any]) -> tuple[str, Any, int | None]:
    pid = peer_id_of(getattr(m, "peer_id", None))
    ent = index.get(pid) if pid else None
    return (entity_name(ent) or (f"chat:{pid}" if pid else "—")), ent, pid


def tg_link(entity: Any, chat_id: int | None, msg_id: Any) -> str:
    uname = getattr(entity, "username", None)
    if uname:
        return f"https://t.me/{uname}/{msg_id}"
    return f"https://t.me/c/{chat_id}/{msg_id}" if chat_id else ""


def _body_of(m: Any, snippet: int) -> str:
    body = S.compact_text(getattr(m, "message", "") or "", max_chars=snippet)
    if body:
        return body
    media = getattr(m, "media", None)
    return f"[{type(media).__name__.removeprefix('MessageMedia').lower()}]" if media else ""


def hit_of(
    m: Any, index: dict[int, Any], *, chat_title: str = "", snippet: int = SNIPPET_CHARS
) -> ServerHit:
    """Xom TL xabari → `ServerHit`. Nom/chat faqat javobdagi entity'lardan."""
    title, ent, pid = chat_of(m, index)
    return ServerHit(
        tg_msg_id=int(getattr(m, "id", 0) or 0),
        tg_peer_id=pid,
        peer_kind=peer_kind_of(getattr(m, "peer_id", None)),
        date=getattr(m, "date", None),
        sender=who_of(m, index),
        text=_body_of(m, snippet),
        chat_title=chat_title or title,
        link=tg_link(ent, pid, getattr(m, "id", 0)),
        raw=m,
    )


# ─── shift va RPC ────────────────────────────────────────────────────────────


def _guard(account_id: int, bucket: str) -> str | None:
    """Shift — `tools._buckets` nusxasi orqali (hisob bitta joyda turadi)."""
    from app.services import tools as T

    return T.bucket_guard(account_id, bucket)


async def send_request(
    account_id: int, request: Any, *, explain: dict[str, str] | None = None
) -> tuple[Any, str | None]:
    """TL so'rovni yuboradi. Xato — **matn**, istisno emas.

    `FloodWaitError` retry qilinmaydi (CLAUDE.md 4.2): matn qaytadi va
    chaqiruvchi lokal qidiruvga tushadi.
    """
    from telethon import errors as tg_errors

    from app.mtproto.pool import PoolError, pool

    try:
        client = await pool.get(account_id)
        return await client(request), None
    except tg_errors.FloodWaitError as exc:
        log.warning("tg_search.flood_wait", account_id=account_id, seconds=exc.seconds)
        return None, f"error: flood_wait {exc.seconds}s — try search_messages (local)"
    except PoolError as exc:
        return None, f"error: {exc.code}"
    except Exception as exc:  # RPC xatosi — matn, istisno emas
        hint = (explain or {}).get(type(exc).__name__)
        if hint:
            return None, f"error: {hint}"
        # L2: xom istisno matni modelga/klientga berilmaydi — u xabar matni,
        # peer id yoki ichki yo'lni olib chiqishi mumkin. Log'da to'liq turadi.
        log.warning(
            "tg_search.rpc_error",
            account_id=account_id,
            error_class=type(exc).__name__,
            error=str(exc)[:200],
        )
        return None, f"error: {type(exc).__name__}"


async def _peer_for_chat(account_id: int, chat_id: int) -> tuple[Any, str | None]:
    """Lokal `chats.id` → `InputPeer`. Tool'lar peer'ni o'zi beradi (`peer=`)."""
    from app.db.base import session_scope
    from app.mtproto.pool import PoolError, pool

    async with session_scope() as db:
        chat = await db.get(Chat, chat_id)
        if chat is None or chat.account_id != account_id:
            return None, f"error: chat {chat_id} not found for this account"
        peer_id = int(chat.tg_peer_id)
    try:
        return await pool.input_peer(account_id, peer_id), None
    except PoolError as exc:
        return None, f"error: {exc.code}"


def _rate_limited(bucket: str, err: str) -> ServerHits:
    return ServerHits(rate_limited=bucket, error_text=err)


# ─── 1. Bitta chat ichida (messages.Search) ──────────────────────────────────


async def server_search_chat(
    account_id: int,
    chat_id: int | None,
    query: str,
    *,
    saved: bool = False,
    from_user: str | None = None,
    from_peer: Any = None,
    topic_id: int | None = None,
    filter: str | None = None,  # TL parametri nomi bilan bir xil
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 20,
    peer: Any = None,
    chat_title: str = "",
) -> ServerHits:
    """`messages.Search` — bitta chat, sinxronlanmagan tarixni ham ko'radi.

    `from_peer` — tayyor `InputPeer` (tool uni o'zi, kesh-birinchi yo'l bilan
    yechadi); `from_user` — `@username` satri, u `contacts.ResolveUsername`
    orqali yechiladi (**resolve** bucket'i, alohida kvota).
    """
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    # `q=""` Telegram uchun xato emas — `from_id` yoki `filter` bilan birga u
    # "shu muallifning / shu turdagi hamma xabari" degani. Yolg'iz bo'sh so'rov
    # esa butun tarixni tortib olardi, shuning uchun u rad etiladi.
    if not (query or "").strip() and not (from_peer is not None or from_user or filter):
        return ServerHits(error_text="error: `query` is required")
    if (err := _guard(account_id, BUCKET_SEARCH)) is not None:
        return _rate_limited(BUCKET_SEARCH, err)

    flt = msg_filter(filter, tlt) if filter else tlt.InputMessagesFilterEmpty()
    if flt is None:
        return ServerHits(error_text=f"error: unknown filter {filter!r}")

    if peer is None:
        if chat_id is None:
            return ServerHits(error_text="error: `chat_id` or `peer` is required")
        peer, err = await _peer_for_chat(account_id, chat_id)
        if err is not None:
            return ServerHits(error_text=err)

    from_id = from_peer
    if from_id is None and from_user:
        from_id, err = await resolve_peer(account_id, from_user)
        if err is not None:
            return ServerHits(error_text=err)

    res, err = await send_request(
        account_id,
        tlf.messages.SearchRequest(
            peer=peer,
            q=query,
            filter=flt,
            min_date=since,
            max_date=until,
            offset_id=0,
            add_offset=0,
            limit=max(1, min(int(limit), SERVER_LIMIT_MAX)),
            max_id=0,
            min_id=0,
            hash=0,
            from_id=from_id,
            top_msg_id=topic_id,
        ),
    )
    if err is not None:
        return ServerHits(error_text=err, calls=1)
    index = entity_index(res)
    hits = [
        hit_of(m, index, chat_title=chat_title) for m in list(getattr(res, "messages", None) or [])
    ]
    for h in hits:
        # M7: `chat_id` faqat **haqiqiy** lokal `chats.id` bo'lganda qo'yiladi.
        # Ilgari "Saved Messages" yo'li 0 ni uzatardi va 0 keyingi bosqichlarda
        # chat sifatida o'qilardi (sitata, havola prefiksi, qamrov).
        if chat_id:
            h.chat_id = chat_id
        h.saved = saved
    return ServerHits(hits=hits, calls=1)


async def resolve_peer(account_id: int, selector: str) -> tuple[Any, str | None]:
    """`@username` → `InputPeer` (**resolve** bucket'i, kesh yo'q — tool keshlaydi)."""
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    name = (selector or "").strip().lstrip("@")
    if not name or name.isdigit():
        return None, f'error: "{selector}" is not a @username'
    if (err := _guard(account_id, BUCKET_RESOLVE)) is not None:
        return None, err
    res, err = await send_request(
        account_id,
        tlf.contacts.ResolveUsernameRequest(username=name),
        explain={
            "UsernameNotOccupiedError": f"no Telegram user/channel named @{name}",
            "UsernameInvalidError": f"@{name} is not a valid username",
        },
    )
    if err is not None:
        return None, err
    for ent in list(getattr(res, "chats", None) or []) + list(getattr(res, "users", None) or []):
        ident = getattr(ent, "id", None)
        access_hash = getattr(ent, "access_hash", None)
        if not ident:
            continue
        if isinstance(ent, tlt.User) and access_hash is not None:
            return tlt.InputPeerUser(user_id=int(ident), access_hash=int(access_hash)), None
        if isinstance(ent, tlt.Channel) and access_hash is not None:
            return tlt.InputPeerChannel(channel_id=int(ident), access_hash=int(access_hash)), None
        if isinstance(ent, tlt.Chat):
            return tlt.InputPeerChat(chat_id=int(ident)), None
    return None, f"error: could not resolve @{name} to a chat or user"


# ─── 2. Barcha chatlar bo'ylab (messages.SearchGlobal) ───────────────────────


async def server_search_global(
    account_id: int,
    query: str,
    *,
    scope: str = "all",
    archived: bool = False,
    filter: str | None = None,  # TL parametri nomi bilan bir xil
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 20,
) -> ServerHits:
    """`messages.SearchGlobal` — akkaunt **kirgan** chatlar bo'ylab.

    Muhim: bu global Telegram qidiruvi EMAS. U faqat foydalanuvchi a'zo
    bo'lgan chatlarni ko'radi; ochiq kanallar `server_discover` orqali.
    """
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    # `SearchGlobal` da `from_id` yo'q — bo'sh so'rov faqat `filter` bilan
    # ma'noli ("hamma joydagi havolalar/fayllar").
    if not (query or "").strip() and not filter:
        return ServerHits(error_text="error: `query` is required")
    flags = GLOBAL_SCOPES.get(scope.strip().lower())
    if flags is None:
        known = ", ".join(GLOBAL_SCOPES)
        return ServerHits(error_text=f"error: unknown scope; use one of: {known}")
    if (err := _guard(account_id, BUCKET_SEARCH)) is not None:
        return _rate_limited(BUCKET_SEARCH, err)

    flt = msg_filter(filter, tlt) if filter else tlt.InputMessagesFilterEmpty()
    if flt is None:
        return ServerHits(error_text=f"error: unknown filter {filter!r}")

    res, err = await send_request(
        account_id,
        tlf.messages.SearchGlobalRequest(
            q=query,
            filter=flt,
            min_date=since,
            max_date=until,
            offset_rate=0,
            offset_peer=tlt.InputPeerEmpty(),
            offset_id=0,
            limit=max(1, min(int(limit), SERVER_LIMIT_MAX)),
            folder_id=1 if archived else None,
            **flags,
        ),
    )
    if err is not None:
        return ServerHits(error_text=err, calls=1)
    index = entity_index(res)
    hits = [hit_of(m, index) for m in list(getattr(res, "messages", None) or [])]
    return ServerHits(hits=hits, calls=1)


async def server_search_saved(
    account_id: int,
    query: str,
    *,
    filter: str | None = None,  # TL parametri nomi bilan bir xil
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 20,
) -> ServerHits:
    """`messages.Search` peer=self — "Saved Messages".

    `chat_id=None`: Saved Messages lokal `chats` jadvalida alohida qator emas,
    shuning uchun topilmalar tashqi (external-like) sifatida ishlanadi —
    dalilda "Saved Messages" deb belgilanadi va sitata `#x<n>` bo'ladi (M7).
    """
    from telethon.tl import types as tlt

    return await server_search_chat(
        account_id,
        None,
        query,
        saved=True,
        filter=filter,
        since=since,
        until=until,
        limit=limit,
        peer=tlt.InputPeerSelf(),
        chat_title="Saved Messages",
    )


# ─── 3. Kalendar (messages.GetSearchResultsCalendar) ─────────────────────────


async def calendar_rows(
    account_id: int,
    chat_id: int,
    *,
    filter: str | None = None,  # TL parametri nomi bilan bir xil
    query: str | None = None,
    peer: Any = None,
) -> tuple[list[tuple[date, int]], int | None, str | None]:
    """Xom kalendar: `(rows, total, error_text)`. `rows` — yangisidan eskisiga."""
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    if (err := _guard(account_id, BUCKET_SEARCH)) is not None:
        return [], None, err
    flt = msg_filter(filter, tlt) if filter else tlt.InputMessagesFilterEmpty()
    if flt is None:
        return [], None, f"error: unknown filter {filter!r}"
    if peer is None:
        peer, err = await _peer_for_chat(account_id, chat_id)
        if err is not None:
            return [], None, err

    kwargs: dict[str, Any] = {"peer": peer, "filter": flt, "offset_id": 0, "offset_date": None}
    if query:
        kwargs["q"] = query
    res, err = await send_request(
        account_id, tlf.messages.GetSearchResultsCalendarRequest(**kwargs)
    )
    if err is not None:
        return [], None, err
    rows: list[tuple[date, int]] = []
    for p in list(getattr(res, "periods", None) or []):
        when = getattr(p, "date", None)
        if when is None:
            continue
        rows.append((when.date(), int(getattr(p, "count", 0) or 0)))
    rows.sort(key=lambda r: r[0], reverse=True)
    total = getattr(res, "count", None)
    return rows, (int(total) if isinstance(total, int) else None), None


async def server_calendar(
    account_id: int,
    chat_id: int,
    filter: str | None = None,  # TL parametri nomi bilan bir xil
) -> list[tuple[date, int]]:
    """Qaysi kunlarda mavzu bor — `(sana, soni)`, yangisidan eskisiga.

    Xato/shift holatida **bo'sh ro'yxat** (kalendar — yordamchi signal,
    uning yo'qligi quvurni to'xtatmaydi).
    """
    rows, _total, err = await calendar_rows(account_id, chat_id, filter=filter)
    if err is not None:
        log.info("tg_search.calendar_skipped", chat_id=chat_id, reason=err[:120])
    return rows


# ─── 4. Ochiq kanallar (channels.SearchPosts) — Stars YO'Q ───────────────────


def flood_reset_at(flood: Any) -> str:
    """`CheckSearchPostsFlood.wait_till` → "YYYY-MM-DD HH:MM UTC" (yo'q bo'lsa "")."""
    wait_till = getattr(flood, "wait_till", None)
    if not wait_till:
        return ""
    return datetime.fromtimestamp(int(wait_till), tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


#: Premium'i yo'q akkauntlar — jarayon xotirasida, akkaunt bo'yicha bir marta.
#: `users.GetFullUser(self).is_premium` bilan **oldindan** tekshirish mumkin
#: emas edi: bu ortiqcha RPC va baribir har chaqiruvda xato ushlash kerak.
#: Shuning uchun xatoning o'zi signal — ikkinchi marta so'ralmaydi.
PREMIUM_MISSING: set[int] = set()


def is_premium_error(err: str | None) -> bool:
    """`channels.SearchPosts` / `CheckSearchPostsFlood` Premium talab qildimi?"""
    return bool(err) and "PremiumAccountRequired" in str(err)


def premium_note() -> str:
    return (
        "public post search (channels.SearchPosts) requires Telegram Premium on this "
        "account; falling back to name search (contacts.Search) + messages.Search in "
        "the public channels it finds"
    )


def flood_note(flood: Any) -> str:
    when = flood_reset_at(flood) or "unknown time"
    return (
        f"free full-text post-search slots exhausted (remains=0, resets at {when}); "
        "not spending Stars. Use a #hashtag query, or fall back to search_global / "
        "search_messages (local)"
    )


async def server_discover(account_id: int, query: str, *, limit: int = 10) -> ServerHits:
    """Ochiq kanal postlari — **faqat bepul slot yoki hashtag**.

    `allow_paid_stars` ATAYLAB yuborilmaydi: Stars sarflash moliyaviy amal
    (`payments.SendStarsForm` bilan bir sinf, u `DENIED` da). Avval
    `CheckSearchPostsFlood`, bepul slot bo'lmasa — hashtag rejimi yoki
    tushuntiruvchi `note`.
    """
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    query = (query or "").strip()
    if not query:
        return ServerHits(error_text="error: `query` is required")
    # Premium'siz akkauntda bu pog'ona **umuman** ishlamaydi — bir marta
    # bilib olgandan keyin RPC ham, bucket tokeni ham sarflanmaydi.
    if account_id in PREMIUM_MISSING:
        return ServerHits(calls=0, note=premium_note(), note_code="premium_required")
    if (err := _guard(account_id, BUCKET_DISCOVER)) is not None:
        return _rate_limited(BUCKET_DISCOVER, err)

    flood, err = await send_request(
        account_id, tlf.channels.CheckSearchPostsFloodRequest(query=query)
    )
    if err is not None:
        if is_premium_error(err):
            PREMIUM_MISSING.add(account_id)
            return ServerHits(calls=1, note=premium_note(), note_code="premium_required")
        return ServerHits(error_text=err, calls=1)
    free = bool(getattr(flood, "query_is_free", False))
    remains = int(getattr(flood, "remains", 0) or 0)
    request: Any | None = None
    if free or remains > 0:
        request = tlf.channels.SearchPostsRequest(
            query=query,
            offset_rate=0,
            offset_peer=tlt.InputPeerEmpty(),
            offset_id=0,
            limit=max(1, min(int(limit), SERVER_LIMIT_MAX)),
        )
    elif query.startswith("#"):
        request = tlf.channels.SearchPostsRequest(
            hashtag=query.lstrip("#"),
            offset_rate=0,
            offset_peer=tlt.InputPeerEmpty(),
            offset_id=0,
            limit=max(1, min(int(limit), SERVER_LIMIT_MAX)),
        )
    if request is None:
        # Bepul slot yo'q va so'rov hashtag emas → **hech narsa qidirilmaydi**.
        # Jonli o'lchovda (`global-04`) aynan shu yo'l 0 tashqi topilma bergan
        # edi, sabab esa faqat logda qolardi — endi u chaqiruvchiga qaytadi.
        return ServerHits(
            calls=1,
            note=flood_note(flood),
            note_code="post_search_flood",
            retry_at=flood_reset_at(flood),
        )

    res, err = await send_request(account_id, request)
    if err is not None:
        if is_premium_error(err):
            PREMIUM_MISSING.add(account_id)
            return ServerHits(calls=2, note=premium_note(), note_code="premium_required")
        return ServerHits(error_text=err, calls=2)
    index = entity_index(res)
    hits = [
        hit_of(m, index, snippet=POST_SNIPPET_CHARS)
        for m in list(getattr(res, "messages", None) or [])
    ]
    return ServerHits(hits=hits, calls=2, mode="hashtag" if request.hashtag else "free text")


@dataclass(slots=True)
class ChannelHits:
    """`contacts.Search` natijasi — kanal **nomlari**, xabar emas."""

    channels: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0
    rate_limited: str | None = None
    error_text: str | None = None


async def server_channel_search(account_id: int, query: str, *, limit: int = 10) -> ChannelHits:
    """Ochiq kanal/guruhlarni **nomi bo'yicha** topish (`contacts.Search` — READ).

    `channels.SearchPosts` bepul sloti tugaganda yagona qolgan tekin yo'l:
    post matnini bermaydi, lekin "bu mavzuda qanday ochiq kanallar bor" degan
    savolga rost javob beradi. Stars sarflanmaydi, natija saqlanmaydi.

    Shift **`BUCKET_CONTACTS`** dan, `BUCKET_DISCOVER` dan emas: `SearchPosts`
    probe'i (burst 1) bu yo'lni ochlikda qoldirardi.
    """
    from telethon.tl import functions as tlf

    query = (query or "").strip()
    if not query:
        return ChannelHits(error_text="error: `query` is required")
    if (err := _guard(account_id, BUCKET_CONTACTS)) is not None:
        return ChannelHits(rate_limited=BUCKET_CONTACTS, error_text=err)

    res, err = await send_request(
        account_id,
        tlf.contacts.SearchRequest(q=query, limit=max(1, min(int(limit), SERVER_LIMIT_MAX))),
    )
    if err is not None:
        return ChannelHits(error_text=err, calls=1)
    out: list[dict[str, Any]] = []
    for ch in list(getattr(res, "chats", None) or []):
        title = str(getattr(ch, "title", "") or "").strip()
        username = str(getattr(ch, "username", "") or "").strip()
        if not title and not username:
            continue
        out.append(
            {
                "title": title,
                "username": username,
                "members": getattr(ch, "participants_count", None),
                "broadcast": bool(getattr(ch, "broadcast", False)),
                "megagroup": bool(getattr(ch, "megagroup", False)),
                # `InputPeerChannel` uchun — `get_entity()` / `get_input_entity()`
                # chaqirilmaydi (ular allowlist'da yo'q metodlarga borishi mumkin):
                # access hash javob bilan **birga** keladi.
                "id": int(getattr(ch, "id", 0) or 0),
                "access_hash": getattr(ch, "access_hash", None),
            }
        )
    return ChannelHits(channels=out[:limit], calls=1)


# ─── 4b. Premium'siz kashfiyot: nom → o'sha kanalda messages.Search ──────────

DISCOVER_NAME_CANDIDATES = 5  # `contacts.Search` dan olinadigan nomzodlar
DISCOVER_NAME_TOP_K = 3  # ulardan nechtasida post o'qiladi
DISCOVER_NAME_MAX_QUERIES = 2  # `contacts.Search` ga yuboriladigan nom nomzodlari soni

# `contacts.Search` kanal **nomi/username**iga mos keladi — mavzu iborasi
# emas (jonli o'lchov: "IT ta'lim O'zbekistonda" → 0 ta kanal, "IT ta'lim" →
# 5 ta). Shu sababli mavzu so'rovidan qisqa "nomga o'xshagan" variantlar
# ajratiladi: joy kelishigi qo'shimchasi (`-da`, `-dan`, ...) va "haqida" /
# "bo'yicha" kabi bog'lovchilar bilan tugagan bo'lak kanal nomi bo'la olmaydi.
_LOC_SUFFIXES: tuple[str, ...] = ("dagi", "gacha", "ning", "dan", "da", "ga", "ni")
_WORD_RE = re.compile(r"[^\s]+")
_EDGE_PUNCT = ".,;:!?\"'()[]{}<>«»"  # so'z chetidan olinadi (ichidagi apostrof qolaveradi)


def _strip_loc_suffix(word: str) -> str:
    """Oxirgi so'zdan joy kelishigi qo'shimchasini olib tashlaydi (bo'lsa).

    Faqat so'z **kamida 3 belgi qoladigan** bo'lsa kesiladi — aks holda
    qisqa so'zlar ("da" kabi) butunlay yo'qolib ketardi.
    """
    low = word.lower()
    for suf in _LOC_SUFFIXES:
        if low.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: len(word) - len(suf)]
    return word


def _normalize_name_candidate(text: str, *, max_words: int = 3) -> str:
    """Erkin iborani "kanal nomiga o'xshagan" qisqa bo'lakka tushiradi.

    Stop-so'zlar ("haqida", "bo'yicha", ...) olib tashlanadi, natija
    `max_words` so'zgacha kesiladi, oxirgi so'zdan joy kelishigi qo'shimchasi
    olinadi. Hammasi stop-so'z bo'lsa — bo'sh satr (nomzod tashlab ketiladi).
    """
    words = [w.strip(_EDGE_PUNCT) for w in _WORD_RE.findall(text or "")]
    # BUTUNLAY BOSH HARF (masalan "IT") — ehtimol qisqartma, stop-so'z
    # ro'yxati bilan tasodifiy to'qnashsa ham (`uz_fold("IT") == "it"`,
    # inglizcha stop-so'z) saqlanadi.
    kept = [w for w in words if w and (w.isupper() or not is_stopword(w))]
    if not kept:
        return ""
    kept = kept[:max_words]
    kept[-1] = _strip_loc_suffix(kept[-1])
    return " ".join(kept).strip()


def discover_name_queries(
    subqueries: Sequence[str],
    entities: Sequence[str],
    *,
    max_queries: int = DISCOVER_NAME_MAX_QUERIES,
) -> list[str]:
    """Mavzu so'rovlaridan `contacts.Search` uchun qisqa **nomga o'xshagan**
    variantlar — deterministik, LLM yo'q.

    `contacts.Search` kanal nomi/username'iga mos keladi, mavzu iborasiga
    emas (module docstring'dagi jonli misol). Ustuvorlik tartibi:

    1. `entities` — reja topgan nom/mavzular (stop-so'z va joy kelishigi
       qo'shimchasi olib tashlangan holda, ≤3 so'z);
    2. har `subqueries` elementining birinchi 2 so'zi;
    3. har `subqueries` elementining birinchi so'zi (≥4 belgi bo'lsa).

    Fold (`uz_fold`) bo'yicha takrorlar olib tashlanadi, 3 belgidan qisqa
    nomzod tashlab ketiladi, natija ≤`max_queries` ga kesiladi.
    """
    raw: list[str] = []
    for ent in entities:
        norm = _normalize_name_candidate(ent)
        if norm:
            raw.append(norm)
    for sq in subqueries:
        words = (sq or "").split()
        head = " ".join(words[:2]).strip()
        if head:
            raw.append(head)
    for sq in subqueries:
        words = (sq or "").split()
        if words and len(words[0]) >= 4:
            raw.append(words[0])

    out: list[str] = []
    seen: set[str] = set()
    for cand in raw:
        cand = cand.strip()
        if len(cand) < 3:
            continue
        key = uz_fold(cand)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= max_queries:
            break
    return out


async def local_peer_ids(account_id: int) -> set[int]:
    """Akkauntning lokal `chats` dagi peer id'lari — ularni qayta kashf qilmaymiz."""
    from app.db.base import session_scope

    stmt = select(Chat.tg_peer_id).where(Chat.account_id == account_id)
    async with session_scope() as db:
        rows = (await db.execute(stmt)).all()
    return {int(p) for (p,) in rows if p}


def _peer_of_found(ch: dict[str, Any], tlt: Any) -> Any | None:
    """`contacts.Search` qatoridan `InputPeerChannel` — qo'shimcha RPC yo'q."""
    ident, access = int(ch.get("id") or 0), ch.get("access_hash")
    if not ident or access is None:
        return None
    if not (ch.get("broadcast") or ch.get("megagroup")):
        return None  # ochiq kanal/supergruh emas
    if not str(ch.get("username") or "").strip():
        return None  # yopiq (username yo'q) — havola ham, o'qish ham yo'q
    return tlt.InputPeerChannel(channel_id=ident, access_hash=int(access))


async def server_discover_by_name(
    account_id: int,
    query: str = "",
    *,
    queries: Sequence[str] | None = None,
    topic_query: str | None = None,
    top_k: int = DISCOVER_NAME_TOP_K,
    candidates: int = DISCOVER_NAME_CANDIDATES,
    limit: int = 10,
    since: datetime | None = None,
    until: datetime | None = None,
    known_peer_ids: set[int] | None = None,
) -> ServerHits:
    """Premium'siz "ochiq kanallar" yo'li: nom bo'yicha topib, ichidan o'qish.

    `channels.SearchPosts` Telegram Premium talab qiladi (jonli `global-04`
    da aynan shu chiqdi: `PremiumAccountRequiredError`). Tekin muqobil ikki
    qadam, ikkalasi ham allowlist'da READ:

    1. `contacts.Search` — **har bir** `queries` elementi uchun (≤2, odatda
       `tg_search.discover_name_queries` chiqargan qisqa "nomga o'xshagan"
       variantlar — mavzu iborasi emas: `contacts.Search` kanal nomi/
       username'iga mos keladi, "IT ta'lim O'zbekistonda" kabi to'liq
       ibora 0 ta natija beradi, "IT ta'lim" esa beradi). Birinchi so'rov
       ≥`candidates` kanal topsa, ikkinchisi **chaqirilmaydi** (byudjet).
       Topilgan kanallar id bo'yicha birlashtiriladi, `top_k` tasi
       (ikkalasida ham topilgani, keyin a'zolar soni, keyin tartib bo'yicha)
       qoladi. Lokal `chats` da bori tashlanadi.
    2. Top `top_k` kanalda `messages.Search` — **`topic_query`** (yoki
       yo'q bo'lsa `query`/birinchi nom so'rovi) bilan, chunki bu
       to'liq matn qidiruvi va mavzu iborasini tushunadi. Ochiq kanal
       tarixi a'zo bo'lmasdan o'qiladi; `InputPeer` **javob bilan kelgan**
       access hash'dan yasaladi (`get_entity()` chaqirilmaydi).

    Natija odatdagi `ServerHits`: `chat_id=None` → quyi oqimda tashqi
    (`[server external]`, manfiy sintetik id, `t.me/<username>/<id>` havolasi).
    Har `messages.Search` **search** bucket'idan o'tadi; `FloodWaitError`
    retry qilinmaydi — matn bo'lib qaytadi va pog'ona to'xtaydi.

    Eski chaqiruv shakli (`query=` bitta satr) ishlashda davom etadi: o'sha
    bitta satr ham nom, ham mavzu sifatida ishlatiladi.
    """
    from telethon.tl import types as tlt

    name_queries = [
        q for q in ((qq or "").strip() for qq in (queries if queries is not None else [query])) if q
    ]
    topic = (topic_query or query or (name_queries[0] if name_queries else "")).strip()
    if not name_queries or not topic:
        return ServerHits(error_text="error: `query` is required")

    out = ServerHits(mode="by name", note_code="discover_by_name")
    # Kalit **odatda** `contacts.Search` id'si (raqamli, ikkala so'rov
    # natijasini birlashtirish uchun); id kelmasa (masalan eski test
    # fixture'i) username'ga, u ham bo'lmasa obyektning o'ziga tushadi — bu
    # holatda dedup ishlamaydi, lekin kanal baribir ro'yxatdan tushib
    # ketmaydi (`_peer_of_found` baribir id/access_hash talab qiladi).
    merged: dict[Any, dict[str, Any]] = {}
    found_at: dict[Any, int] = {}  # kalit -> nechta nom so'rovida topilgani
    order: list[Any] = []

    def _key(ch: dict[str, Any]) -> Any:
        cid = int(ch.get("id") or 0)
        if cid:
            return cid
        uname = str(ch.get("username") or "").strip()
        return f"u:{uname}" if uname else id(ch)

    for i, nq in enumerate(name_queries):
        found = await server_channel_search(account_id, nq, limit=max(1, int(candidates)))
        out.calls += found.calls
        if found.rate_limited:
            out.rate_limited = found.rate_limited
            out.error_text = found.error_text
            out.note = f"channel name search skipped: {found.rate_limited} quota is full"
            if not merged:
                return out
            break
        if found.error_text is not None:
            out.error_text = found.error_text
            out.note = f"channel name search failed ({found.error_text}); no retry"
            if not merged:
                return out
            break
        out.name_queries.append(nq)
        for ch in found.channels:
            key = _key(ch)
            if key in merged:
                found_at[key] += 1
            else:
                merged[key] = ch
                found_at[key] = 1
                order.append(key)
        # Birinchi so'rov allaqachon to'liq to'plam qaytargan bo'lsa —
        # ikkinchi `contacts.Search` (yana bitta RPC) shart emas.
        if i == 0 and len(found.channels) >= candidates:
            break

    out.channels = [merged[cid] for cid in order]
    if not merged:
        names = " / ".join(f'"{q}"' for q in out.name_queries or name_queries)
        out.note = f"no public channels found for {names}"
        return out

    def _rank(key: Any) -> tuple[int, int, int]:
        ch = merged[key]
        members = ch.get("members")
        members_val = int(members) if isinstance(members, int) else -1
        return (-found_at[key], -members_val, order.index(key))

    ranked = sorted(order, key=_rank)
    pool_ = [(merged[cid], _peer_of_found(merged[cid], tlt)) for cid in ranked]
    usable = [(ch, peer) for ch, peer in pool_ if peer is not None]
    names = " / ".join(f'"{q}"' for q in out.name_queries)
    if not usable:
        out.note = f"no public channels outside your chats match {names}"
        return out

    known = known_peer_ids if known_peer_ids is not None else await local_peer_ids(account_id)
    fresh = [(ch, peer) for ch, peer in usable if int(ch["id"]) not in known][: max(0, int(top_k))]
    if not fresh:
        out.note = f"every public channel matching {names} is already in your chats"
        return out

    for ch, peer in fresh:
        res = await server_search_chat(
            account_id,
            None,
            topic,
            peer=peer,
            chat_title=str(ch.get("title") or ch.get("username") or ""),
            since=since,
            until=until,
            limit=limit,
        )
        out.calls += res.calls
        if res.rate_limited:
            out.rate_limited = res.rate_limited
            break
        if res.error_text is not None:
            log.info(
                "tg_search.discover_by_name_skipped",
                channel=str(ch.get("username") or ""),
                reason=res.error_text[:120],
            )
            continue
        out.searched.append(str(ch.get("username") or ""))
        out.hits.extend(res.hits)
    searched_names = ", ".join(f"@{u}" for u in out.searched)
    out.note = (
        f"name search (contacts.Search) + messages.Search in {len(out.searched)} "
        f"public channel(s): {searched_names}"
        if out.searched
        else f"found public channels for {names} but could not read their posts"
    )
    return out


# ─── 5. Saqlash — faqat ma'lum chatlar ───────────────────────────────────────


async def persist_server_hits(
    session: AsyncSession, account_id: int, hits: Sequence[ServerHit]
) -> int:
    """Server hitlarini `messages` ga yozadi — **faqat lokal `chats` da bori**.

    Bu o'sha ingest yo'li (`ingestion.message_to_row` + `upsert_messages`):
    matn `strip_invisible` dan o'tadi, `edited_at` `edit_date` dan olinadi.
    Boshqa hech narsa belgilanmaydi (chat `synced_*` hisoblari tegilmaydi —
    bu sync emas, bitta xabarni sitata qilish uchun ko'chirish).

    Noma'lum chatdan kelgan hit (discover) **saqlanmaydi**: uning `chat_id`
    `None` bo'lib qoladi va dalilda `external` deb belgilanadi.

    Yon ta'sir: har hitga `chat_id` va (saqlangan bo'lsa) `message_id`
    qo'yiladi — sitatalar lokal id fazasida bo'lishi uchun.
    """
    from app.services import ingestion as ING

    peers = {int(h.tg_peer_id) for h in hits if h.tg_peer_id}
    if not peers:
        return 0
    rows = (
        await session.execute(
            select(Chat.id, Chat.tg_peer_id, Chat.title, Chat.link_prefix, Chat.type).where(
                Chat.account_id == account_id, Chat.tg_peer_id.in_(peers)
            )
        )
    ).all()
    # M6: kalit — **(peer id, peer turi)**. Raqamli id Telegram'da tur ichida
    # unikal: `PeerUser(777)` va `PeerChannel(777)` boshqa-boshqa chatlar.
    known: dict[tuple[int, str], tuple[int, str, str]] = {}
    for cid, peer, title, prefix, ctype in rows:
        for kind, types in PEER_KIND_CHAT_TYPES.items():
            if str(ctype) in types:
                known[(int(peer), kind)] = (int(cid), title, prefix)
    if not known:
        return 0

    to_write: list[dict[str, Any]] = []
    pairs: set[tuple[int, int]] = set()
    for h in hits:
        found = (
            known.get((int(h.tg_peer_id), h.peer_kind)) if h.tg_peer_id and h.peer_kind else None
        )
        if found is None:
            if h.tg_peer_id and not h.peer_kind:
                log.info("tg_search.persist_skipped_no_peer_kind", peer=int(h.tg_peer_id))
            continue
        cid, title, prefix = found
        h.chat_id = cid
        if title:
            h.chat_title = title
        if prefix:
            h.link = f"{prefix}{h.tg_msg_id}"
        row = ING.message_to_row(cid, h.raw) if h.raw is not None else None
        if row is not None:
            to_write.append(row)
            pairs.add((cid, int(h.tg_msg_id)))
    if not to_write:
        return 0
    written = await ING.upsert_messages(to_write)

    found_ids = (
        await session.execute(
            select(Message.id, Message.chat_id, Message.tg_msg_id).where(
                tuple_(Message.chat_id, Message.tg_msg_id).in_(sorted(pairs))
            )
        )
    ).all()
    by_pair = {(int(c), int(t)): int(mid) for mid, c, t in found_ids}
    for h in hits:
        if h.chat_id is not None:
            h.message_id = by_pair.get((h.chat_id, int(h.tg_msg_id)))
    log.info("tg_search.persisted", account_id=account_id, rows=written)
    return written
