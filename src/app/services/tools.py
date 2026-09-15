"""Agent tool'lari — faqat O'QISH, DB ustidan (5-bosqich, read-only).

Har tool: `ToolSpec` (JSON Schema) + `run(ctx, args) -> ToolResult`. Natijalar
ixcham (token byudjeti bilan) va **har doim** `<untrusted_data>` konvertida
modelga qaytariladi (chat kontenti). Xato — matn ("error: …"), istisno emas.

Delete/yozish tool'i YO'Q — bu registry'ga qo'shilmaydi (CLAUDE.md invariant 1).
Yozish 6-bosqichda alohida registry + `assert_writable()` + tasdiq bilan.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chat, Message
from app.llm import LLM, ToolSpec
from app.logging import get_logger
from app.mtproto.ratelimit import MethodBuckets
from app.services import search as S
from app.services import untrusted as U
from app.services.analytics import chat_stats, msg_link
from app.services.digests import digests_for_window

log = get_logger(__name__)

# Natija chegaralari — token byudjeti
MAX_RESULT_CHARS = 9_000  # bitta tool natijasi (≈2.5k token)
SEARCH_LIMIT_MAX = 30
RECENT_LIMIT_MAX = 100
WINDOW_DAYS_MAX = 62
CONTEXT_RADIUS_MAX = 15
TG_SEARCH_LIMIT_MAX = 20  # server qidiruvi — DB'dan qimmatroq, qattiqroq qopqoq

# Serverga boradigan har bir tool metod **sinfi** bo'yicha shiftlanadi
# (`mtproto/ratelimit.py`): history / search / resolve / discover. Har chaqiruv
# real MTProto so'rovi, agent esa bir savolda bir nechtasini qilishi mumkin.
# `api` va `worker` alohida jarayonlar — har biri o'z nusxasini yaratadi.
_buckets = MethodBuckets.from_settings()


@dataclass(slots=True)
class ToolContext:
    session: AsyncSession
    account_id: int
    pinned_chat_id: int | None = None  # UI'da tanlangan chat (DB id)
    llm: LLM | None = None
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    user_id: int | None = None  # rasm generatsiya limiti/egalik uchun


@dataclass(slots=True)
class ToolResult:
    text: str  # modelga ketadigan (konvertsiz) matn
    ok: bool = True
    meta: dict[str, Any] = field(default_factory=dict)  # audit/log uchun (chat, hits, …)


ToolFn = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolResult]]


@dataclass(slots=True)
class Tool:
    spec: ToolSpec
    run: ToolFn


# ─── yordamchilar ────────────────────────────────────────────────────────────


def _clip(text: str, n: int = MAX_RESULT_CHARS) -> str:
    return text if len(text) <= n else text[: n - 40].rstrip() + "\n… (truncated for budget)"


async def _resolve_chat(ctx: ToolContext, args: dict[str, Any]) -> Chat | None:
    """`chat` argumenti: DB id, @username, sarlavha bo'lagi; yo'q bo'lsa pinned."""
    raw = args.get("chat")
    q = select(Chat).where(Chat.account_id == ctx.account_id, Chat.synced_total > 0)
    if raw is None or str(raw).strip() == "":
        if ctx.pinned_chat_id is None:
            return None
        return await ctx.session.get(Chat, ctx.pinned_chat_id)
    s = str(raw).strip()
    if s.isdigit():
        chat = await ctx.session.get(Chat, int(s))
        if chat is not None and chat.account_id == ctx.account_id:
            return chat
        row = (await ctx.session.execute(q.where(Chat.tg_peer_id == int(s)))).scalars().first()
        if row is not None:
            return row
    uname = s.lstrip("@").lower()
    row = (await ctx.session.execute(q.where(Chat.username.ilike(uname)))).scalars().first()
    if row is not None:
        return row
    row = (
        (
            await ctx.session.execute(
                q.where(Chat.title.ilike(f"%{s}%")).order_by(Chat.synced_total.desc()).limit(1)
            )
        )
        .scalars()
        .first()
    )
    return row


def _line(chat: Chat, m: Message, *, with_link: bool = True) -> str:
    when = m.published_at.strftime("%Y-%m-%d %H:%M") if m.published_at else "?"
    who = (
        chat.title
        if chat.type == "channel" and not m.sender_id
        else (f"user:{m.sender_id}" if m.sender_id else "—")
    )
    body = S.compact_text(m.text or "", max_chars=500) or (
        f"[{m.media_type}]" if m.media_type else ""
    )
    metrics = " ".join(
        f"{k}:{v}"
        for k, v in (("views", m.views), ("re", m.reactions_total), ("fwd", m.forwards))
        if v
    )
    link = msg_link(chat, m.tg_msg_id) if with_link else None
    tail = f" ({metrics})" if metrics else ""
    lnk = f" {link}" if link else ""
    return f"[{when}] #{m.tg_msg_id} {who}{tail}: {body}{lnk}"


def _no_chat() -> ToolResult:
    return ToolResult(
        "error: no chat resolved. Pass `chat` (id, @username or title) or ask the user; "
        "call list_chats to see synced chats.",
        ok=False,
    )


# ─── tool'lar ────────────────────────────────────────────────────────────────


CHAT_INDEX_LIMIT = 25  # user turn'iga qo'yiladigan ixcham ro'yxat


async def chat_index(
    session: AsyncSession,
    account_id: int,
    *,
    pinned_chat_id: int | None = None,
    limit: int = 60,
    compact: bool = False,
) -> list[str]:
    """Sinxronlangan chatlar ro'yxati, oxirgi faollik bo'yicha.

    `compact=True` — agent har so'rovda user turn'ida oladigan qisqartma
    (`list_chats` chaqirig'ini ortiqcha qiladi, qarang: `agent.py`). Har
    iteratsiyada qayta yuborilgani uchun ortiqcha maydonlar tashlanadi.
    """
    rows = await session.execute(
        select(Chat)
        .where(Chat.account_id == account_id, Chat.synced_total > 0)
        .order_by(Chat.last_message_at.desc().nulls_last())
        .limit(limit)
    )
    items = []
    for c in rows.scalars().all():
        line = f'id={c.id} {c.type} "{c.title}" msgs={c.synced_total}'
        if not compact:
            line = (
                f'id={c.id} {c.type} "{c.title}"'
                + (f" @{c.username}" if c.username else "")
                + f" msgs={c.synced_total}"
                + (f" members={c.participants_count}" if c.participants_count else "")
                + (" admin" if c.is_admin else "")
            )
        if c.id == pinned_chat_id:
            line += " [pinned]"
        items.append(line)
    return items


async def list_chats(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    items = await chat_index(
        ctx.session, ctx.account_id, pinned_chat_id=ctx.pinned_chat_id, limit=60
    )
    if not items:
        return ToolResult("no synced chats yet (ingestion may still be running)", ok=False)
    return ToolResult("\n".join(items), meta={"chats": len(items)})


async def search_messages(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    query = str(args.get("query") or "").strip()
    if not query:
        return ToolResult("error: `query` is required", ok=False)
    limit = max(1, min(int(args.get("limit") or 15), SEARCH_LIMIT_MAX))
    hits = await S.hybrid_search(ctx.session, chat.id, query, limit=limit)
    since = _parse_days(args.get("days"))
    if not hits:
        return ToolResult(
            f'no matches for "{query}" in "{chat.title}"', meta={"chat": chat.title, "hits": 0}
        )
    ids = [h.tg_msg_id for h in hits]
    q = select(Message).where(Message.chat_id == chat.id, Message.tg_msg_id.in_(ids))
    if since:
        q = q.where(Message.published_at >= ctx.now - since)
    rows = {m.tg_msg_id: m for m in (await ctx.session.execute(q)).scalars().all()}
    lines = [_line(chat, rows[i]) for i in ids if i in rows]
    head = f'chat="{chat.title}" query="{query}" hits={len(lines)} (ranked, best first)'
    return ToolResult(
        _clip(head + "\n" + "\n".join(lines)), meta={"chat": chat.title, "hits": len(lines)}
    )


async def get_recent_messages(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    limit = max(1, min(int(args.get("limit") or 30), RECENT_LIMIT_MAX))
    rows = await ctx.session.execute(
        select(Message)
        .where(Message.chat_id == chat.id)
        .order_by(Message.tg_msg_id.desc())
        .limit(limit)
    )
    msgs = list(reversed(rows.scalars().all()))
    lines = [_line(chat, m) for m in msgs]
    head = f'chat="{chat.title}" latest {len(lines)} messages (oldest first)'
    return ToolResult(
        _clip(head + "\n" + "\n".join(lines)), meta={"chat": chat.title, "n": len(lines)}
    )


async def get_message_context(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    try:
        mid = int(args.get("message_id") or "")
    except (TypeError, ValueError):
        return ToolResult("error: `message_id` (integer) is required", ok=False)
    radius = max(1, min(int(args.get("radius") or 5), CONTEXT_RADIUS_MAX))
    rows = await ctx.session.execute(
        select(Message)
        .where(Message.chat_id == chat.id, Message.tg_msg_id.between(mid - radius, mid + radius))
        .order_by(Message.tg_msg_id)
    )
    msgs = rows.scalars().all()
    if not msgs:
        return ToolResult(f'no messages around #{mid} in "{chat.title}"', ok=False)
    lines = [("→ " if m.tg_msg_id == mid else "  ") + _line(chat, m) for m in msgs]
    return ToolResult(
        _clip(f'chat="{chat.title}" around #{mid}\n' + "\n".join(lines)), meta={"chat": chat.title}
    )


async def get_window_digest(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    days = max(1, min(int(args.get("days") or 7), WINDOW_DAYS_MAX))
    since = ctx.now - timedelta(days=days)
    digests = await digests_for_window(ctx.session, chat, since=since, until=ctx.now, llm=ctx.llm)
    if not digests:
        return ToolResult(
            f'no messages in the last {days} days in "{chat.title}"',
            meta={"chat": chat.title, "days": days},
        )
    parts = [f"[{d.day.date()}] ({d.msg_count} msgs) {d.text}" for d in digests]
    tokens = sum(d.tokens_in + d.tokens_out for d in digests)
    head = f'chat="{chat.title}" daily digests, last {days} days ({len(parts)} active days)'
    return ToolResult(
        _clip(head + "\n\n" + "\n\n".join(parts), n=MAX_RESULT_CHARS * 2),
        meta={"chat": chat.title, "days": days, "computed_tokens": tokens},
    )


async def get_chat_stats(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    days = max(1, min(int(args.get("days") or 7), 366))
    top_n = max(1, min(int(args.get("top_n") or 5), 10))
    st = await chat_stats(ctx.session, chat, days=days, until=ctx.now, top_n=top_n)
    d = st.to_dict()
    # ixcham JSON: bo'sh/None'larni tashlaymiz
    compact = {k: v for k, v in d.items() if v not in (None, [], {}, "")}
    for key in ("top_by_views", "top_by_reactions", "top_by_forwards"):
        if key in compact:
            compact[key] = [
                {kk: vv for kk, vv in item.items() if vv not in (None, "")} for item in compact[key]
            ]
    return ToolResult(
        _clip(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))),
        meta={"chat": chat.title, "days": days, "posts": st.posts},
    )


async def generate_image(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Rasm generatsiya (Gemini). Natija — /api/images/<id>; yuborish: send_message(image_id)."""
    from app.services import images as IMG

    if ctx.user_id is None:
        return ToolResult("error: image generation needs a signed-in user", ok=False)
    prompt = str(args.get("prompt") or "").strip()
    try:
        info = await IMG.generate(
            ctx.session,
            user_id=ctx.user_id,
            account_id=ctx.account_id,
            prompt=prompt,
            style=str(args.get("style") or ""),
            llm=ctx.llm,
        )
    except IMG.ImageError as exc:
        return ToolResult(f"error: {exc.code} — {exc.detail}", ok=False)
    return ToolResult(
        f"image generated: id={info.id} url={info.url} ({info.size_bytes // 1024} KB). "
        f"Show it to the user with markdown ![{prompt[:60]}]({info.url}). "
        "To post it, propose send_message with image_id and the caption as text.",
        meta={"image_id": info.id, "url": info.url},
    )


async def list_scheduled_messages(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Telegram serverida rejalashtirilgan postlar (messages.GetScheduledHistory — READ)."""
    from app.mtproto.pool import PoolError, pool

    if (err := _tg_guard(ctx.account_id, BUCKET_HISTORY)) is not None:
        return ToolResult(err, ok=False)
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    try:
        client = await pool.get(ctx.account_id)
        peer = await pool.input_peer(ctx.account_id, chat.tg_peer_id)
        msgs = await client.get_messages(peer, limit=50, scheduled=True)
    except PoolError as exc:
        return ToolResult(f"error: {exc.code}", ok=False)
    except Exception as exc:  # RPC xatosi — matn
        return ToolResult(f"error: {type(exc).__name__}: {str(exc)[:200]}", ok=False)
    items = []
    for m in msgs or []:
        when = getattr(m, "date", None)
        text = S.compact_text(getattr(m, "message", "") or "", max_chars=300)
        media = getattr(m, "media", None)
        items.append(
            f"[{when.strftime('%Y-%m-%d %H:%M') if when else '?'}] #{m.id}"
            f"{' [media]' if media is not None else ''}: {text}"
        )
    if not items:
        return ToolResult(
            f'no scheduled messages in "{chat.title}"', meta={"chat": chat.title, "n": 0}
        )
    return ToolResult(
        _clip(f'chat="{chat.title}" scheduled ({len(items)}):\n' + "\n".join(items)),
        meta={"chat": chat.title, "n": len(items)},
    )


# ─── Telegram server-side qidiruv ────────────────────────────────────────────
# `messages.Search` / `messages.SearchGlobal` — allowlist'da READ sifatida
# ochiq. DB qidiruvidan farqi: sinxronlanmagan tarixni ham topadi, lekin
# semantik emas (faqat kalit so'z) va har chaqiruv real MTProto so'rovi.


def _entity_name(obj: Any) -> str:
    """Telethon entity nomi — qo'shimcha so'rovsiz (`m.sender` / `m.chat` keshi)."""
    if obj is None:
        return ""
    title = getattr(obj, "title", None)
    if title:
        return str(title)
    parts = [getattr(obj, "first_name", None), getattr(obj, "last_name", None)]
    name = " ".join(p for p in parts if p)
    return name or (getattr(obj, "username", None) or "")


def _peer_id_of(peer: Any) -> int | None:
    """`PeerUser/PeerChat/PeerChannel` (yoki InputPeer*) → raqamli id."""
    for attr in ("user_id", "chat_id", "channel_id"):
        val = getattr(peer, attr, None)
        if val:
            return int(val)
    return None


def _entity_index(res: Any) -> dict[int, Any]:
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


def _who_of(m: Any, index: dict[int, Any]) -> str:
    """Muallif nomi; entity kelmagan bo'lsa `user:<id>` (— emas)."""
    pid = _peer_id_of(getattr(m, "from_id", None))
    if pid is None:  # kanal nomidan yozilgan post
        pid = _peer_id_of(getattr(m, "peer_id", None))
    if pid is None:
        return "—"
    return _entity_name(index.get(pid)) or f"user:{pid}"


def _chat_of(m: Any, index: dict[int, Any]) -> tuple[str, Any, int | None]:
    pid = _peer_id_of(getattr(m, "peer_id", None))
    ent = index.get(pid) if pid else None
    return (_entity_name(ent) or (f"chat:{pid}" if pid else "—")), ent, pid


def _tg_link(entity: Any, chat_id: int | None, msg_id: Any) -> str:
    uname = getattr(entity, "username", None)
    if uname:
        return f"https://t.me/{uname}/{msg_id}"
    return f"https://t.me/c/{chat_id}/{msg_id}" if chat_id else ""


def _raw_line(
    m: Any,
    index: dict[int, Any],
    *,
    chat_title: str | None = None,
    with_chat: bool = False,
    with_link: bool = False,
    snippet: int = 400,
) -> str:
    """Xom TL xabari → bitta qator. Manba — javobdagi entity ro'yxatlari."""
    when = getattr(m, "date", None)
    ts = when.strftime("%Y-%m-%d %H:%M") if when else "?"
    body = S.compact_text(getattr(m, "message", "") or "", max_chars=snippet)
    if not body:
        media = getattr(m, "media", None)
        body = f"[{type(media).__name__.removeprefix('MessageMedia').lower()}]" if media else ""
    title, ent, cid = _chat_of(m, index)
    where = f' in "{chat_title}"' if chat_title else (f' in "{title}"' if with_chat else "")
    link = ""
    if with_link and (url := _tg_link(ent, cid, getattr(m, "id", 0))):
        link = f" {url}"
    return f"[{ts}] #{getattr(m, 'id', '?')} {_who_of(m, index)}{where}: {body}{link}"


# ─── shift (bucket) ──────────────────────────────────────────────────────────

BUCKET_HISTORY = "history"
BUCKET_SEARCH = "search"
BUCKET_RESOLVE = "resolve"
BUCKET_DISCOVER = "discover"


def _tg_guard(account_id: int, bucket: str) -> str | None:
    """Shu sinfdagi serverga boradigan chaqiruv mumkinmi? Xato matni yoki None.

    Bo'sh bucket — xato emas, **holat**: modelga tushunarli matn qaytadi va u
    lokal qidiruvga tushadi (kutish yo'q, retry-storm yo'q).
    """
    if get_settings().tg_search_rate_per_minute <= 0:
        return "error: telegram server access is disabled (TG_SEARCH_RATE_PER_MINUTE=0)"
    if not _buckets.try_acquire(account_id, bucket):
        wait = _buckets.retry_after(account_id, bucket)
        return (
            f"error: rate_limited:{bucket} — retry in {wait}s; use search_messages (local) instead"
        )
    return None


# ─── umumiy TL yordamchilari ────────────────────────────────────────────────

# `filter` argumenti → `InputMessagesFilter*` klass nomi
_FILTERS: dict[str, str] = {
    "url": "InputMessagesFilterUrl",
    "pinned": "InputMessagesFilterPinned",
    "mentions": "InputMessagesFilterMyMentions",
    "poll": "InputMessagesFilterPoll",
    "photo": "InputMessagesFilterPhotos",
    "document": "InputMessagesFilterDocument",
    "voice": "InputMessagesFilterVoice",
    "video": "InputMessagesFilterVideo",
}


def _msg_filter(name: str, tlt: Any) -> Any | None:
    cls_name = _FILTERS.get(name.strip().lower())
    cls = getattr(tlt, cls_name, None) if cls_name else None
    return cls() if cls is not None else None


def _bad_filter(name: str) -> ToolResult:
    return ToolResult(
        f"error: unknown filter {name!r}; use one of: " + ", ".join(_FILTERS), ok=False
    )


def _parse_date(raw: Any, *, end_of_day: bool = False) -> datetime | None:
    """`YYYY-MM-DD` yoki to'liq ISO → UTC datetime. Noto'g'ri bo'lsa None."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.combine(date.fromisoformat(text), datetime.min.time())
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    if end_of_day and len(text) == 10:  # faqat sana berilgan — kun oxirigacha
        parsed = parsed + timedelta(days=1) - timedelta(seconds=1)
    return parsed


async def _run_tg_request(
    ctx: ToolContext, request: Any, *, explain: dict[str, str] | None = None
) -> tuple[Any, ToolResult | None]:
    """TL so'rovni yuboradi. Xato — modelga **matn**, istisno emas.

    `explain` — Telegram cheklovlari (xato emas, holat) uchun tushunarli matn:
    `{"ChannelForumMissingError": "this chat is not a forum"}`.
    """
    from telethon import errors as tg_errors

    from app.mtproto.pool import PoolError, pool

    try:
        client = await pool.get(ctx.account_id)
        return await client(request), None
    except tg_errors.FloodWaitError as exc:
        # Retry yo'q (CLAUDE.md: retry-storm taqiqlangan) — modelga aytamiz.
        log.warning("tools.tg_flood", account_id=ctx.account_id, seconds=exc.seconds)
        return None, ToolResult(
            f"error: flood_wait {exc.seconds}s — try search_messages (local)", ok=False
        )
    except PoolError as exc:
        return None, ToolResult(f"error: {exc.code}", ok=False)
    except Exception as exc:  # RPC xatosi — matn, istisno emas
        hint = (explain or {}).get(type(exc).__name__)
        if hint:
            return None, ToolResult(f"error: {hint}", ok=False)
        return None, ToolResult(f"error: {type(exc).__name__}: {str(exc)[:200]}", ok=False)


def _input_peer_of_entity(ent: Any, tlt: Any) -> Any | None:
    """Resolve natijasidagi entity → InputPeer (access_hash bilan)."""
    ident = getattr(ent, "id", None)
    if not ident:
        return None
    access_hash = getattr(ent, "access_hash", None)
    if isinstance(ent, tlt.User) and access_hash is not None:
        return tlt.InputPeerUser(user_id=int(ident), access_hash=int(access_hash))
    if isinstance(ent, tlt.Channel) and access_hash is not None:
        return tlt.InputPeerChannel(channel_id=int(ident), access_hash=int(access_hash))
    if isinstance(ent, tlt.Chat):
        return tlt.InputPeerChat(chat_id=int(ident))
    return None


async def _peer_for_selector(ctx: ToolContext, raw: str) -> tuple[Any, str | None]:
    """Selector (@username / id / sarlavha) → InputPeer. **Kesh birinchi**.

    Avval lokal `chats` jadvali (bepul), faqat u topmasa
    `contacts.ResolveUsername` — `resolve` bucket'i ostida (kunlik chegara
    bilan): username yechish Telegram tomonda alohida o'lchanadigan amal.
    """
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    from app.mtproto.pool import PoolError, pool

    selector = raw.strip()
    chat = await _resolve_chat(ctx, {"chat": selector})
    if chat is not None:
        try:
            return await pool.input_peer(ctx.account_id, chat.tg_peer_id), None
        except PoolError as exc:
            return None, f"error: {exc.code}"

    name = selector.lstrip("@")
    if not name or name.isdigit():
        return None, (
            f'error: "{raw}" is not a synced chat; pass a @username or a title from list_chats'
        )
    if (err := _tg_guard(ctx.account_id, BUCKET_RESOLVE)) is not None:
        return None, err
    res, failed = await _run_tg_request(
        ctx,
        tlf.contacts.ResolveUsernameRequest(username=name),
        explain={
            "UsernameNotOccupiedError": f"no Telegram user/channel named @{name}",
            "UsernameInvalidError": f"@{name} is not a valid username",
        },
    )
    if failed is not None:
        return None, failed.text
    for ent in list(getattr(res, "chats", None) or []) + list(getattr(res, "users", None) or []):
        peer = _input_peer_of_entity(ent, tlt)
        if peer is not None:
            return peer, None
    return None, f"error: could not resolve @{name} to a chat or user"


def _input_channel_of(peer: Any, tlt: Any) -> Any | None:
    if isinstance(peer, tlt.InputPeerChannel):
        return tlt.InputChannel(channel_id=peer.channel_id, access_hash=peer.access_hash)
    return None


# ─── server qidiruvi ─────────────────────────────────────────────────────────


def _search_args(args: dict[str, Any], tlt: Any) -> tuple[dict[str, Any], ToolResult | None]:
    """Umumiy qidiruv argumentlari: query, limit, filter, since/until."""
    query = str(args.get("query") or "").strip()
    if not query:
        return {}, ToolResult("error: `query` is required", ok=False)
    flt_name = str(args.get("filter") or "").strip()
    flt = _msg_filter(flt_name, tlt) if flt_name else tlt.InputMessagesFilterEmpty()
    if flt is None:
        return {}, _bad_filter(flt_name)
    since = _parse_date(args.get("since"))
    until = _parse_date(args.get("until"), end_of_day=True)
    for label, raw, parsed in (
        ("since", args.get("since"), since),
        ("until", args.get("until"), until),
    ):
        if raw and parsed is None:
            return {}, ToolResult(f"error: `{label}` must be an ISO date (YYYY-MM-DD)", ok=False)
    return {
        "q": query,
        "filter": flt,
        "filter_name": flt_name.lower(),
        "limit": max(1, min(int(args.get("limit") or 15), TG_SEARCH_LIMIT_MAX)),
        "min_date": since,
        "max_date": until,
    }, None


def _search_head(scope: str, opts: dict[str, Any], hits: int) -> str:
    bits = [scope, f'query="{opts["q"]}"']
    if opts.get("filter_name"):
        bits.append(f"filter={opts['filter_name']}")
    if opts.get("min_date"):
        bits.append(f"since={opts['min_date'].date()}")
    if opts.get("max_date"):
        bits.append(f"until={opts['max_date'].date()}")
    bits.append(f"hits={hits} (Telegram server search, newest first)")
    return " ".join(bits)


async def search_telegram(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Bitta chat ichida Telegram serverida qidirish (messages.Search — READ)."""
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    from app.mtproto.pool import PoolError, pool

    if (err := _tg_guard(ctx.account_id, BUCKET_SEARCH)) is not None:
        return ToolResult(err, ok=False)
    opts, bad = _search_args(args, tlt)
    if bad is not None:
        return bad
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()

    from_id = None
    if from_user := str(args.get("from_user") or "").strip():
        from_id, err_text = await _peer_for_selector(ctx, from_user)
        if err_text is not None:
            return ToolResult(err_text, ok=False)
    topic_id = None
    if args.get("topic_id") not in (None, ""):
        try:
            topic_id = int(args["topic_id"])
        except (TypeError, ValueError):
            return ToolResult("error: `topic_id` must be an integer", ok=False)

    try:
        peer = await pool.input_peer(ctx.account_id, chat.tg_peer_id)
    except PoolError as exc:
        return ToolResult(f"error: {exc.code}", ok=False)

    res, failed = await _run_tg_request(
        ctx,
        tlf.messages.SearchRequest(
            peer=peer,
            q=opts["q"],
            filter=opts["filter"],
            min_date=opts["min_date"],
            max_date=opts["max_date"],
            offset_id=0,
            add_offset=0,
            limit=opts["limit"],
            max_id=0,
            min_id=0,
            hash=0,
            from_id=from_id,
            top_msg_id=topic_id,
        ),
    )
    if failed is not None:
        return failed
    index = _entity_index(res)
    msgs = list(getattr(res, "messages", None) or [])
    lines = [_raw_line(m, index, chat_title=chat.title) for m in msgs]
    scope = f'chat="{chat.title}"'
    if not lines:
        return ToolResult(
            f'no matches for "{opts["q"]}" ({scope}) on Telegram server',
            meta={"query": opts["q"], "hits": 0, "source": "telegram"},
        )
    return ToolResult(
        _clip(_search_head(scope, opts, len(lines)) + "\n" + "\n".join(lines)),
        meta={"query": opts["q"], "hits": len(lines), "source": "telegram"},
    )


_GLOBAL_SCOPES = {
    "all": {},
    "channels": {"broadcasts_only": True},
    "groups": {"groups_only": True},
    "users": {"users_only": True},
}


async def search_global(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Akkaunt kirgan barcha chatlar bo'ylab qidirish (messages.SearchGlobal — READ)."""
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    if (err := _tg_guard(ctx.account_id, BUCKET_SEARCH)) is not None:
        return ToolResult(err, ok=False)
    opts, bad = _search_args(args, tlt)
    if bad is not None:
        return bad
    scope_name = str(args.get("scope") or "all").strip().lower()
    flags = _GLOBAL_SCOPES.get(scope_name)
    if flags is None:
        return ToolResult(
            "error: unknown scope; use one of: " + ", ".join(_GLOBAL_SCOPES), ok=False
        )
    archived = bool(args.get("archived"))

    res, failed = await _run_tg_request(
        ctx,
        tlf.messages.SearchGlobalRequest(
            q=opts["q"],
            filter=opts["filter"],
            min_date=opts["min_date"],
            max_date=opts["max_date"],
            offset_rate=0,
            offset_peer=tlt.InputPeerEmpty(),
            offset_id=0,
            limit=opts["limit"],
            folder_id=1 if archived else None,
            **flags,
        ),
    )
    if failed is not None:
        return failed
    index = _entity_index(res)
    msgs = list(getattr(res, "messages", None) or [])
    lines = [_raw_line(m, index, with_chat=True) for m in msgs]
    scope = f"scope={scope_name} chats" + (" (archived folder)" if archived else "")
    if not lines:
        return ToolResult(
            f'no matches for "{opts["q"]}" ({scope}) on Telegram server',
            meta={"query": opts["q"], "hits": 0, "source": "telegram"},
        )
    return ToolResult(
        _clip(_search_head(scope, opts, len(lines)) + "\n" + "\n".join(lines)),
        meta={"query": opts["q"], "hits": len(lines), "source": "telegram"},
    )


# ─── a'zolar ro'yxati ────────────────────────────────────────────────────────
# `channels.GetParticipants` — allowlist'da READ. Telefon raqami ATAYLAB
# chiqarilmaydi: natija LLM promptiga tushadi, raqam esa u yerda kerak emas.

MEMBERS_LIMIT_MAX = 200  # Telegram bir so'rovda shundan ko'p bermaydi


def _member_line(u: Any) -> str:
    name = " ".join(
        p for p in (getattr(u, "first_name", None), getattr(u, "last_name", None)) if p
    ).strip()
    uname = getattr(u, "username", None)
    tags = []
    if getattr(u, "bot", False):
        tags.append("bot")
    if getattr(u, "deleted", False):
        tags.append("deleted")
    if getattr(u, "is_self", False):
        tags.append("you")
    head = f"@{uname}" if uname else f"id:{getattr(u, 'id', '?')}"
    tail = f" [{', '.join(tags)}]" if tags else ""
    return f"{head} — {name or '—'}{tail}"


async def list_members(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Kanal/guruh a'zolari (channels.GetParticipants — READ)."""
    from telethon import errors as tg_errors

    from app.mtproto.pool import PoolError, pool

    if (err := _tg_guard(ctx.account_id, BUCKET_HISTORY)) is not None:
        return ToolResult(err, ok=False)
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    if chat.type == "private":
        return ToolResult("error: private chats have no member list", ok=False)

    limit = max(1, min(int(args.get("limit") or 100), MEMBERS_LIMIT_MAX))
    query = str(args.get("query") or "").strip()
    try:
        client = await pool.get(ctx.account_id)
        peer = await pool.input_peer(ctx.account_id, chat.tg_peer_id)
        users = await client.get_participants(peer, limit=limit, search=query or "")
    except tg_errors.ChatAdminRequiredError:
        return ToolResult(
            f'error: admin rights required to list members of "{chat.title}"', ok=False
        )
    except tg_errors.FloodWaitError as exc:
        log.warning("tools.members_flood", account_id=ctx.account_id, seconds=exc.seconds)
        return ToolResult(f"error: flood_wait {exc.seconds}s", ok=False)
    except PoolError as exc:
        return ToolResult(f"error: {exc.code}", ok=False)
    except Exception as exc:
        return ToolResult(f"error: {type(exc).__name__}: {str(exc)[:200]}", ok=False)

    lines = [_member_line(u) for u in (users or [])]
    if not lines:
        return ToolResult(
            f'no members found in "{chat.title}"' + (f' for "{query}"' if query else ""),
            meta={"chat": chat.title, "n": 0},
        )
    # `total` — Telegram bergan umumiy son (limit bilan kesilgan bo'lsa ham to'g'ri)
    total = getattr(users, "total", None) or len(lines)
    head = f'chat="{chat.title}" members={len(lines)} of {total}'
    if len(lines) < total:
        head += f" (first {len(lines)}; raise `limit` up to {MEMBERS_LIMIT_MAX})"
    return ToolResult(
        _clip(head + "\n" + "\n".join(lines)),
        meta={"chat": chat.title, "n": len(lines)},
    )


# ─── profil / kontekst ma'lumotlari ──────────────────────────────────────────
# Telegram peer turiga qarab **boshqa** TL metod kerak — DB'dagi `type`
# ustuniga ishonmang, haqiqiy InputPeer turini tekshiring:
#   InputPeerChannel → channels.GetFullChannel   (kanal va supergruh)
#   InputPeerChat    → messages.GetFullChat      (oddiy guruh)
#   InputPeerUser    → users.GetFullUser
# Noto'g'ri metod TypeError beradi (jonli sinovda tekshirilgan).


async def _peer_of(ctx: ToolContext, chat: Chat) -> Any:
    from app.mtproto.pool import pool

    return await pool.input_peer(ctx.account_id, chat.tg_peer_id)


def _fmt_counters(counters: Any, types: Any) -> str:
    names = {
        types.InputMessagesFilterPhotos: "photos",
        types.InputMessagesFilterVideo: "videos",
        types.InputMessagesFilterUrl: "links",
        types.InputMessagesFilterDocument: "files",
        types.InputMessagesFilterVoice: "voice",
    }
    out = []
    for c in counters or []:
        label = names.get(type(getattr(c, "filter", None)))
        count = getattr(c, "count", None)
        if label and count:
            out.append(f"{label}:{count}")
    return " ".join(out)


@dataclass(slots=True)
class _Target:
    peer: Any
    title: str
    username: str | None
    kind: str


async def _profile_target(
    ctx: ToolContext, args: dict[str, Any]
) -> tuple[_Target | None, ToolResult | None]:
    """Profil tool'lari uchun peer — **kesh birinchi**.

    Sinxronlangan chat bo'lsa DB'dan (bepul). Topilmasa va selector `@username`
    bo'lsa — `contacts.ResolveUsername`, ya'ni `resolve` bucket'i (daqiqada 4,
    kuniga 60): username yechish Telegram tomonda alohida o'lchanadi.
    """
    from app.mtproto.pool import PoolError, pool

    chat = await _resolve_chat(ctx, args)
    if chat is not None:
        try:
            peer = await pool.input_peer(ctx.account_id, chat.tg_peer_id)
        except PoolError as exc:
            return None, ToolResult(f"error: {exc.code}", ok=False)
        return _Target(peer, chat.title, chat.username, str(chat.type)), None

    selector = str(args.get("chat") or "").strip()
    if not selector:
        return None, _no_chat()
    peer, err_text = await _peer_for_selector(ctx, selector)
    if err_text is not None:
        return None, ToolResult(err_text, ok=False)
    name = selector.lstrip("@")
    return _Target(peer, name, None if name.isdigit() else name, "?"), None


async def get_chat_info(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Chat profili: tavsif, a'zolar soni, pin, bog'langan guruh, media inventari."""
    from telethon import errors as tg_errors
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    from app.mtproto.pool import PoolError, pool

    if (err := _tg_guard(ctx.account_id, BUCKET_HISTORY)) is not None:
        return ToolResult(err, ok=False)
    target, failed = await _profile_target(ctx, args)
    if target is None:
        return failed if failed is not None else _no_chat()

    try:
        client = await pool.get(ctx.account_id)
        peer = target.peer
        facts: list[str] = []
        if isinstance(peer, tlt.InputPeerChannel):
            f = (await client(tlf.channels.GetFullChannelRequest(channel=peer))).full_chat
        elif isinstance(peer, tlt.InputPeerChat):
            f = (await client(tlf.messages.GetFullChatRequest(chat_id=peer.chat_id))).full_chat
        else:
            return ToolResult("error: this is a private chat — use get_user_info", ok=False)

        about = (getattr(f, "about", "") or "").strip()
        n = getattr(f, "participants_count", None)
        if n is None:  # oddiy guruhda ro'yxat ichida keladi
            parts = getattr(getattr(f, "participants", None), "participants", None)
            n = len(parts) if parts else None
        for label, val in (
            ("members", n),
            ("online", getattr(f, "online_count", None)),
            ("pinned_msg", getattr(f, "pinned_msg_id", None)),
            ("linked_discussion_chat", getattr(f, "linked_chat_id", None)),
            ("slowmode_sec", getattr(f, "slowmode_seconds", None)),
        ):
            if val:
                facts.append(f"{label}={val}")

        counters = await client(
            tlf.messages.GetSearchCountersRequest(
                peer=peer,
                filters=[
                    tlt.InputMessagesFilterPhotos(),
                    tlt.InputMessagesFilterVideo(),
                    tlt.InputMessagesFilterUrl(),
                    tlt.InputMessagesFilterDocument(),
                    tlt.InputMessagesFilterVoice(),
                ],
            )
        )
        media = _fmt_counters(counters, tlt)
    except tg_errors.FloodWaitError as exc:
        return ToolResult(f"error: flood_wait {exc.seconds}s", ok=False)
    except PoolError as exc:
        return ToolResult(f"error: {exc.code}", ok=False)
    except Exception as exc:
        return ToolResult(f"error: {type(exc).__name__}: {str(exc)[:200]}", ok=False)

    uname = f" @{target.username}" if target.username else ""
    lines = [f'chat="{target.title}" type={target.kind}{uname}']
    if facts:
        lines.append(" ".join(facts))
    if media:
        lines.append(f"media: {media}")
    if about:
        lines.append(f"description: {S.compact_text(about, max_chars=600)}")
    return ToolResult(_clip("\n".join(lines)), meta={"chat": target.title})


async def get_user_info(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Foydalanuvchi profili: bio, umumiy chatlar, bot/premium belgilari."""
    from telethon import errors as tg_errors
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    from app.mtproto.pool import PoolError, pool

    if (err := _tg_guard(ctx.account_id, BUCKET_HISTORY)) is not None:
        return ToolResult(err, ok=False)
    target, failed = await _profile_target(ctx, args)
    if target is None:
        return failed if failed is not None else _no_chat()

    try:
        client = await pool.get(ctx.account_id)
        peer = target.peer
        if not isinstance(peer, tlt.InputPeerUser):
            return ToolResult("error: not a user — use get_chat_info for groups/channels", ok=False)
        res = await client(tlf.users.GetFullUserRequest(id=peer))
        full = res.full_user
        user = next((u for u in res.users if u.id == full.id), None)
    except tg_errors.FloodWaitError as exc:
        return ToolResult(f"error: flood_wait {exc.seconds}s", ok=False)
    except PoolError as exc:
        return ToolResult(f"error: {exc.code}", ok=False)
    except Exception as exc:
        return ToolResult(f"error: {type(exc).__name__}: {str(exc)[:200]}", ok=False)

    head = _member_line(user) if user is not None else f'user="{target.title}"'
    facts = []
    for label, val in (
        ("common_chats", getattr(full, "common_chats_count", None)),
        ("blocked", getattr(full, "blocked", None) or None),
        ("premium", getattr(user, "premium", None) or None),
        ("verified", getattr(user, "verified", None) or None),
    ):
        if val:
            facts.append(f"{label}={val}")
    bio = (getattr(full, "about", "") or "").strip()
    lines = [head]
    if facts:
        lines.append(" ".join(facts))
    if bio:
        lines.append(f"bio: {S.compact_text(bio, max_chars=400)}")
    return ToolResult(_clip("\n".join(lines)), meta={"chat": target.title})


async def get_reactions(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Bitta xabarga kim qanday reaksiya qo'ygan (messages.GetMessageReactionsList)."""
    from telethon import errors as tg_errors
    from telethon.tl import functions as tlf

    from app.mtproto.pool import PoolError, pool

    if (err := _tg_guard(ctx.account_id, BUCKET_HISTORY)) is not None:
        return ToolResult(err, ok=False)
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    try:
        msg_id = int(args["message_id"])
    except (KeyError, TypeError, ValueError):
        return ToolResult("error: `message_id` is required", ok=False)

    try:
        client = await pool.get(ctx.account_id)
        peer = await _peer_of(ctx, chat)
        res = await client(
            tlf.messages.GetMessageReactionsListRequest(peer=peer, id=msg_id, limit=100)
        )
    except tg_errors.BroadcastForbiddenError:
        # Telegram kanallarda kim bosganini oshkor qilmaydi — faqat umumiy son
        return ToolResult(
            "error: broadcast channels do not expose who reacted; "
            "use get_chat_stats for reaction totals",
            ok=False,
        )
    except tg_errors.FloodWaitError as exc:
        return ToolResult(f"error: flood_wait {exc.seconds}s", ok=False)
    except PoolError as exc:
        return ToolResult(f"error: {exc.code}", ok=False)
    except Exception as exc:
        return ToolResult(f"error: {type(exc).__name__}: {str(exc)[:200]}", ok=False)

    by_id = {u.id: u for u in getattr(res, "users", [])}
    lines = []
    for r in getattr(res, "reactions", []):
        custom = getattr(r.reaction, "document_id", "?")
        emo = getattr(r.reaction, "emoticon", None) or f"custom:{custom}"
        uid = getattr(getattr(r, "peer_id", None), "user_id", None)
        who = _member_line(by_id[uid]) if uid in by_id else f"id:{uid}"
        lines.append(f"{emo}  {who}")
    if not lines:
        return ToolResult(f'no reactions on #{msg_id} in "{chat.title}"', meta={"n": 0})
    total = getattr(res, "count", len(lines))
    return ToolResult(
        _clip(f'chat="{chat.title}" msg=#{msg_id} reactions={total}\n' + "\n".join(lines)),
        meta={"chat": chat.title, "n": len(lines)},
    )


async def get_post_comments(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Kanal posti ostidagi izohlar (messages.GetReplies — bog'langan guruh kerak)."""
    from telethon import errors as tg_errors
    from telethon.tl import functions as tlf

    from app.mtproto.pool import PoolError, pool

    if (err := _tg_guard(ctx.account_id, BUCKET_HISTORY)) is not None:
        return ToolResult(err, ok=False)
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    try:
        msg_id = int(args["message_id"])
    except (KeyError, TypeError, ValueError):
        return ToolResult("error: `message_id` is required", ok=False)
    limit = max(1, min(int(args.get("limit") or 30), RECENT_LIMIT_MAX))

    try:
        client = await pool.get(ctx.account_id)
        peer = await _peer_of(ctx, chat)
        res = await client(
            tlf.messages.GetRepliesRequest(
                peer=peer,
                msg_id=msg_id,
                offset_id=0,
                offset_date=None,
                add_offset=0,
                limit=limit,
                max_id=0,
                min_id=0,
                hash=0,
            )
        )
    except tg_errors.MsgIdInvalidError:
        return ToolResult(
            f"error: #{msg_id} has no comment thread (channel has no linked discussion group, "
            f"or the id is wrong)",
            ok=False,
        )
    except tg_errors.FloodWaitError as exc:
        return ToolResult(f"error: flood_wait {exc.seconds}s", ok=False)
    except PoolError as exc:
        return ToolResult(f"error: {exc.code}", ok=False)
    except Exception as exc:
        return ToolResult(f"error: {type(exc).__name__}: {str(exc)[:200]}", ok=False)

    msgs = list(reversed(list(getattr(res, "messages", []))))  # eskidan yangiga
    by_id = {u.id: u for u in getattr(res, "users", [])}
    lines = []
    for m in msgs:
        uid = getattr(getattr(m, "from_id", None), "user_id", None)
        who = _entity_name(by_id.get(uid)) or (f"id:{uid}" if uid else "—")
        when = getattr(m, "date", None)
        ts = when.strftime("%Y-%m-%d %H:%M") if when else "?"
        body = S.compact_text(getattr(m, "message", "") or "", max_chars=300)
        lines.append(f"[{ts}] #{m.id} {who}: {body}")
    if not lines:
        return ToolResult(f'no comments under #{msg_id} in "{chat.title}"', meta={"n": 0})
    total = getattr(res, "count", len(lines))
    return ToolResult(
        _clip(
            f'chat="{chat.title}" post=#{msg_id} comments={len(lines)} of {total}\n'
            + "\n".join(lines)
        ),
        meta={"chat": chat.title, "n": len(lines)},
    )


# ─── kalendar / forum / saqlangan / kashfiyot (1-bosqich, arxitektura v2 §8) ─
# Hammasi O'QISH. Telegram cheklovlari (forum emas, kvota tugagan, kanal
# yopiq) — **xato emas, holat**: modelga tushunarli matn qaytadi, aks holda u
# bir xil chaqiruvni qayta-qayta takrorlaydi.

CALENDAR_ROWS_MAX = 60
FORUM_TOPICS_MAX = 100
DISCOVER_LIMIT_MAX = 20
POST_SNIPPET_CHARS = 200


@lru_cache(maxsize=1)
def _calendar_supports_query() -> bool:
    """`messages.GetSearchResultsCalendar` shu layerda `q` ni qabul qiladimi?

    Layer 227 gacha kalendar faqat **filtr** bo'yicha ishlaydi. Telethon
    versiyasidan qat'i nazar to'g'ri javob berish uchun imzodan o'qiymiz.
    """
    from telethon.tl import functions as tlf

    params = inspect.signature(tlf.messages.GetSearchResultsCalendarRequest.__init__).parameters
    return "q" in params


async def search_calendar(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Qaysi kunlarda mavzu bo'lgan — kun bo'yicha sanoq (GetSearchResultsCalendar)."""
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    from app.mtproto.pool import PoolError, pool

    if (err := _tg_guard(ctx.account_id, BUCKET_SEARCH)) is not None:
        return ToolResult(err, ok=False)
    query = str(args.get("query") or "").strip()
    flt_name = str(args.get("filter") or "").strip()
    if not query and not flt_name:
        return ToolResult(
            "error: search_calendar needs a `query` or a `filter` — Telegram rejects a calendar "
            'request with neither. Pass e.g. filter="photo" (or url/voice/video/…), or a keyword.',
            ok=False,
        )
    flt = _msg_filter(flt_name, tlt) if flt_name else tlt.InputMessagesFilterEmpty()
    if flt is None:
        return _bad_filter(flt_name)
    if query and not _calendar_supports_query():
        return ToolResult(
            "error: this Telegram layer's calendar API takes no free-text query; pass a `filter` "
            "instead, or use search_telegram with `since`/`until` to narrow a period",
            ok=False,
        )

    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    try:
        peer = await pool.input_peer(ctx.account_id, chat.tg_peer_id)
    except PoolError as exc:
        return ToolResult(f"error: {exc.code}", ok=False)

    kwargs: dict[str, Any] = {"peer": peer, "filter": flt, "offset_id": 0, "offset_date": None}
    if query:
        kwargs["q"] = query
    res, failed = await _run_tg_request(ctx, tlf.messages.GetSearchResultsCalendarRequest(**kwargs))
    if failed is not None:
        return failed

    periods = list(getattr(res, "periods", None) or [])
    rows: list[tuple[Any, int]] = []
    for p in periods:
        when = getattr(p, "date", None)
        rows.append((when.date() if when else None, int(getattr(p, "count", 0) or 0)))
    rows = [r for r in rows if r[0] is not None]
    rows.sort(key=lambda r: r[0], reverse=True)  # yangisidan eskisiga
    if not rows:
        return ToolResult(
            f'no matching days in "{chat.title}"', meta={"chat": chat.title, "days": 0}
        )
    shown = rows[:CALENDAR_ROWS_MAX]
    head = (
        f'chat="{chat.title}" calendar'
        + (f' query="{query}"' if query else "")
        + (f" filter={flt_name.lower()}" if flt_name else "")
        + f" total={getattr(res, 'count', '?')} days={len(rows)}"
        + (f" (showing {len(shown)} newest)" if len(shown) < len(rows) else "")
    )
    body = "\n".join(f"{d}  {c}" for d, c in shown)
    return ToolResult(
        _clip(head + "\n" + body),
        meta={"chat": chat.title, "days": len(rows), "source": "telegram"},
    )


def _forum_topics_request(tlf: Any, peer: Any, query: str, limit: int) -> Any:
    """Telethon 1.44+ da `messages.GetForumTopics`, eskisida `channels.*`."""
    common = {
        "offset_date": None,
        "offset_id": 0,
        "offset_topic": 0,
        "limit": limit,
        "q": query or None,
    }
    cls = getattr(tlf.messages, "GetForumTopicsRequest", None)
    if cls is not None:
        return cls(peer=peer, **common)
    return tlf.channels.GetForumTopicsRequest(channel=peer, **common)


async def list_forum_topics(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Forum (topicli supergruh) mavzulari ro'yxati."""
    from telethon.tl import functions as tlf

    from app.mtproto.pool import PoolError, pool

    if (err := _tg_guard(ctx.account_id, BUCKET_DISCOVER)) is not None:
        return ToolResult(err, ok=False)
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        return _no_chat()
    query = str(args.get("query") or "").strip()
    try:
        peer = await pool.input_peer(ctx.account_id, chat.tg_peer_id)
    except PoolError as exc:
        return ToolResult(f"error: {exc.code}", ok=False)

    res, failed = await _run_tg_request(
        ctx,
        _forum_topics_request(tlf, peer, query, FORUM_TOPICS_MAX),
        explain={
            "ChannelForumMissingError": (
                f'"{chat.title}" is not a forum — it has no topics; use get_recent_messages or '
                "search_messages there instead"
            ),
            "ChatAdminRequiredError": f'admin rights are required to list topics of "{chat.title}"',
        },
    )
    if failed is not None:
        return failed

    lines = []
    for t in list(getattr(res, "topics", None) or []):
        title = getattr(t, "title", None)
        if title is None:  # ForumTopicDeleted — sarlavhasi yo'q
            continue
        unread = int(getattr(t, "unread_count", 0) or 0)
        tags = [
            tag
            for tag, on in (
                ("closed", getattr(t, "closed", False)),
                ("pinned", getattr(t, "pinned", False)),
            )
            if on
        ]
        lines.append(
            f'topic_id={getattr(t, "id", "?")} "{title}" top_msg=#{getattr(t, "top_message", "?")}'
            + (f" unread={unread}" if unread else "")
            + (f" [{', '.join(tags)}]" if tags else "")
        )
    if not lines:
        return ToolResult(
            f'no topics found in "{chat.title}"' + (f' for "{query}"' if query else ""),
            meta={"chat": chat.title, "n": 0},
        )
    head = (
        f'chat="{chat.title}" topics={len(lines)} of {getattr(res, "count", len(lines))} — '
        "pass topic_id to search_telegram to search inside one topic"
    )
    return ToolResult(
        _clip(head + "\n" + "\n".join(lines)),
        meta={"chat": chat.title, "n": len(lines), "source": "telegram"},
    )


async def saved_messages_search(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Foydalanuvchining "Saved Messages" ichida qidirish (messages.Search, peer=self)."""
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    if (err := _tg_guard(ctx.account_id, BUCKET_SEARCH)) is not None:
        return ToolResult(err, ok=False)
    opts, bad = _search_args(args, tlt)
    if bad is not None:
        return bad

    res, failed = await _run_tg_request(
        ctx,
        tlf.messages.SearchRequest(
            peer=tlt.InputPeerSelf(),
            q=opts["q"],
            filter=opts["filter"],
            min_date=opts["min_date"],
            max_date=opts["max_date"],
            offset_id=0,
            add_offset=0,
            limit=opts["limit"],
            max_id=0,
            min_id=0,
            hash=0,
        ),
    )
    if failed is not None:
        return failed
    index = _entity_index(res)
    lines = [
        _raw_line(m, index, chat_title="Saved Messages")
        for m in list(getattr(res, "messages", None) or [])
    ]
    if not lines:
        return ToolResult(
            f'no matches for "{opts["q"]}" in Saved Messages',
            meta={"query": opts["q"], "hits": 0, "source": "telegram"},
        )
    return ToolResult(
        _clip(_search_head('chat="Saved Messages"', opts, len(lines)) + "\n" + "\n".join(lines)),
        meta={"query": opts["q"], "hits": len(lines), "source": "telegram"},
    )


def _channel_line(ch: Any) -> str:
    uname = getattr(ch, "username", None)
    head = f"@{uname}" if uname else f"id:{getattr(ch, 'id', '?')}"
    n = getattr(ch, "participants_count", None)
    tags = [
        t
        for t, on in (
            ("verified", getattr(ch, "verified", False)),
            ("scam", getattr(ch, "scam", False)),
            ("group", getattr(ch, "megagroup", False)),
        )
        if on
    ]
    return (
        f"{head} — {getattr(ch, 'title', '') or '—'}"
        + (f" members={n}" if n else "")
        + (f" [{', '.join(tags)}]" if tags else "")
    )


def _flood_note(flood: Any) -> str:
    wait_till = getattr(flood, "wait_till", None)
    when = (
        datetime.fromtimestamp(int(wait_till), tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
        if wait_till
        else "unknown time"
    )
    return (
        f"free full-text post-search slots exhausted (remains=0, resets at {when}); "
        "not spending Stars. Use a #hashtag query, or fall back to search_global / "
        "search_messages (local)"
    )


async def discover_channels(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Yangi kanallar: tavsiyalar (`similar_to`) va bepul global post qidiruvi.

    **Stars hech qachon sarflanmaydi**: `channels.SearchPosts` ga
    `allow_paid_stars` uzatilmaydi. Avval `CheckSearchPostsFlood` — bepul slot
    bo'lmasa hashtag rejimi yoki tushuntiruvchi matn (moliyaviy amal
    `payments.SendStarsForm` bilan bir sinf, u `DENIED` da).
    """
    from telethon.tl import functions as tlf
    from telethon.tl import types as tlt

    if (err := _tg_guard(ctx.account_id, BUCKET_DISCOVER)) is not None:
        return ToolResult(err, ok=False)
    query = str(args.get("query") or "").strip()
    similar_to = str(args.get("similar_to") or "").strip()
    limit = max(1, min(int(args.get("limit") or 10), DISCOVER_LIMIT_MAX))
    blocks: list[str] = []

    if similar_to or not query:
        channel = None
        if similar_to:
            peer, err_text = await _peer_for_selector(ctx, similar_to)
            if err_text is not None:
                return ToolResult(err_text, ok=False)
            channel = _input_channel_of(peer, tlt)
            if channel is None:
                return ToolResult(
                    f'error: "{similar_to}" is not a channel/supergroup — recommendations only '
                    "exist for channels",
                    ok=False,
                )
        res, failed = await _run_tg_request(
            ctx, tlf.channels.GetChannelRecommendationsRequest(channel=channel)
        )
        if failed is not None:
            return failed
        chans = [_channel_line(c) for c in list(getattr(res, "chats", None) or [])][:limit]
        label = f'channels similar to "{similar_to}"' if similar_to else "recommended for you"
        blocks.append(
            f"{label} ({len(chans)}):\n" + "\n".join(chans)
            if chans
            else f"{label}: none returned by Telegram"
        )

    if query:
        flood, failed = await _run_tg_request(
            ctx, tlf.channels.CheckSearchPostsFloodRequest(query=query)
        )
        if failed is not None:
            return failed
        free = bool(getattr(flood, "query_is_free", False))
        remains = int(getattr(flood, "remains", 0) or 0)
        request: Any | None = None
        if free or remains > 0:
            # `allow_paid_stars` ATAYLAB berilmaydi (default None = yubormaslik)
            request = tlf.channels.SearchPostsRequest(
                query=query,
                offset_rate=0,
                offset_peer=tlt.InputPeerEmpty(),
                offset_id=0,
                limit=limit,
            )
        elif query.startswith("#"):
            request = tlf.channels.SearchPostsRequest(
                hashtag=query.lstrip("#"),
                offset_rate=0,
                offset_peer=tlt.InputPeerEmpty(),
                offset_id=0,
                limit=limit,
            )
        if request is None:
            blocks.append(_flood_note(flood))
        else:
            res, failed = await _run_tg_request(ctx, request)
            if failed is not None:
                return failed
            index = _entity_index(res)
            posts = [
                _raw_line(m, index, with_chat=True, with_link=True, snippet=POST_SNIPPET_CHARS)
                for m in list(getattr(res, "messages", None) or [])
            ]
            mode = "hashtag" if request.hashtag else "free text"
            blocks.append(
                f'public posts for "{query}" ({mode}, {len(posts)}):\n' + "\n".join(posts)
                if posts
                else f'no public posts found for "{query}" ({mode})'
            )

    if not blocks:
        return ToolResult("error: pass `query` and/or `similar_to`", ok=False)
    return ToolResult(_clip("\n\n".join(blocks)), meta={"source": "telegram"})


def _parse_days(raw: Any) -> timedelta | None:
    try:
        d = int(raw)
    except (TypeError, ValueError):
        return None
    return timedelta(days=max(1, min(d, 3660)))


# ─── registry ────────────────────────────────────────────────────────────────

_CHAT_ARG = {
    "type": "string",
    "description": "Chat selector: DB id, @username or a title fragment. Omit to use the "
    "chat the user pinned in the UI.",
}

READ_TOOLS: dict[str, Tool] = {
    "list_chats": Tool(
        ToolSpec(
            "list_chats",
            "List the user's synced Telegram chats/channels with ids, type, message counts. "
            "Call once when the target chat is unclear.",
            {"type": "object", "properties": {}, "additionalProperties": False},
        ),
        list_chats,
    ),
    "search_messages": Tool(
        ToolSpec(
            "search_messages",
            "Find messages in one chat matching a query (keyword + semantic search, ranked). "
            "Use for 'find / where / who said / posts about X'. Returns up to `limit` lines "
            "with #id, date, author, metrics, text and t.me link.",
            {
                "type": "object",
                "properties": {
                    "chat": _CHAT_ARG,
                    "query": {
                        "type": "string",
                        "description": "2-6 keywords or a short phrase in the chat's language",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": SEARCH_LIMIT_MAX,
                        "default": 15,
                    },
                    "days": {
                        "type": "integer",
                        "description": "Only messages from the last N days (optional)",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        search_messages,
    ),
    "get_recent_messages": Tool(
        ToolSpec(
            "get_recent_messages",
            "Latest N messages of a chat (oldest first). Use for 'what's new / latest posts'.",
            {
                "type": "object",
                "properties": {
                    "chat": _CHAT_ARG,
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": RECENT_LIMIT_MAX,
                        "default": 30,
                    },
                },
                "additionalProperties": False,
            },
        ),
        get_recent_messages,
    ),
    "get_message_context": Tool(
        ToolSpec(
            "get_message_context",
            "Messages around one message id (±radius) to read a thread or verify a claim.",
            {
                "type": "object",
                "properties": {
                    "chat": _CHAT_ARG,
                    "message_id": {"type": "integer"},
                    "radius": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": CONTEXT_RADIUS_MAX,
                        "default": 5,
                    },
                },
                "required": ["message_id"],
                "additionalProperties": False,
            },
        ),
        get_message_context,
    ),
    "get_window_digest": Tool(
        ToolSpec(
            "get_window_digest",
            "Compact per-day digests of a chat for the last N days (cached, cheap). Use for "
            "summaries and 'what happened this week/month' instead of reading raw messages.",
            {
                "type": "object",
                "properties": {
                    "chat": _CHAT_ARG,
                    "days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": WINDOW_DAYS_MAX,
                        "default": 7,
                    },
                },
                "additionalProperties": False,
            },
        ),
        get_window_digest,
    ),
    "get_chat_stats": Tool(
        ToolSpec(
            "get_chat_stats",
            "Numeric statistics for a chat over the last N days: posts, views (sum/avg/median), "
            "reactions, forwards, replies, views growth from snapshots, posting hours/weekdays, "
            "media mix, top posts, and the previous period for comparison. Use for any "
            "'how many / most viewed / growth / best time' question — do not count by hand.",
            {
                "type": "object",
                "properties": {
                    "chat": _CHAT_ARG,
                    "days": {"type": "integer", "minimum": 1, "maximum": 366, "default": 7},
                    "top_n": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "additionalProperties": False,
            },
        ),
        get_chat_stats,
    ),
}


READ_TOOLS["generate_image"] = Tool(
    ToolSpec(
        "generate_image",
        "Generate an illustration/cover image with AI for a post. Returns an image id and a "
        "URL you can show as markdown. Use only when the user asks for an image/picture/cover; "
        "write a concrete visual prompt (subject, composition, mood, no text in image unless "
        "asked). To publish, follow with send_message(image_id=..., text=caption).",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Visual description in English, 1-3 sentences",
                },
                "style": {
                    "type": "string",
                    "description": "Optional style: flat illustration, photo, 3d, minimal…",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    ),
    generate_image,
)
READ_TOOLS["list_scheduled_messages"] = Tool(
    ToolSpec(
        "list_scheduled_messages",
        "List posts scheduled on Telegram's side for a chat (date, id, text). Use for "
        "'what is scheduled / queued'.",
        {"type": "object", "properties": {"chat": _CHAT_ARG}, "additionalProperties": False},
    ),
    list_scheduled_messages,
)

_TG_QUERY_ARG = {
    "type": "string",
    "description": "Keywords as they literally appear in the messages (no synonyms — "
    "Telegram matches text, not meaning)",
}
_TG_LIMIT_ARG = {
    "type": "integer",
    "minimum": 1,
    "maximum": TG_SEARCH_LIMIT_MAX,
    "default": 15,
}

_FILTER_ARG = {
    "type": "string",
    "enum": sorted(_FILTERS),
    "description": "Only messages of this kind (url = messages containing links, "
    "mentions = messages mentioning you, pinned = pinned ones)",
}
_SINCE_ARG = {"type": "string", "description": "Only messages from this date on (YYYY-MM-DD)"}
_UNTIL_ARG = {"type": "string", "description": "Only messages up to this date (YYYY-MM-DD)"}

READ_TOOLS["search_telegram"] = Tool(
    ToolSpec(
        "search_telegram",
        "Keyword search inside one chat on Telegram's servers — covers the FULL history, "
        "including messages not synced to the local database. Prefer search_messages first "
        "(it is ranked and understands meaning); use this when search_messages finds nothing, "
        "when the chat is not synced yet, or when you need older history. Narrow it instead of "
        "guessing more words: `from_user` for one author, `topic_id` for one forum topic (see "
        "list_forum_topics), `filter` for links/photos/files, `since`/`until` for a period.",
        {
            "type": "object",
            "properties": {
                "chat": _CHAT_ARG,
                "query": _TG_QUERY_ARG,
                "limit": _TG_LIMIT_ARG,
                "from_user": {
                    "type": "string",
                    "description": "Only messages by this author: @username or a numeric id",
                },
                "topic_id": {
                    "type": "integer",
                    "description": "Only inside this forum topic (id from list_forum_topics)",
                },
                "filter": _FILTER_ARG,
                "since": _SINCE_ARG,
                "until": _UNTIL_ARG,
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    search_telegram,
)
READ_TOOLS["search_global"] = Tool(
    ToolSpec(
        "search_global",
        "Keyword search across the user's chats at once on Telegram's servers — it covers ONLY "
        "chats, groups and channels the account is a member of (it is not a public directory; "
        "for that use discover_channels). Use for 'which chat was X mentioned in / find that "
        "message somewhere' when the chat is unknown. Results show the chat each hit came from. "
        "`scope` narrows to channels, groups or private chats; `archived` searches the archive "
        "folder instead.",
        {
            "type": "object",
            "properties": {
                "query": _TG_QUERY_ARG,
                "limit": _TG_LIMIT_ARG,
                "scope": {
                    "type": "string",
                    "enum": sorted(_GLOBAL_SCOPES),
                    "default": "all",
                    "description": "all | channels (broadcasts) | groups | users (private chats)",
                },
                "archived": {
                    "type": "boolean",
                    "description": "Search the archived folder instead of the main list",
                },
                "filter": _FILTER_ARG,
                "since": _SINCE_ARG,
                "until": _UNTIL_ARG,
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    search_global,
)
READ_TOOLS["search_calendar"] = Tool(
    ToolSpec(
        "search_calendar",
        "Per-day message counts for one chat (Telegram's calendar): which days actually have "
        "matching messages, newest first. Use it BEFORE reading history for 'when did we talk "
        "about X / which days were busy / when were photos posted', then read only those days "
        "with search_telegram (since/until) or get_message_context. Needs a `filter` (and, on "
        "newer Telegram layers, accepts a `query`) — an empty request is rejected.",
        {
            "type": "object",
            "properties": {
                "chat": _CHAT_ARG,
                "query": {"type": "string", "description": "Keywords (newer Telegram layers only)"},
                "filter": _FILTER_ARG,
            },
            "additionalProperties": False,
        },
    ),
    search_calendar,
)
READ_TOOLS["list_forum_topics"] = Tool(
    ToolSpec(
        "list_forum_topics",
        "List the topics of a forum-style supergroup (id, title, last message id, unread count). "
        "Use when the user names a topic/branch, then pass `topic_id` to search_telegram. Chats "
        "that are not forums say so.",
        {
            "type": "object",
            "properties": {
                "chat": _CHAT_ARG,
                "query": {"type": "string", "description": "Optional topic title filter"},
            },
            "additionalProperties": False,
        },
    ),
    list_forum_topics,
)
READ_TOOLS["saved_messages_search"] = Tool(
    ToolSpec(
        "saved_messages_search",
        "Search the user's own Saved Messages (the notes-to-self chat) on Telegram's servers. "
        "Use for 'what did I save / bookmark about X', links and files the user kept for later.",
        {
            "type": "object",
            "properties": {
                "query": _TG_QUERY_ARG,
                "limit": _TG_LIMIT_ARG,
                "filter": _FILTER_ARG,
                "since": _SINCE_ARG,
                "until": _UNTIL_ARG,
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    saved_messages_search,
)
READ_TOOLS["discover_channels"] = Tool(
    ToolSpec(
        "discover_channels",
        "Find PUBLIC channels the user is not in yet: `similar_to` returns channels similar to a "
        "given one, and `query` searches public posts (free quota only — the tool never spends "
        "Stars and says so when the free quota is used up; a #hashtag query still works). With "
        "no arguments it returns Telegram's recommendations for this account. Use only for "
        "'find channels about X / what else should I follow' — for the user's OWN chats use "
        "search_global or search_messages.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic keywords or a #hashtag for public post search",
                },
                "similar_to": {
                    "type": "string",
                    "description": "@username or title of a channel to find similar ones to",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": DISCOVER_LIMIT_MAX,
                    "default": 10,
                },
            },
            "additionalProperties": False,
        },
    ),
    discover_channels,
)
READ_TOOLS["list_members"] = Tool(
    ToolSpec(
        "list_members",
        "List the members/subscribers of a group or channel (username, name, bot/deleted "
        "flags) and the total count. Use for 'who is in this chat / list the subscribers / "
        "how many members'. Needs admin rights on large channels. Phone numbers are never "
        "returned. Private chats have no member list.",
        {
            "type": "object",
            "properties": {
                "chat": _CHAT_ARG,
                "query": {
                    "type": "string",
                    "description": "Optional name/username filter",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MEMBERS_LIMIT_MAX,
                    "default": 100,
                },
            },
            "additionalProperties": False,
        },
    ),
    list_members,
)

_MSG_ID_ARG = {"type": "integer", "description": "Message id (the #N shown in results)"}

READ_TOOLS["get_chat_info"] = Tool(
    ToolSpec(
        "get_chat_info",
        "Profile of a group/channel: description, member and online counts, pinned message id, "
        "linked discussion group, slow mode, and how much media it holds (photos, videos, "
        "links, files, voice). Use for 'what is this chat about / how many members / how many "
        "photos'. For a person use get_user_info instead.",
        {
            "type": "object",
            "properties": {"chat": _CHAT_ARG},
            "additionalProperties": False,
        },
    ),
    get_chat_info,
)
READ_TOOLS["get_user_info"] = Tool(
    ToolSpec(
        "get_user_info",
        "Profile of one person: bio, username, number of chats you share, bot/premium/verified "
        "flags. Use for 'who is X / what does their bio say'. Phone numbers are never returned.",
        {
            "type": "object",
            "properties": {"chat": _CHAT_ARG},
            "additionalProperties": False,
        },
    ),
    get_user_info,
)
READ_TOOLS["get_reactions"] = Tool(
    ToolSpec(
        "get_reactions",
        "Who reacted to one message and with which emoji (groups only — broadcast channels "
        "keep this private, use get_chat_stats there). Use for 'who liked / who reacted'.",
        {
            "type": "object",
            "properties": {"chat": _CHAT_ARG, "message_id": _MSG_ID_ARG},
            "required": ["message_id"],
            "additionalProperties": False,
        },
    ),
    get_reactions,
)
READ_TOOLS["get_post_comments"] = Tool(
    ToolSpec(
        "get_post_comments",
        "Comment thread under a channel post (needs a linked discussion group). Use for 'what "
        "did people comment / any feedback on that post'.",
        {
            "type": "object",
            "properties": {
                "chat": _CHAT_ARG,
                "message_id": _MSG_ID_ARG,
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": RECENT_LIMIT_MAX,
                    "default": 30,
                },
            },
            "required": ["message_id"],
            "additionalProperties": False,
        },
    ),
    get_post_comments,
)


def tool_specs() -> list[ToolSpec]:
    return [t.spec for t in READ_TOOLS.values()]


async def run_tool(name: str, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    tool = READ_TOOLS.get(name)
    if tool is None:
        return ToolResult(f"error: unknown tool {name!r}", ok=False)
    try:
        return await tool.run(ctx, args or {})
    except Exception as exc:  # tool xatosi modelga matn sifatida — sikl davom etadi
        log.warning("tool.failed", tool=name, error=str(exc)[:200])
        return ToolResult(f"error: {type(exc).__name__}: {str(exc)[:200]}", ok=False)


def envelope(name: str, result: ToolResult) -> str:
    """Tool natijasi → modelga (ishonchsiz ma'lumot konverti)."""
    status = "ok" if result.ok else "error"
    body = U.neutralize(result.text)  # kontent konvertni yopa olmaydi (invariant 3)
    head = f'<untrusted_data source="tool:{U.attr(name)}" status="{status}">'
    return f"{head}\n{body}\n</untrusted_data>"
