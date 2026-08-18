"""Agent tool'lari — faqat O'QISH, DB ustidan (5-bosqich, read-only).

Har tool: `ToolSpec` (JSON Schema) + `run(ctx, args) -> ToolResult`. Natijalar
ixcham (token byudjeti bilan) va **har doim** `<untrusted_data>` konvertida
modelga qaytariladi (chat kontenti). Xato — matn ("error: …"), istisno emas.

Delete/yozish tool'i YO'Q — bu registry'ga qo'shilmaydi (CLAUDE.md invariant 1).
Yozish 6-bosqichda alohida registry + `assert_writable()` + tasdiq bilan.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chat, Message
from app.llm import LLM, ToolSpec
from app.logging import get_logger
from app.services import search as S
from app.services.analytics import chat_stats, msg_link
from app.services.digests import digests_for_window

log = get_logger(__name__)

# Natija chegaralari — token byudjeti
MAX_RESULT_CHARS = 9_000  # bitta tool natijasi (≈2.5k token)
SEARCH_LIMIT_MAX = 30
RECENT_LIMIT_MAX = 100
WINDOW_DAYS_MAX = 62
CONTEXT_RADIUS_MAX = 15


@dataclass(slots=True)
class ToolContext:
    session: AsyncSession
    account_id: int
    pinned_chat_id: int | None = None  # UI'da tanlangan chat (DB id)
    llm: LLM | None = None
    now: datetime = field(default_factory=lambda: datetime.now(UTC))


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


async def list_chats(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    rows = await ctx.session.execute(
        select(Chat)
        .where(Chat.account_id == ctx.account_id, Chat.synced_total > 0)
        .order_by(Chat.last_message_at.desc().nulls_last())
        .limit(60)
    )
    items = []
    for c in rows.scalars().all():
        items.append(
            f'id={c.id} {c.type} "{c.title}"'
            + (f" @{c.username}" if c.username else "")
            + f" msgs={c.synced_total}"
            + (f" members={c.participants_count}" if c.participants_count else "")
            + (" admin" if c.is_admin else "")
            + (" [pinned]" if c.id == ctx.pinned_chat_id else "")
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
    return (
        f'<untrusted_data source="tool:{name}" status="{status}">\n{result.text}\n</untrusted_data>'
    )
