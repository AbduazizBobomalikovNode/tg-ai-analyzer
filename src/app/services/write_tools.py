"""Yozish tool'lari (6-bosqich) — taklif → tasdiq → bajarish.

Zero-risk tamoyili (CLAUDE.md invariant 2 va 4):
  * Agent yozish tool'ini chaqirsa — **hech narsa yuborilmaydi**. `agent_actions`
    ga `proposed` yozuv tushadi, model foydalanuvchiga nima qilinishini aytadi,
    UI'da ✅/❌ tugma chiqadi. Faqat `confirm` → `execute` (`services/actions.py`).
  * Istisno — chat `write_mode == autonomous` (user aniq yoqqan, akkauntga 1 ta chat):
    darhol bajariladi, baribir audit'ga tushadi.
  * Har taklif va bajarish oldidan `assert_writable(peer)` — boshqaruv boti /
    777000 / @replies ga hech qachon. Bloklansa `blocked` yozuv, istisno chiqmaydi.
  * Delete tool'i YO'Q va bo'lmaydi.

Tool'lar: send_message, edit_message, forward_message, pin_message.
Hammasi `Chat` DB id / @username / sarlavha orqali chatni topadi (o'qish tool'lari
kabi) va `chat.write_mode` ni tekshiradi (`read_only` → rad).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import WriteMode, get_settings
from app.db.models import ActionStatus, AgentAction, Chat
from app.llm import ToolSpec
from app.logging import get_logger
from app.mtproto.guard import PeerBlocked, assert_writable
from app.services.tools import _CHAT_ARG, ToolContext, ToolResult, _resolve_chat

log = get_logger(__name__)

WRITE_TOOL_NAMES = frozenset({"send_message", "edit_message", "forward_message", "pin_message"})
MAX_TEXT = 4096  # Telegram xabar limiti


@dataclass(slots=True)
class Proposal:
    """Tekshirilgan, bajarishga tayyor amal (hali bajarilmagan)."""

    tool: str
    account_id: int
    chat_id: int  # DB id
    target_peer_id: int
    args: dict[str, Any]  # normallashtirilgan, bajarish uchun yetarli
    preview: dict[str, Any] = field(default_factory=dict)  # UI karta uchun


class ProposalError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


# ─── validatsiya (sof, DB'siz qism alohida testlanadi) ───────────────────────


def _text_arg(args: dict[str, Any], key: str = "text") -> str:
    text = str(args.get(key) or "").strip()
    if not text:
        raise ProposalError("empty_text", f"`{key}` is required")
    if len(text) > MAX_TEXT:
        raise ProposalError("text_too_long", f"{len(text)} > {MAX_TEXT}")
    return text


def _int_arg(args: dict[str, Any], key: str, *, required: bool = True) -> int | None:
    raw = args.get(key)
    if raw is None or raw == "":
        if required:
            raise ProposalError("missing_arg", f"`{key}` is required")
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ProposalError("bad_arg", f"`{key}` must be an integer") from exc


def _schedule_arg(args: dict[str, Any]) -> str | None:
    raw = args.get("schedule_at")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
    except ValueError as exc:
        raise ProposalError("bad_arg", "`schedule_at` must be ISO-8601") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if dt <= datetime.now(UTC):
        raise ProposalError("bad_arg", "`schedule_at` must be in the future")
    return dt.isoformat()


async def build_proposal(ctx: ToolContext, tool: str, args: dict[str, Any]) -> Proposal:
    """Argumentlarni tekshirib, `Proposal` yasaydi. Guard shu yerda ham chaqiriladi."""
    if tool not in WRITE_TOOL_NAMES:
        raise ProposalError("unknown_tool", tool)
    chat = await _resolve_chat(ctx, args)
    if chat is None:
        raise ProposalError("no_chat", "pass `chat` (id, @username or title) or pin a chat")
    if chat.write_mode == WriteMode.READ_ONLY:
        raise ProposalError(
            "read_only",
            f'chat "{chat.title}" is read-only; the user must enable writing for it in the UI',
        )
    if not chat.is_writable and tool != "forward_message":
        raise ProposalError("not_writable", f'the account cannot post to "{chat.title}"')
    assert_writable(chat.tg_peer_id)  # PeerBlocked → chaqiruvchi blocked yozuvini yaratadi

    norm: dict[str, Any] = {
        "chat_id": chat.id,
        "peer_id": chat.tg_peer_id,
        "chat_title": chat.title,
    }
    preview: dict[str, Any] = {"chat": chat.title, "chat_id": chat.id}

    if tool == "send_message":
        image_id = str(args.get("image_id") or "").strip() or None
        if image_id:
            from app.services import images as IMG

            try:
                await IMG.path_for_send(ctx.session, image_id=image_id, account_id=ctx.account_id)
            except IMG.ImageError as exc:
                raise ProposalError("bad_image", exc.code) from exc
            text = str(args.get("text") or "").strip()  # rasm bilan matn ixtiyoriy (caption)
            if len(text) > 1024:
                raise ProposalError("text_too_long", "caption > 1024")
            norm["text"] = text
        else:
            norm["text"] = _text_arg(args)
        norm["image_id"] = image_id
        norm["reply_to"] = _int_arg(args, "reply_to", required=False)
        norm["schedule_at"] = _schedule_arg(args)
        preview.update(
            {
                "text": norm["text"],
                "reply_to": norm["reply_to"],
                "schedule_at": norm["schedule_at"],
                "image_id": image_id,
                "image_url": f"/api/images/{image_id}" if image_id else None,
            }
        )
    elif tool == "edit_message":
        norm["message_id"] = _int_arg(args, "message_id")
        norm["text"] = _text_arg(args)
        preview.update({"message_id": norm["message_id"], "text": norm["text"]})
    elif tool == "pin_message":
        norm["message_id"] = _int_arg(args, "message_id")
        norm["silent"] = bool(args.get("silent", True))
        preview.update({"message_id": norm["message_id"], "silent": norm["silent"]})
    elif tool == "forward_message":
        src = await _resolve_chat(ctx, {"chat": args.get("from_chat")})
        if src is None:
            raise ProposalError("no_chat", "`from_chat` not found")
        norm["message_id"] = _int_arg(args, "message_id")
        norm["from_chat_id"] = src.id
        norm["from_peer_id"] = src.tg_peer_id
        norm["drop_author"] = bool(args.get("drop_author", False))
        preview.update(
            {
                "message_id": norm["message_id"],
                "from_chat": src.title,
                "drop_author": norm["drop_author"],
            }
        )

    return Proposal(tool, ctx.account_id, chat.id, chat.tg_peer_id, norm, preview)


# ─── agent tomonidan chaqirilganda ───────────────────────────────────────────


async def propose_or_execute(
    session: AsyncSession, ctx: ToolContext, *, run_id: int, tool: str, args: dict[str, Any]
) -> ToolResult:
    """Agent tool chaqiruvi → `agent_actions` yozuvi. Natija — modelga matn.

    * blocked/xato → `blocked`/`failed` yozuv, ok=False;
    * `write_with_confirm` → `proposed`, ok=True (model userga tushuntiradi);
    * `autonomous` → darhol bajariladi (`services/actions.execute_action`).
    """
    from app.services import actions as ACT

    try:
        p = await build_proposal(ctx, tool, args)
    except PeerBlocked as exc:
        session.add(
            AgentAction(
                run_id=run_id,
                tool=tool,
                args=args,
                status=ActionStatus.BLOCKED,
                block_reason=exc.reason[:128],
                target_peer_id=exc.peer_id,
            )
        )
        await session.flush()
        log.warning("write.blocked", tool=tool, peer=exc.peer_id, reason=exc.reason)
        return ToolResult(f"blocked: {exc.reason}. This target can never be written to.", ok=False)
    except ProposalError as exc:
        session.add(
            AgentAction(
                run_id=run_id,
                tool=tool,
                args=args,
                status=ActionStatus.FAILED,
                error=f"{exc.code}: {exc.detail}"[:500],
            )
        )
        await session.flush()
        return ToolResult(f"error: {exc.code} — {exc.detail}", ok=False)

    action = AgentAction(
        run_id=run_id,
        tool=tool,
        args=p.args,
        status=ActionStatus.PROPOSED,
        target_peer_id=p.target_peer_id,
    )
    session.add(action)
    await session.flush()

    chat = await session.get(Chat, p.chat_id)
    if chat is not None and chat.write_mode == WriteMode.AUTONOMOUS:
        try:
            res = await ACT.execute_action(session, action, actor="autonomous")
        except ACT.ActionError as exc:
            return ToolResult(
                f"error: {exc.code} — {exc.detail}", ok=False, meta={"action_id": action.id}
            )
        return ToolResult(
            f'executed (autonomous mode): {tool} in "{p.preview["chat"]}" → message id {res}',
            meta={"action_id": action.id, "executed": True},
        )

    log.info("write.proposed", action_id=action.id, tool=tool, chat_id=p.chat_id)
    return ToolResult(
        f'proposed action #{action.id} ({tool} → "{p.preview["chat"]}") is waiting for the '
        "user's confirmation in the UI. Do not claim it was sent. Briefly tell the user what "
        "will happen and that they can confirm or reject it.",
        meta={"action_id": action.id, "proposed": True},
    )


# ─── registry ────────────────────────────────────────────────────────────────

WRITE_TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        "send_message",
        "PROPOSE sending a new message/post to a chat as the user. Nothing is sent until the "
        "user confirms in the UI (unless the chat is in autonomous mode). Use only when the "
        "user explicitly asks to send/post/publish; for drafts, just write the text in your "
        "answer. Text: final, ready to publish, in the channel's language, ≤ 4096 chars.",
        {
            "type": "object",
            "properties": {
                "chat": _CHAT_ARG,
                "text": {
                    "type": "string",
                    "description": "Final message text (plain text or simple Markdown); "
                    "with image_id it is the photo caption (≤ 1024 chars)",
                },
                "reply_to": {"type": "integer", "description": "Message id to reply to (optional)"},
                "schedule_at": {
                    "type": "string",
                    "description": "ISO-8601 datetime for a Telegram-side scheduled post",
                },
                "image_id": {
                    "type": "string",
                    "description": "id returned by generate_image to attach as a photo (optional)",
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "edit_message",
        "PROPOSE editing the text of an existing message the user posted. Requires confirmation.",
        {
            "type": "object",
            "properties": {
                "chat": _CHAT_ARG,
                "message_id": {"type": "integer"},
                "text": {"type": "string", "description": "New full text"},
            },
            "required": ["message_id", "text"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "forward_message",
        "PROPOSE forwarding (or copying with drop_author=true) one message from one chat to "
        "another. Requires confirmation.",
        {
            "type": "object",
            "properties": {
                "chat": {
                    "type": "string",
                    "description": "Destination chat (id, @username or title)",
                },
                "from_chat": {
                    "type": "string",
                    "description": "Source chat (id, @username or title)",
                },
                "message_id": {"type": "integer"},
                "drop_author": {
                    "type": "boolean",
                    "description": "true = copy without 'forwarded from'",
                },
            },
            "required": ["chat", "from_chat", "message_id"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        "pin_message",
        "PROPOSE pinning a message in a chat. Requires confirmation.",
        {
            "type": "object",
            "properties": {
                "chat": _CHAT_ARG,
                "message_id": {"type": "integer"},
                "silent": {
                    "type": "boolean",
                    "description": "Pin without notifying members (default true)",
                },
            },
            "required": ["message_id"],
            "additionalProperties": False,
        },
    ),
]


async def writable_chats_exist(session: AsyncSession, account_id: int) -> bool:
    row = await session.execute(
        select(Chat.id)
        .where(Chat.account_id == account_id, Chat.write_mode != WriteMode.READ_ONLY)
        .limit(1)
    )
    return row.first() is not None


def default_write_mode() -> str:
    return str(get_settings().default_write_mode)
