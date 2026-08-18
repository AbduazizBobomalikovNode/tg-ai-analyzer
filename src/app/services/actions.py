"""Yozish amallarini boshqarish: ro'yxat, tasdiqlash, rad etish, bajarish.

Bajarish yo'li (bittasi, boshqasi yo'q):
    AgentAction(proposed) ──confirm──► confirmed ──► MTProto ──► executed | failed
                          ──reject───► rejected
Har bosqichda faqat `status / confirmed_at / result_msg_id / error` o'zgaradi —
append-only trigger boshqa maydonlarni to'sadi.

Himoya qatlamlari (har biri mustaqil):
  1. egalik: amal foydalanuvchining `agent_runs` yozuviga tegishli bo'lishi shart;
  2. TTL: `WRITE_PROPOSAL_TTL_HOURS` dan eski taklif bajarilmaydi;
  3. rate limit: akkaunt bo'yicha soatiga `WRITE_RATE_PER_HOUR` bajarilgan amal;
  4. `assert_writable(peer)` — bajarish paytida ham (guard.py);
  5. `GuardedTelegramClient` — TL allowlist (delete yo'q).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from telethon import errors as tg_errors

from app.config import get_settings
from app.db.models import ActionStatus, AgentAction, AgentRun, Chat
from app.logging import get_logger
from app.mtproto.guard import PeerBlocked, assert_writable
from app.mtproto.pool import PoolError, pool
from app.observability import WRITE_ACTIONS

log = get_logger(__name__)


class ActionError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code  # i18n: action.err.<code>
        self.detail = detail
        super().__init__(code)


@dataclass(slots=True)
class ActionView:
    id: int
    run_id: int
    tool: str
    status: str
    args: dict[str, Any]
    preview: dict[str, Any]
    target_peer_id: int | None
    result_msg_id: int | None
    error: str | None
    block_reason: str | None
    created_at: datetime | None
    confirmed_at: datetime | None
    expires_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in asdict(self).items()
        }


def _preview(a: AgentAction) -> dict[str, Any]:
    args = a.args or {}
    return {
        "chat": args.get("chat_title"),
        "chat_id": args.get("chat_id"),
        "text": args.get("text"),
        "message_id": args.get("message_id"),
        "reply_to": args.get("reply_to"),
        "schedule_at": args.get("schedule_at"),
        "image_id": args.get("image_id"),
        "image_url": f"/api/images/{args['image_id']}" if args.get("image_id") else None,
        "from_chat_id": args.get("from_chat_id"),
        "drop_author": args.get("drop_author"),
        "silent": args.get("silent"),
    }


def _view(a: AgentAction) -> ActionView:
    ttl = timedelta(hours=get_settings().write_proposal_ttl_hours)
    return ActionView(
        id=a.id,
        run_id=a.run_id,
        tool=a.tool,
        status=a.status,
        args=a.args or {},
        preview=_preview(a),
        target_peer_id=a.target_peer_id,
        result_msg_id=a.result_msg_id,
        error=a.error,
        block_reason=a.block_reason,
        created_at=a.created_at,
        confirmed_at=a.confirmed_at,
        expires_at=(a.created_at + ttl)
        if a.created_at and a.status == ActionStatus.PROPOSED
        else None,
    )


async def _owned(session: AsyncSession, user_id: int, action_id: int) -> AgentAction:
    row = (
        await session.execute(
            select(AgentAction)
            .join(AgentRun, AgentRun.id == AgentAction.run_id)
            .where(AgentAction.id == action_id, AgentRun.user_id == user_id)
        )
    ).scalar_one_or_none()
    if row is None:
        raise ActionError("not_found")
    return row


async def list_actions(
    session: AsyncSession,
    user_id: int,
    *,
    status: str | None = None,
    run_id: int | None = None,
    limit: int = 50,
) -> list[ActionView]:
    q = (
        select(AgentAction)
        .join(AgentRun, AgentRun.id == AgentAction.run_id)
        .where(AgentRun.user_id == user_id)
        .order_by(AgentAction.id.desc())
        .limit(limit)
    )
    if status:
        q = q.where(AgentAction.status == status)
    if run_id:
        q = q.where(AgentAction.run_id == run_id)
    return [_view(a) for a in (await session.execute(q)).scalars().all()]


async def actions_for_runs(
    session: AsyncSession, run_ids: list[int]
) -> dict[int, list[dict[str, Any]]]:
    if not run_ids:
        return {}
    rows = await session.execute(
        select(AgentAction).where(AgentAction.run_id.in_(run_ids)).order_by(AgentAction.id)
    )
    out: dict[int, list[dict[str, Any]]] = {}
    for a in rows.scalars().all():
        if a.tool in ("send_message", "edit_message", "forward_message", "pin_message"):
            out.setdefault(a.run_id, []).append(_view(a).to_dict())
    return out


async def reject_action(session: AsyncSession, user_id: int, action_id: int) -> ActionView:
    a = await _owned(session, user_id, action_id)
    if a.status != ActionStatus.PROPOSED:
        raise ActionError("wrong_status", a.status)
    a.status = ActionStatus.REJECTED
    a.confirmed_at = datetime.now(UTC)
    await session.flush()
    WRITE_ACTIONS.labels(a.tool, "rejected").inc()
    log.info("write.rejected", action_id=a.id, tool=a.tool)
    return _view(a)


async def confirm_action(session: AsyncSession, user_id: int, action_id: int) -> ActionView:
    a = await _owned(session, user_id, action_id)
    if a.status != ActionStatus.PROPOSED:
        raise ActionError("wrong_status", a.status)
    ttl = timedelta(hours=get_settings().write_proposal_ttl_hours)
    if a.created_at and datetime.now(UTC) - a.created_at > ttl:
        a.status = ActionStatus.REJECTED
        a.error = "expired"
        await session.flush()
        raise ActionError("expired")
    await execute_action(session, a, actor=f"user:{user_id}")
    return _view(a)


async def _rate_ok(session: AsyncSession, account_id: int) -> bool:
    limit = get_settings().write_rate_per_hour
    since = datetime.now(UTC) - timedelta(hours=1)
    n = (
        await session.execute(
            select(func.count(AgentAction.id))
            .join(AgentRun, AgentRun.id == AgentAction.run_id)
            .where(
                AgentRun.account_id == account_id,
                AgentAction.status == ActionStatus.EXECUTED,
                AgentAction.confirmed_at >= since,
            )
        )
    ).scalar_one()
    return int(n or 0) < limit


async def execute_action(session: AsyncSession, a: AgentAction, *, actor: str) -> int | None:
    """Tasdiqlangan amalni MTProto orqali bajaradi. Qaytaradi: natija xabar id."""
    run = await session.get(AgentRun, a.run_id)
    if run is None or run.account_id is None:
        raise ActionError("not_found")
    account_id = run.account_id
    args = a.args or {}
    peer_id = int(args.get("peer_id") or a.target_peer_id or 0)

    a.status = ActionStatus.CONFIRMED
    a.confirmed_at = datetime.now(UTC)
    await session.flush()

    try:
        assert_writable(peer_id)  # 4-qatlam: bajarish paytida ham
        if not await _rate_ok(session, account_id):
            raise ActionError("rate_limited")
        chat = await session.get(Chat, int(args.get("chat_id") or 0))
        if chat is None or chat.write_mode == "read_only":
            raise ActionError("read_only")
        result_id = await _run_mtproto(account_id, a.tool, args)
    except PeerBlocked as exc:
        a.status = ActionStatus.FAILED
        a.error = f"blocked: {exc.reason}"[:500]
        await session.flush()
        raise ActionError("blocked", exc.reason) from exc
    except ActionError as exc:
        a.status = ActionStatus.FAILED
        a.error = exc.code
        await session.flush()
        raise
    except (PoolError, tg_errors.RPCError, OSError, TimeoutError) as exc:
        a.status = ActionStatus.FAILED
        a.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        await session.flush()
        log.warning("write.failed", action_id=a.id, tool=a.tool, error=a.error)
        raise ActionError("telegram", type(exc).__name__) from exc

    a.status = ActionStatus.EXECUTED
    a.result_msg_id = result_id
    await session.flush()
    WRITE_ACTIONS.labels(a.tool, "executed").inc()
    log.info(
        "write.executed", action_id=a.id, tool=a.tool, peer=peer_id, result=result_id, actor=actor
    )
    return result_id


async def _run_mtproto(account_id: int, tool: str, args: dict[str, Any]) -> int | None:
    """Faqat allowlist'dagi yozish metodlari (GuardedTelegramClient tekshiradi)."""
    client = await pool.get(account_id)
    peer = await pool.input_peer(account_id, int(args["peer_id"]))
    if tool == "send_message":
        schedule = datetime.fromisoformat(args["schedule_at"]) if args.get("schedule_at") else None
        if args.get("image_id"):
            path = await _image_path(account_id, str(args["image_id"]))
            msg = await client.send_file(
                peer,
                str(path),
                caption=args.get("text") or "",
                reply_to=args.get("reply_to"),
                schedule=schedule,
            )
        else:
            msg = await client.send_message(
                peer,
                args["text"],
                reply_to=args.get("reply_to"),
                schedule=schedule,
                link_preview=True,
            )
        return int(getattr(msg, "id", 0) or 0) or None
    if tool == "edit_message":
        msg = await client.edit_message(peer, int(args["message_id"]), args["text"])
        return int(getattr(msg, "id", 0) or 0) or int(args["message_id"])
    if tool == "pin_message":
        await client.pin_message(
            peer, int(args["message_id"]), notify=not bool(args.get("silent", True))
        )
        return int(args["message_id"])
    if tool == "forward_message":
        src = await pool.input_peer(account_id, int(args["from_peer_id"]))
        res = await client.forward_messages(
            peer, int(args["message_id"]), src, drop_author=bool(args.get("drop_author"))
        )
        first = res[0] if isinstance(res, list) and res else res
        return int(getattr(first, "id", 0) or 0) or None
    raise ActionError("unknown_tool", tool)


async def _image_path(account_id: int, image_id: str) -> Any:
    from app.db.base import session_scope
    from app.services import images as IMG

    async with session_scope() as db:
        try:
            return await IMG.path_for_send(db, image_id=image_id, account_id=account_id)
        except IMG.ImageError as exc:
            raise ActionError("bad_image", exc.code) from exc


# ─── chat write mode ─────────────────────────────────────────────────────────

MODES = ("read_only", "write_with_confirm", "autonomous")


async def set_chat_write_mode(
    session: AsyncSession, *, user_id: int, chat_id: int, mode: str
) -> Chat:
    """Per-chat rejim. `autonomous` — akkauntda bir vaqtda faqat bitta chat (invariant 4)."""
    if mode not in MODES:
        raise ActionError("bad_mode", mode)
    from app.db.models import Account

    chat = await session.get(Chat, chat_id)
    if chat is None:
        raise ActionError("not_found")
    acc = await session.get(Account, chat.account_id)
    if acc is None or acc.user_id != user_id:
        raise ActionError("not_found")
    if mode == "autonomous":
        other = (
            await session.execute(
                select(Chat.id).where(
                    Chat.account_id == chat.account_id,
                    Chat.write_mode == "autonomous",
                    Chat.id != chat.id,
                )
            )
        ).first()
        if other is not None:
            raise ActionError("autonomous_limit", str(other[0]))
        if not chat.is_writable:
            raise ActionError("not_writable")
    chat.write_mode = mode
    await session.flush()
    log.info("chat.write_mode", chat_id=chat.id, mode=mode, user_id=user_id)
    return chat
