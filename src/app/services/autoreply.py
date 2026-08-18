"""Auto-reply (7-bosqich) — qoida bo'yicha yangi kiruvchi xabarlarga javob **taklifi**.

Oqim (worker, har 5 daqiqa):
  1. `auto_reply_rules.enabled` → chatdagi `last_processed_msg_id` dan yangi xabarlar
     (DB'dan — ingestion incremental cron 10 daqiqada olib keladi);
  2. o'zimizniki emas (sender != akkaunt), trigger'ga mos (hamma / mention /
     kalit so'z / savol), quiet hours emas, soatlik limit oshmagan;
  3. LLM (`Task.SEARCH`) qoida ko'rsatmasi + oxirgi N xabar (`<untrusted_data>`)
     bilan javob yozadi; `SKIP` desa — o'tkazib yuboriladi;
  4. `agent_actions.proposed` (`send_message`, `reply_to`) — foydalanuvchi UI'da
     tasdiqlaydi; chat `autonomous` bo'lsa darhol yuboriladi.
Yozish yo'li 6-bosqichdagi bilan bir xil (`write_tools.propose_or_execute`) —
guard, rate limit, audit o'zgarmaydi. Delete/edit hech qachon.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Account, ActionStatus, AgentAction, AgentRun, AutoReplyRule, Chat, Message
from app.llm import LLM, LLMError, Msg, Task
from app.logging import get_logger
from app.services.prompts import AUTOREPLY_PROMPT
from app.services.search import compact_text

log = get_logger(__name__)

TRIGGERS = ("all", "mentions", "keywords", "questions")
MAX_NEW_PER_RUN = 10  # bir qoida, bir ishga tushirishda
_Q_WORDS = (
    "qanday|qancha|nechta|qachon|qayer|nima|kim"
    "|как|сколько|когда|где|что|кто"
    "|how|what|when|where|why|who|can|is there"
)
_QUESTION = re.compile(rf"\?|\b({_Q_WORDS})\b", re.I)


class AutoReplyError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


@dataclass(slots=True)
class RuleView:
    id: int
    chat_id: int
    enabled: bool
    trigger: str
    keywords: str
    instructions: str
    max_per_hour: int
    quiet_from: int | None
    quiet_to: int | None
    last_processed_msg_id: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _view(r: AutoReplyRule) -> RuleView:
    return RuleView(
        id=r.id,
        chat_id=r.chat_id,
        enabled=r.enabled,
        trigger=r.trigger,
        keywords=r.keywords,
        instructions=r.instructions,
        max_per_hour=r.max_per_hour,
        quiet_from=r.quiet_from,
        quiet_to=r.quiet_to,
        last_processed_msg_id=r.last_processed_msg_id,
    )


# ─── sof mantiq (testlanadi) ─────────────────────────────────────────────────


def matches(rule: Any, text: str, *, mentioned: bool = False) -> bool:
    text = text or ""
    if not text.strip():
        return False
    trig = rule.trigger
    if trig == "all":
        return True
    if trig == "mentions":
        return mentioned
    if trig == "questions":
        return bool(_QUESTION.search(text))
    if trig == "keywords":
        kws = [k.strip().lower() for k in (rule.keywords or "").split(",") if k.strip()]
        low = text.lower()
        return any(k in low for k in kws)
    return False


def in_quiet_hours(rule: Any, now: datetime) -> bool:
    if rule.quiet_from is None or rule.quiet_to is None:
        return False
    h = now.astimezone(UTC).hour
    a, b = int(rule.quiet_from), int(rule.quiet_to)
    if a == b:
        return False
    return a <= h < b if a < b else (h >= a or h < b)  # tun oralig'i (22 → 7)


# ─── qoida CRUD ──────────────────────────────────────────────────────────────


async def get_rule(session: AsyncSession, *, user_id: int, chat_id: int) -> RuleView | None:
    r = (
        await session.execute(
            select(AutoReplyRule)
            .join(Chat, Chat.id == AutoReplyRule.chat_id)
            .join(Account, Account.id == Chat.account_id)
            .where(AutoReplyRule.chat_id == chat_id, Account.user_id == user_id)
        )
    ).scalar_one_or_none()
    return _view(r) if r else None


async def upsert_rule(
    session: AsyncSession, *, user_id: int, chat_id: int, data: dict[str, Any]
) -> RuleView:
    chat = await session.get(Chat, chat_id)
    if chat is None:
        raise AutoReplyError("not_found")
    acc = await session.get(Account, chat.account_id)
    if acc is None or acc.user_id != user_id:
        raise AutoReplyError("not_found")
    trigger = str(data.get("trigger") or "questions")
    if trigger not in TRIGGERS:
        raise AutoReplyError("bad_trigger", trigger)
    enabled = bool(data.get("enabled", False))
    if enabled and chat.write_mode == "read_only":
        raise AutoReplyError("read_only")
    if enabled and not chat.is_writable:
        raise AutoReplyError("not_writable")

    r = (
        await session.execute(select(AutoReplyRule).where(AutoReplyRule.chat_id == chat_id))
    ).scalar_one_or_none()
    if r is None:
        # boshlang'ich nuqta — hozirgi eng yangi xabar (eski tarixga javob bermaslik uchun)
        last = (
            await session.execute(
                select(func.max(Message.tg_msg_id)).where(Message.chat_id == chat_id)
            )
        ).scalar_one()
        r = AutoReplyRule(
            account_id=chat.account_id, chat_id=chat_id, last_processed_msg_id=int(last or 0)
        )
        session.add(r)
    r.enabled = enabled
    r.trigger = trigger
    r.keywords = str(data.get("keywords") or "")[:1000]
    r.instructions = str(data.get("instructions") or "")[:4000]
    r.max_per_hour = max(1, min(int(data.get("max_per_hour") or 5), 60))
    r.quiet_from = _hour_or_none(data.get("quiet_from"))
    r.quiet_to = _hour_or_none(data.get("quiet_to"))
    r.updated_at = datetime.now(UTC)
    await session.flush()
    log.info("autoreply.rule", chat_id=chat_id, enabled=enabled, trigger=trigger)
    return _view(r)


def _hour_or_none(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        h = int(v)
    except (TypeError, ValueError):
        return None
    return h if 0 <= h <= 23 else None


# ─── worker ──────────────────────────────────────────────────────────────────


async def _sent_last_hour(session: AsyncSession, chat_id: int) -> int:
    since = datetime.now(UTC) - timedelta(hours=1)
    n = (
        await session.execute(
            select(func.count(AgentAction.id))
            .join(AgentRun, AgentRun.id == AgentAction.run_id)
            .where(
                AgentRun.chat_id == chat_id,
                AgentRun.prompt.like("autoreply:%"),
                AgentAction.created_at >= since,
                AgentAction.status.in_(
                    (ActionStatus.PROPOSED, ActionStatus.CONFIRMED, ActionStatus.EXECUTED)
                ),
            )
        )
    ).scalar_one()
    return int(n or 0)


async def draft_reply(
    *, rule: Any, chat_title: str, target: Message, context: list[Message], llm: LLM | None = None
) -> str | None:
    """LLM javob matni yoki None (SKIP). Kontekst — untrusted konvertda."""
    lines = []
    for m in context:
        when = m.published_at.strftime("%m-%d %H:%M") if m.published_at else "?"
        who = f"user:{m.sender_id}" if m.sender_id else "—"
        lines.append(f"[{when}] #{m.tg_msg_id} {who}: {compact_text(m.text or '', max_chars=400)}")
    body = "\n".join(lines)
    user = (
        f"Rule instructions from the owner:\n{rule.instructions.strip() or '(none)'}\n\n"
        f'<untrusted_data source="telegram" chat="{chat_title}" kind="recent">\n'
        f"{body}\n</untrusted_data>\n\n"
        f"Reply to message #{target.tg_msg_id}:\n"
        f'<untrusted_data source="telegram" kind="target">\n'
        f"{compact_text(target.text or '', max_chars=1500)}\n</untrusted_data>"
    )
    try:
        res = await (llm or LLM()).chat(
            Task.SEARCH, [Msg.system(AUTOREPLY_PROMPT), Msg.user(user)], max_tokens=600
        )
    except LLMError as exc:
        log.warning("autoreply.llm_failed", error=str(exc)[:200])
        return None
    text = res.text.strip()
    if not text or text.upper().startswith("SKIP"):
        return None
    return text[:4000]


async def process_rule(
    session: AsyncSession,
    rule: AutoReplyRule,
    *,
    llm: LLM | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Bitta qoida uchun yangi xabarlarni ko'rib chiqadi. Qaytaradi: hisobot."""
    from app.services import write_tools as W
    from app.services.tools import ToolContext

    now = now or datetime.now(UTC)
    s = get_settings()
    chat = await session.get(Chat, rule.chat_id)
    acc = await session.get(Account, rule.account_id)
    if chat is None or acc is None or acc.status != "active":
        return {"skipped": "inactive"}
    if chat.write_mode == "read_only":
        return {"skipped": "read_only"}
    if in_quiet_hours(rule, now):
        return {"skipped": "quiet"}

    rows = await session.execute(
        select(Message)
        .where(Message.chat_id == chat.id, Message.tg_msg_id > rule.last_processed_msg_id)
        .order_by(Message.tg_msg_id)
        .limit(200)
    )
    new_msgs = list(rows.scalars().all())
    if not new_msgs:
        return {"new": 0}

    me = acc.tg_account_id
    budget = rule.max_per_hour - await _sent_last_hour(session, chat.id)
    proposed = 0
    considered = 0
    last_id = rule.last_processed_msg_id
    for m in new_msgs:
        last_id = max(last_id, m.tg_msg_id)
        if m.sender_id == me:
            continue  # o'zimizniki
        mentioned = bool(acc.label and acc.label.lstrip("@").lower() in (m.text or "").lower())
        if not matches(rule, m.text or "", mentioned=mentioned):
            continue
        considered += 1
        if considered > MAX_NEW_PER_RUN or budget <= 0:
            break
        ctx_rows = await session.execute(
            select(Message)
            .where(Message.chat_id == chat.id, Message.tg_msg_id < m.tg_msg_id)
            .order_by(Message.tg_msg_id.desc())
            .limit(s.autoreply_context_messages)
        )
        context = list(reversed(ctx_rows.scalars().all()))
        text = await draft_reply(
            rule=rule, chat_title=chat.title, target=m, context=context, llm=llm
        )
        if not text:
            continue
        run = AgentRun(
            user_id=acc.user_id,
            account_id=acc.id,
            chat_id=chat.id,
            prompt=f"autoreply:#{m.tg_msg_id}",
        )
        session.add(run)
        await session.flush()
        tctx = ToolContext(
            session=session, account_id=acc.id, pinned_chat_id=chat.id, llm=llm, user_id=acc.user_id
        )
        res = await W.propose_or_execute(
            session,
            tctx,
            run_id=run.id,
            tool="send_message",
            args={"chat": str(chat.id), "text": text, "reply_to": m.tg_msg_id},
        )
        if res.ok:
            proposed += 1
            budget -= 1
        else:
            log.warning(
                "autoreply.propose_failed", chat_id=chat.id, msg=m.tg_msg_id, error=res.text[:200]
            )

    rule.last_processed_msg_id = last_id
    await session.flush()
    log.info(
        "autoreply.processed",
        chat_id=chat.id,
        new=len(new_msgs),
        matched=considered,
        proposed=proposed,
    )
    return {"new": len(new_msgs), "matched": considered, "proposed": proposed}


async def enabled_rule_ids(session: AsyncSession) -> list[int]:
    rows = await session.execute(select(AutoReplyRule.id).where(AutoReplyRule.enabled.is_(True)))
    return [int(r[0]) for r in rows.all()]
