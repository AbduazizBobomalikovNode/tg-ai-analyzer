"""Javob sifatini baholash: foydalanuvchi bahosi (👍/👎) + avtomatik LLM-judge.

Auto-judge arzon vazifa (`Task.ROUTE`: haiku / flash-lite) bilan, javob
yuborilgandan **keyin** fon rejimida — foydalanuvchi kutmaydi. Kontekst
qayta yuborilmaydi (token tejash): faqat savol + javob (+ kontekst borligi
haqida belgi). Natija: relevance 1-5, usefulness 1-5, grounded (javob
"bilmayman/kontekst yetmadi" deb halol aytganmi yoki to'qiganmi — savolga
qarab), qisqa izoh. `WEB_AUTO_EVAL=false` bilan o'chiriladi.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.base import session_scope
from app.db.models import ConversationMessage
from app.llm import LLM, LLMError, Msg, Task
from app.logging import get_logger

log = get_logger(__name__)

from app.services.prompts import JUDGE_PROMPT  # noqa: E402

# Judge xarajati shu nomli (yashirin) suhbatga yoziladi — briefing bilan bir xil
# uslub, `limits.daily_usage` faqat `conversation_messages` ni ko'radi.
JUDGE_CONVERSATION = "Judge"


@dataclass(slots=True)
class Judgement:
    relevance: int
    usefulness: int
    grounded: bool
    note: str
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    provider: str = ""


def _clamp(v: Any, lo: int = 1, hi: int = 5) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return lo


def parse_judgement(text: str) -> Judgement | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return Judgement(
        relevance=_clamp(data.get("relevance")),
        usefulness=_clamp(data.get("usefulness")),
        grounded=bool(data.get("grounded", True)),
        note=str(data.get("note") or "")[:256],
    )


async def judge(
    question: str, answer: str, *, had_context: bool, llm: LLM | None = None
) -> Judgement | None:
    user = (
        f"Context provided: {'yes' if had_context else 'no'}\n\n"
        f"<question>\n{question[:2000]}\n</question>\n\n"
        f"<answer>\n{answer[:4000]}\n</answer>"
    )
    try:
        res = await (llm or LLM()).chat(
            Task.ROUTE, [Msg.system(JUDGE_PROMPT), Msg.user(user)], json_mode=True, max_tokens=200
        )
    except LLMError as exc:
        log.warning("eval.judge_failed", error=str(exc)[:200])
        return None
    j = parse_judgement(res.text)
    if j is not None:
        j.tokens_in, j.tokens_out = res.usage.tokens_in, res.usage.tokens_out
        j.model, j.provider = res.model, res.provider
    return j


async def _record_judge_usage(db: AsyncSession, row: ConversationMessage, j: Judgement) -> None:
    """Judge tokenlarini kunlik byudjet yo'liga qo'shadi (briefing bilan bir uslub).

    Judge har javobdan keyin yugurardi, lekin tokeni hech qayerda saqlanmasdi —
    ya'ni `limits.daily_usage` uni ko'rmasdi va real xarajat limitdan
    yashiringan edi. Endi u **o'sha suhbatga** alohida `assistant` qatori
    bo'lib tushadi (`task="judge"`), demak `daily_usage` uni sanaydi.
    Kontent yozilmaydi: faqat qisqa izoh — chat kontenti takrorlanmaydi.
    Qator **baholanayotgan suhbatga emas**, egasining "Judge" suhbatiga
    tushadi: aks holda u chat transkriptida ko'rinib qolardi.
    """
    if not (j.tokens_in or j.tokens_out):
        return
    from sqlalchemy import select

    from app.db.models import Conversation
    from app.llm.pricing import estimate_cost

    src = await db.get(Conversation, row.conversation_id)
    if src is None:
        return
    conv = (
        await db.execute(
            select(Conversation)
            .where(
                Conversation.user_id == src.user_id,
                Conversation.title == JUDGE_CONVERSATION,
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if conv is None:
        conv = Conversation(
            user_id=src.user_id, account_id=src.account_id, title=JUDGE_CONVERSATION
        )
        db.add(conv)
        await db.flush()
    cost = estimate_cost(j.provider, j.model, j.tokens_in, j.tokens_out) or 0.0
    db.add(
        ConversationMessage(
            conversation_id=conv.id,
            role="assistant",
            content=f"[judge] {j.note[:200]}",
            model=j.model,
            provider=j.provider,
            tokens_in=j.tokens_in,
            tokens_out=j.tokens_out,
            task="judge",
            cost_usd=round(cost, 6) or None,
        )
    )
    await db.flush()


async def auto_evaluate(message_id: int, question: str, answer: str, *, had_context: bool) -> None:
    """Fon vazifasi: baholab `conversation_messages.auto_*` ga yozadi. Xato — jim log."""
    s = get_settings()
    if not s.web_auto_eval:
        return
    if s.web_auto_eval_sample < 1.0 and random.random() > s.web_auto_eval_sample:  # noqa: S311
        return
    j = await judge(question, answer, had_context=had_context)
    if j is None:
        return
    async with session_scope() as db:
        row = await db.get(ConversationMessage, message_id)
        if row is None:
            return
        row.auto_relevance = j.relevance
        row.auto_usefulness = j.usefulness
        row.auto_grounded = j.grounded
        row.auto_note = j.note
        await _record_judge_usage(db, row, j)
    log.info(
        "eval.auto",
        message_id=message_id,
        relevance=j.relevance,
        usefulness=j.usefulness,
        grounded=j.grounded,
        tokens=j.tokens_in + j.tokens_out,
    )


async def rate_message(
    session: AsyncSession, *, user_id: int, message_id: int, rating: int, comment: str = ""
) -> ConversationMessage:
    """Foydalanuvchi bahosi. `rating` ∈ {1, -1, 0} (0 — bekor qilish)."""
    from app.db.models import Conversation

    row = await session.get(ConversationMessage, message_id)
    if row is None:
        raise ValueError("not_found")
    conv = await session.get(Conversation, row.conversation_id)
    if conv is None or conv.user_id != user_id:
        raise ValueError("not_found")
    if row.role != "assistant":
        raise ValueError("not_assistant")
    row.rating = None if rating == 0 else (1 if rating > 0 else -1)
    row.rating_comment = comment[:512] or None
    row.rated_at = datetime.now(UTC) if row.rating is not None else None
    await session.flush()
    return row
