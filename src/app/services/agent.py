"""Read-only agent sikli (5-bosqich): LLM tool'larni chaqiradi, biz bajaramiz, audit yozamiz.

Xarajat nazorati:
  * `AGENT_MAX_ITERATIONS` (default 5) va har iteratsiyada ≤ `MAX_CALLS_PER_TURN` tool;
  * tool natijalari umumiy byudjeti `AGENT_TOOL_RESULT_TOKENS` — oshsa keskin kesiladi;
  * tarix qisqartirilgan (`chat_service.build_history`), system prompt barqaror (kesh);
  * oxirgi iteratsiyada tool'siz "javob ber" chaqiruvi — sikl hech qachon javobsiz tugamaydi.

Xavfsizlik: faqat `tools.READ_TOOLS`; har natija `<untrusted_data>`; har chaqiruv
`agent_actions` ga (append-only) `executed`/`failed` bilan tushadi.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import ActionStatus, AgentAction, AgentRun, Chat
from app.llm import LLM, LLMError, Msg, Task
from app.logging import get_logger
from app.services import tools as T
from app.services import write_tools as W
from app.services.prompts import AGENT_SYSTEM_PROMPT, runtime_note
from app.services.search import est_tokens

log = get_logger(__name__)

MAX_CALLS_PER_TURN = 4
FINAL_MAX_TOKENS = 3000


@dataclass(slots=True)
class AgentOutcome:
    text: str
    model: str
    provider: str
    tokens_in: int
    tokens_out: int
    iterations: int
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    latency_ms: int = 0
    run_id: int | None = None
    result_tokens: int = 0  # tool natijalari (taxminiy)
    action_ids: list[int] = field(default_factory=list)  # taklif/bajarilgan yozish amallari


async def run_agent(
    session: AsyncSession,
    *,
    user_id: int,
    account_id: int,
    question: str,
    history: list[Msg],
    pinned_chat: Chat | None,
    locale: str,
    llm: LLM | None = None,
    max_iterations: int | None = None,
) -> AgentOutcome:
    s = get_settings()
    client = llm or LLM()
    max_iter = max_iterations or s.agent_max_iterations
    result_budget = s.agent_tool_result_tokens
    started = time.monotonic()

    run = AgentRun(
        user_id=user_id,
        account_id=account_id,
        chat_id=pinned_chat.id if pinned_chat else None,
        prompt=question[:4000],
    )
    session.add(run)
    await session.flush()

    ctx = T.ToolContext(
        session=session,
        account_id=account_id,
        pinned_chat_id=pinned_chat.id if pinned_chat else None,
        llm=client,
        user_id=user_id,
    )
    note = runtime_note(
        now_iso=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        locale=locale,
        pinned_chat=pinned_chat.title if pinned_chat else None,
    )
    messages: list[Msg] = [
        Msg.system(AGENT_SYSTEM_PROMPT),
        *history,
        Msg.user(f"{question}\n\n({note})"),
    ]
    specs = T.tool_specs()
    write_enabled = await W.writable_chats_exist(session, account_id)
    if write_enabled:
        specs = specs + W.WRITE_TOOL_SPECS
    action_ids: list[int] = []

    tokens_in = tokens_out = 0
    used_result_tokens = 0
    calls_log: list[dict[str, Any]] = []
    model = provider = ""
    final_text = ""
    finish = ""
    iterations = 0

    for it in range(1, max_iter + 1):
        iterations = it
        last_round = it == max_iter
        try:
            res = await client.chat(
                Task.TOOLS,
                messages,
                tools=None if last_round else specs,
                max_tokens=FINAL_MAX_TOKENS,
            )
        except LLMError as exc:
            log.warning("agent.llm_failed", run_id=run.id, iteration=it, error=str(exc)[:200])
            raise
        tokens_in += res.usage.tokens_in
        tokens_out += res.usage.tokens_out
        model, provider, finish = res.model, res.provider, res.finish_reason

        if not res.tool_calls or last_round:
            # oxirgi raundda tool so'ralsa ham bajarilmaydi — matn yo'q bo'lsa majburiy javob
            final_text = res.text.strip() if not res.tool_calls else ""
            break

        messages.append(Msg.assistant(res.text, res.tool_calls))
        for call in res.tool_calls[:MAX_CALLS_PER_TURN]:
            t0 = time.monotonic()
            if call.name in W.WRITE_TOOL_NAMES:
                if not write_enabled:
                    result = T.ToolResult("error: writing is disabled for this account", ok=False)
                else:
                    result = await W.propose_or_execute(
                        session, ctx, run_id=run.id, tool=call.name, args=call.arguments
                    )
                if result.meta.get("action_id"):
                    action_ids.append(int(result.meta["action_id"]))
            else:
                result = await T.run_tool(call.name, ctx, call.arguments)
            text = result.text
            # umumiy byudjet — oshsa keskin qisqartiramiz
            cost = est_tokens(text)
            if used_result_tokens + cost > result_budget:
                allowed = max(200, (result_budget - used_result_tokens)) * 3
                text = text[:allowed] + "\n… (result budget exhausted; answer with what you have)"
                cost = est_tokens(text)
            used_result_tokens += cost
            messages.append(
                Msg.tool_result(
                    call.id, call.name, T.envelope(call.name, T.ToolResult(text, result.ok))
                )
            )
            entry = {
                "tool": call.name,
                "args": call.arguments,
                "ok": result.ok,
                "ms": int((time.monotonic() - t0) * 1000),
                **{
                    k: v
                    for k, v in result.meta.items()
                    if k in ("chat", "hits", "n", "days", "posts", "proposed", "executed")
                },
            }
            if result.meta.get("action_id"):
                entry["action_id"] = result.meta["action_id"]
            calls_log.append(entry)
            if call.name not in W.WRITE_TOOL_NAMES:  # yozish tool'i o'z yozuvini o'zi qo'shgan
                session.add(
                    AgentAction(
                        run_id=run.id,
                        tool=call.name,
                        args=call.arguments,
                        status=ActionStatus.EXECUTED if result.ok else ActionStatus.FAILED,
                        error=None if result.ok else result.text[:500],
                    )
                )
        # javob berilmagan chaqiruvlar (limitdan tashqari) — modelga aytamiz
        for call in res.tool_calls[MAX_CALLS_PER_TURN:]:
            messages.append(
                Msg.tool_result(
                    call.id,
                    call.name,
                    T.envelope(
                        call.name,
                        T.ToolResult("error: too many tool calls in one turn — skipped", ok=False),
                    ),
                )
            )
        if used_result_tokens >= result_budget:
            messages.append(
                Msg.user(
                    "Tool result budget is exhausted. Answer now with the data you already have."
                )
            )

    if not final_text:
        # sikl tool bilan tugadi — tool'siz yakuniy chaqiruv
        messages.append(Msg.user("Answer now with the data you already have; do not call tools."))
        res = await client.chat(Task.TOOLS, messages, tools=None, max_tokens=FINAL_MAX_TOKENS)
        tokens_in += res.usage.tokens_in
        tokens_out += res.usage.tokens_out
        model, provider, finish = res.model, res.provider, res.finish_reason
        final_text = res.text.strip()
        iterations += 1

    run.model = model
    run.tokens_in = tokens_in
    run.tokens_out = tokens_out
    run.finished_at = datetime.now(UTC)
    await session.flush()

    outcome = AgentOutcome(
        text=final_text,
        model=model,
        provider=provider,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        iterations=iterations,
        tool_calls=calls_log,
        finish_reason=finish,
        latency_ms=int((time.monotonic() - started) * 1000),
        run_id=run.id,
        result_tokens=used_result_tokens,
        action_ids=action_ids,
    )
    log.info(
        "agent.done",
        run_id=run.id,
        iterations=iterations,
        tools=[c["tool"] for c in calls_log],
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        result_tokens=used_result_tokens,
        latency_ms=outcome.latency_ms,
    )
    return outcome
