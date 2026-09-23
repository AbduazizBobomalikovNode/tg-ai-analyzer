"""Read-only agent sikli (5-bosqich): LLM tool'larni chaqiradi, biz bajaramiz, audit yozamiz.

Xarajat nazorati:
  * `AGENT_MAX_ITERATIONS` (default 5) va har iteratsiyada ≤ `MAX_CALLS_PER_TURN` tool;
  * **yugurish bo'yicha** bajarilgan chaqiruvlar qopqog'i `AGENT_MAX_EXECUTED_CALLS`
    va narx qopqog'i `AGENT_MAX_COST_USD` — yetilganda tool'lar o'chiriladi va
    "bor narsa bilan javob ber" turn'i qo'shiladi (`stop_reason`);
  * tool natijalari umumiy byudjeti `AGENT_TOOL_RESULT_TOKENS` — oshsa keskin kesiladi;
  * tarix qisqartirilgan (`chat_service.build_history`), system prompt barqaror (kesh);
  * oxirgi iteratsiyada tool'siz "javob ber" chaqiruvi — sikl hech qachon javobsiz tugamaydi.

Xavfsizlik: faqat `tools.READ_TOOLS`; har natija `<untrusted_data>`; har chaqiruv
`agent_actions` ga (append-only) `executed`/`failed` bilan tushadi.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import ActionStatus, AgentAction, AgentRun, Chat
from app.llm import LLM, LLMError, Msg, Role, Task
from app.llm.pricing import estimate_cost
from app.logging import get_logger
from app.observability import TOOL_CALLS
from app.services import tool_cache as TC
from app.services import tools as T
from app.services import write_tools as W
from app.services.prompts import AGENT_SYSTEM_PROMPT, runtime_note
from app.services.search import est_tokens
from app.services.textnorm import uz_fold

log = get_logger(__name__)

MAX_CALLS_PER_TURN = 4
FINAL_MAX_TOKENS = 3000
# Bitta tool bir yugurishda shuncha marta chaqirilishi mumkin. O'lchov ko'rsatdi:
# model qoniqmasa `search_messages` ni boshqa so'zlar bilan 4 martagacha
# takrorlaydi va so'rov narxini ikki barobar oshiradi. Prompt ("adjust once")
# buni to'xtatmadi — shuning uchun kod darajasidagi qopqoq.
MAX_CALLS_PER_TOOL = 3
# Bitta yugurishda bajariladigan (to'silmagan) chaqiruvlar soni — `MAX_CALLS_PER_TURN`
# va `MAX_CALLS_PER_TOOL` dan farqli o'laroq **butun yugurish** bo'yicha. Jonli
# o'lchovda (24 turn, $2.71) aynan shu qopqoq yo'qligi narxni belgilagan edi:
# model iteratsiya shiftigacha yugurib, bitta savolga 6-11 chaqiruv sarflardi.
# `AGENT_MAX_EXECUTED_CALLS` bilan bekor qilinadi.
MAX_EXECUTED_CALLS_PER_RUN = 6
# Byudjet tugaganda qo'shiladigan **bitta** user turn'i (oxirgi majburiy
# javob bilan bir xil matn — model uchun bir xil signal).
FINAL_NUDGE = "Answer now with the data you already have; do not call tools."


def fold_query(text: Any) -> str:
    """Qidiruv so'rovini taqqoslash shakliga keltiradi (registr/bo'shliq/tinish)."""
    raw = str(text or "").strip().lower()
    return " ".join("".join(ch if ch.isalnum() or ch.isspace() else " " for ch in raw).split())


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
    calls_executed: int = 0  # to'silmagan (haqiqatan bajarilgan) chaqiruvlar
    cache_hits: int = 0  # suhbat keshidan olingan tool natijalari (`tool_cache`)
    # `answered` | `call_budget` | `cost_budget` | `iterations`
    stop_reason: str = ""
    #: Takror-qo'riqchisi ishladimi: keshdan kelgan qidiruvdan keyin kamida
    #: bitta server qidiruvi to'sildi (eval `econ-01`).
    repeat_guard: bool = False
    cost_usd: float | None = None  # taxminiy yig'ma narx (pricing.estimate_cost)


async def _mark_failed(
    session: AsyncSession, run: AgentRun, exc: Exception, *, tokens_in: int, tokens_out: int
) -> None:
    """LLM uzilganda run yarim bo'sh qolmasin — xato va tugash vaqti yoziladi."""
    run.error = f"{type(exc).__name__}: {str(exc)[:400]}"[:500]
    run.tokens_in = tokens_in
    run.tokens_out = tokens_out
    run.finished_at = datetime.now(UTC)
    await session.flush()


async def _notify_tool(
    on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None, tool: str
) -> None:
    """UI hodisasi — hech qachon javobni yiqitmaydi (oqim uzilgan bo'lishi mumkin)."""
    if on_event is None:
        return
    try:
        await on_event("status", {"stage": "tool", "tool": tool})
    except Exception as exc:  # klient ketdi / navbat yopildi
        log.debug("agent.event_failed", tool=tool, error=str(exc)[:200])


#: Telegram **serveriga** boradigan matn qidiruvlari. Lokal qidiruv o'sha
#: so'zlar bilan allaqachon topgan bo'lsa, ularni serverda takrorlash yangi
#: ma'lumot bermaydi — faqat MTProto kvotasi va token sarflaydi (o'lchovda
#: `search_messages` uch marta, keyin `search_telegram` uch marta naqshi).
SERVER_SEARCH_TOOLS = frozenset({"search_telegram", "search_global"})

#: Takror-qo'riqchisi (eval `econ-01`) uchun ikki ro'yxat.
#:
#: `CACHEABLE_SEARCH_TOOLS` — natijasi keshdan kelsa "bu savol allaqachon
#: berilgan" degan signal beradigan qidiruvlar. `GUARDED_SERVER_TOOLS` — shu
#: signaldan keyin **to'siladigan** chaqiruvlar: hammasi Telegram serveriga
#: boradi, ya'ni MTProto kvotasi + haqiqiy pul. Lokal `search_messages` ro'yxatda
#: yo'q: u bepul va keshdan kelishi mumkin.
CACHEABLE_SEARCH_TOOLS = frozenset(
    {"search_messages", "search_telegram", "search_global", "search_calendar"}
)
GUARDED_SERVER_TOOLS = frozenset(
    {"search_telegram", "search_global", "search_calendar", "discover_channels"}
)
#: Model uchun qisqa matn — nima bo'lganini va nima qilish kerakligini aytadi.
REPEAT_GUARD_NOTE = (
    "note: repeat detected — server search is disabled for this turn; "
    "answer from the cached results you already have"
)
#: "Chuqurroq / yana qidir" — foydalanuvchi **aniq** yangi qidiruv so'ragan
#: holat. Faqat shunda qo'riqchi o'chiriladi. Ro'yxat ataylab qisqa: noto'g'ri
#: ijobiy (guard ishlamay qolishi) eski xatti-harakat, noto'g'ri salbiy esa
#: foydalanuvchining haqiqiy so'rovini to'sib qo'yardi.
_DEEPER_INTENT = re.compile(
    r"chuqurroq|batafsilroq|yana\s+qidir|qayta\s+qidir|yangidan\s+qidir|tekshirib\s+ko"
    r"|boshqa\s+(?:so|usul|joy|kalit|variant)"
    r"|глубже|подробнее|ещ[её]\s+раз|ещ[её]\s+поищ|переищ|перепровер|поищи\s+ещ[её]"
    r"|\bdeeper\b|\bsearch\s+again\b|\btry\s+again\b|\blook\s+again\b|\bre-?check\b"
    r"|\bmore\s+(?:detail|thorough|sources)",
    re.IGNORECASE,
)


def wants_deeper_search(text: str) -> bool:
    """Foydalanuvchi aniq "chuqurroq/yana qidir" deganmi (takror-qo'riqchi istisnosi)."""
    return bool(_DEEPER_INTENT.search(text or ""))


#: Takror-**turn** aniqlash (eval `econ-01`, turn 2): yangi savol shu suhbatda
#: allaqachon berilganmi — modelning tool tanlashiga qarab emas, kirishdan
#: **oldin** kodda hal qilinadi. Oxirgi shuncha user turn tekshiriladi.
REPEAT_TURN_HISTORY = 6
#: Fold qilingan token to'plamlari Jaccard o'xshashligi shu chegaradan yuqori
#: bo'lsa — "aynan o'sha savol" deymiz (kichik imlo/so'z tartibi farqi bilan).
REPEAT_TURN_JACCARD_MIN = 0.9
#: Bundan kam fold-token bo'lgan savol hech qachon repeat deb hisoblanmaydi —
#: qisqa savollar ("va?", ".") tasodifan mos kelishi mumkin (soxta pozitiv
#: xavfi token sonidan yuqori bo'ladi).
REPEAT_TURN_MIN_TOKENS = 3
#: Model uchun runtime_note'ga qo'shiladigan qator (o'zbekcha — user tilidan
#: mustaqil, chunki bu ko'rsatma modelga, foydalanuvchiga emas).
REPEAT_TURN_NOTE = (
    "Bu savol shu suhbatda allaqachon berilgan va javob berilgan — avvalgi javobni "
    "qisqacha takrorla/yangila; ko'pi bilan bitta lokal qidiruv (search_messages) "
    "ruxsat, server qidiruvi yo'q."
)


def _repeat_fold(text: str) -> str:
    """`uz_fold` (kirill→lotin, apostrofsiz) + `fold_query` (tinish olib tashlash)."""
    return fold_query(uz_fold(text))


def is_repeat_turn(question: str, history: list[Msg]) -> bool:
    """Yangi savol so'nggi `REPEAT_TURN_HISTORY` user turn'idan biriga mos keladimi.

    Aniq moslik yoki fold qilingan token to'plamlari Jaccard >= `REPEAT_TURN_JACCARD_MIN`
    — va foydalanuvchi aniq "chuqurroq qidir" demagan bo'lsa
    (`wants_deeper_search`, takror-qo'riqchi istisnosi shu yerda ham amal qiladi).
    Juda qisqa savollar hech qachon repeat emas — `REPEAT_TURN_MIN_TOKENS`.
    """
    if wants_deeper_search(question):
        return False
    q_folded = _repeat_fold(question)
    q_tokens = set(q_folded.split())
    if len(q_tokens) < REPEAT_TURN_MIN_TOKENS:
        return False
    prior_user_turns = [m.content for m in history if m.role == Role.USER]
    for prior in prior_user_turns[-REPEAT_TURN_HISTORY:]:
        p_folded = _repeat_fold(prior)
        if not p_folded:
            continue
        if p_folded == q_folded:
            return True
        p_tokens = set(p_folded.split())
        if not p_tokens:
            continue
        union = q_tokens | p_tokens
        if union and len(q_tokens & p_tokens) / len(union) >= REPEAT_TURN_JACCARD_MIN:
            return True
    return False


def _blocked_reresearch(call: Any, local_queries: dict[str, int]) -> T.ToolResult | None:
    """Aynan shu so'zlar lokal qidiruvda topilgan bo'lsa — serverga chiqarmaymiz.

    Lokal qidiruv **nol** bergan bo'lsa server yo'li ochiq qoladi: aynan shu
    holatda u haqiqatan yangi ma'lumot beradi (sinxronlanmagan tarix).
    """
    if call.name not in SERVER_SEARCH_TOOLS:
        return None
    folded = fold_query(call.arguments.get("query"))
    hits = local_queries.get(folded, 0) if folded else 0
    if hits <= 0:
        return None
    return T.ToolResult(
        f"note: local search already found {hits} hits for this query; "
        "refine the wording or answer with what you have",
        ok=False,
    )


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
    on_event: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    conversation_id: int | None = None,
) -> AgentOutcome:
    """`on_event` — UI oqimi uchun ixtiyoriy kuzatuvchi (`chat_service.EventSink`).

    Har **bajariladigan** tool oldidan `("status", {"stage": "tool", "tool": nom})`
    beriladi. To'silgan chaqiruv (takror/limit) hodisa bermaydi: u bajarilmaydi,
    ya'ni foydalanuvchiga "hozir shu tool ishlayapti" deyish yolg'on bo'lardi
    (`calls_log` esa uni baribir yozadi — CLAUDE.md, "To'silgan chaqiruv").
    Kuzatuvchidagi xato javobni **buzmaydi**: u faqat ko'rsatish uchun.

    `conversation_id` berilsa o'qish tool'larining natijalari suhbat keshiga
    (`services/tool_cache.py`, 10 daqiqa) tushadi va bir xil savol qaytarilganda
    qidiruv **ikkinchi marta yugurmaydi** (eval `econ-01`). Kesh bo'lmasa
    (Redis yo'q, `conversation_id` yo'q) hech narsa o'zgarmaydi.

    `question` `history` ichidagi so'nggi user turn'lardan biriga mos kelsa
    (`is_repeat_turn`) — bu **kodda**, model hech narsa chaqirmasdan turib
    aniqlanadi: `runtime_note` ogohlantirish qatori oladi, server qidiruv
    tool'lari birinchi iteratsiyadanoq to'siladi va bu yugurishning bajariladigan
    chaqiruvlar qopqog'i 2 taga tushadi (eval `econ-01`, turn 2).
    """
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
    # Takror-turn aniqlash **tool chaqirilishidan oldin**, kodda: shu savol
    # suhbatning so'nggi user turn'lariga mos keladimi (eval `econ-01`, turn 2).
    # Model tool tanlovidan mustaqil — cache-hit asosidagi qo'riqchi (pastda,
    # `repeat_seen`) faqat *keyin*, natijaga qarab ishlaydi.
    deeper_requested = wants_deeper_search(question)
    repeat_turn = is_repeat_turn(question, history)

    # Chat ro'yxati user turn'ida beriladi — system'da emas (kesh buzilmasin).
    # Shu tufayli agent `list_chats` chaqirmasdan chat nomini aniqlaydi.
    note = runtime_note(
        now_iso=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        locale=locale,
        pinned_chat=pinned_chat.title if pinned_chat else None,
        chats=await T.chat_index(
            session,
            account_id,
            pinned_chat_id=pinned_chat.id if pinned_chat else None,
            limit=T.CHAT_INDEX_LIMIT,
            compact=True,
        ),
    )
    if repeat_turn:
        note = f"{note}\n\n{REPEAT_TURN_NOTE}"
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
    seen_calls: set[tuple[str, str]] = set()  # (tool, args) — takrorni to'sish
    per_tool: Counter[str] = Counter()
    calls_log: list[dict[str, Any]] = []
    # Lokal qidiruv natijalari: `fold_query(query)` → topilma soni. Server
    # qidiruvi aynan shu so'zlar bilan takrorlanmasin (4-band, o'lchov).
    local_queries: dict[str, int] = {}
    model = provider = ""
    final_text = ""
    finish = ""
    iterations = 0
    calls_executed = 0
    cache_hits = 0
    stop_reason = ""
    # Takror-qo'riqchisi: keshdan qidiruv natijasi kelgan bo'lsa (ya'ni bu savol
    # shu suhbatda allaqachon berilgan) server qidiruvlari shu yugurishda
    # yopiladi — model boshqa so'zlar bilan qayta urinmasin (eval `econ-01`:
    # `cache_hits=2`, keyin ikkita qayta yozilgan `search_telegram`, $0.086).
    # `repeat_turn` yuqorida allaqachon aniqlangan bo'lsa — qo'riqchi **birinchi**
    # iteratsiyadanoq ishlaydi, kesh urilishini kutmaydi.
    repeat_seen = repeat_turn
    repeat_guard = repeat_turn
    tools_off = False  # byudjet tugadi — keyingi chaqiruv tool'siz
    cost_usd: float | None = None
    call_cap = max(1, int(s.agent_max_executed_calls or MAX_EXECUTED_CALLS_PER_RUN))
    if repeat_turn:
        # Takroriy turn — ko'pi bilan ikkita bajarilgan chaqiruv (bitta
        # tasdiqlovchi lokal qidiruv + bitta qo'shimcha), server qidiruvi yo'q.
        call_cap = min(call_cap, 2)
    cost_cap = float(s.agent_max_cost_usd or 0.0)

    for it in range(1, max_iter + 1):
        iterations = it
        last_round = it == max_iter
        allow_tools = not last_round and not tools_off
        try:
            res = await client.chat(
                Task.TOOLS,
                messages,
                tools=specs if allow_tools else None,
                max_tokens=FINAL_MAX_TOKENS,
            )
        except LLMError as exc:
            log.warning("agent.llm_failed", run_id=run.id, iteration=it, error=str(exc)[:200])
            await _mark_failed(session, run, exc, tokens_in=tokens_in, tokens_out=tokens_out)
            raise
        tokens_in += res.usage.tokens_in
        tokens_out += res.usage.tokens_out
        model, provider, finish = res.model, res.provider, res.finish_reason
        cost_usd = estimate_cost(provider, model, tokens_in, tokens_out)

        if not res.tool_calls or not allow_tools:
            # tool'siz raundda tool so'ralsa ham bajarilmaydi — matn yo'q bo'lsa majburiy javob
            final_text = res.text.strip() if not res.tool_calls else ""
            if not stop_reason:
                stop_reason = "answered" if not res.tool_calls and not last_round else "iterations"
            break

        # `res.tool_calls` ichidagi `signature` (Gemini 3 thought signature) shu
        # yerdan keyingi so'rovga o'tadi — aks holda ikkinchi iteratsiya
        # `400 INVALID_ARGUMENT … missing a thought_signature` bilan yiqiladi.
        messages.append(Msg.assistant(res.text, res.tool_calls, signature=res.signature))
        for call in res.tool_calls[:MAX_CALLS_PER_TURN]:
            t0 = time.monotonic()
            cached_text: str | None = None
            guard_blocked = False
            # ── takroriy chaqiruvlar qopqog'i ──────────────────────────────
            # Aynan bir xil (tool, args) — natija o'zgarmaydi, qayta bajarish
            # DB/MTProto ishini ham, tokenni ham behuda sarflaydi.
            sig = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
            if sig in seen_calls:
                result = T.ToolResult(
                    "note: identical call already made in this run — use that result", ok=False
                )
            elif calls_executed >= call_cap:
                # Yugurish byudjeti shu turn ichida tugadi: qolgan chaqiruvlar
                # **bajarilmaydi** (bloklangan chaqiruv ham `calls_log` ga tushadi).
                result = T.ToolResult(
                    f"note: this run's budget of {call_cap} tool calls is used up; "
                    "answer now with what you have and say what is missing",
                    ok=False,
                )
            elif repeat_seen and not deeper_requested and call.name in GUARDED_SERVER_TOOLS:
                # Bu savol shu suhbatda allaqachon berilgan (kesh buni isbotladi).
                # Boshqa so'z bilan serverga chiqish yangi ma'lumot bermaydi —
                # faqat MTProto kvotasi va pul. Lokal `search_messages` ochiq.
                guard_blocked = True
                repeat_guard = True
                result = T.ToolResult(REPEAT_GUARD_NOTE, ok=False)
            elif (blocked := _blocked_reresearch(call, local_queries)) is not None:
                result = blocked
            elif per_tool[call.name] >= MAX_CALLS_PER_TOOL:
                result = T.ToolResult(
                    f"note: {call.name} was already called {MAX_CALLS_PER_TOOL} times; "
                    "answer with what you have or try a different tool",
                    ok=False,
                )
            elif call.name in W.WRITE_TOOL_NAMES:
                # Yozish tool'i ham **bir xil** qopqoq ostida: ilgari hisob
                # faqat o'qish shoxida oshardi, ya'ni model bitta raundda
                # cheksiz taklif yasay olardi (har biri DB qatori + tasdiq
                # so'rovi). `seen_calls` faqat aynan bir xil argumentni to'sadi.
                per_tool[call.name] += 1
                calls_executed += 1
                await _notify_tool(on_event, call.name)
                if not write_enabled:
                    result = T.ToolResult("error: writing is disabled for this account", ok=False)
                else:
                    result = await W.propose_or_execute(
                        session, ctx, run_id=run.id, tool=call.name, args=call.arguments
                    )
                if result.meta.get("action_id"):
                    action_ids.append(int(result.meta["action_id"]))
            else:
                per_tool[call.name] += 1
                calls_executed += 1
                await _notify_tool(on_event, call.name)
                # Suhbat keshi tool'dan **oldin**: bir xil (tool, argument)
                # 10 daqiqa ichida ikkinchi marta bajarilmaydi. Byudjet
                # baribir sanaladi — aks holda kesh cheksiz sikl ochib
                # qo'yardi (model bir xil chaqiruvni qayta-qayta so'rasa).
                ckey = TC.cache_key(conversation_id, call.name, call.arguments)
                cached_text = await TC.get(ckey) if ckey else None
                if cached_text is not None:
                    cache_hits += 1
                    if call.name in CACHEABLE_SEARCH_TOOLS:
                        repeat_seen = True
                    result = T.ToolResult(f"{cached_text}\n{TC.CACHED_NOTE}", ok=True)
                else:
                    result = await T.run_tool(call.name, ctx, call.arguments)
                    if ckey and result.ok:
                        await TC.put(ckey, result.text)
                if call.name == "search_messages":
                    folded = fold_query(call.arguments.get("query"))
                    if folded:
                        hits = int(result.meta.get("hits") or 0)
                        local_queries[folded] = max(local_queries.get(folded, 0), hits)
            seen_calls.add(sig)
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
                **({"cached": True} if cached_text is not None else {}),
                # To'silgan chaqiruv ham `calls_log` ga tushadi (nima so'ralganini
                # ko'rish uchun); sabab bilan — eval buni bajarilgandan ajratadi.
                **({"blocked": "repeat_guard"} if guard_blocked else {}),
                **{
                    k: v
                    for k, v in result.meta.items()
                    if k in ("chat", "hits", "n", "days", "posts", "proposed", "executed")
                },
            }
            if result.meta.get("action_id"):
                entry["action_id"] = result.meta["action_id"]
            calls_log.append(entry)
            TOOL_CALLS.labels(call.name, str(result.ok).lower()).inc()
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
        # ── yugurish byudjetlari: chaqiruvlar soni va taxminiy narx ──────────
        if not tools_off and calls_executed >= call_cap:
            tools_off, stop_reason = True, "call_budget"
            messages.append(Msg.user(FINAL_NUDGE))
        elif not tools_off and cost_cap and (cost_usd or 0.0) > cost_cap:
            tools_off, stop_reason = True, "cost_budget"
            log.info(
                "agent.cost_cap", run_id=run.id, cost_usd=round(cost_usd or 0.0, 4), cap=cost_cap
            )
            messages.append(Msg.user(FINAL_NUDGE))

    if not final_text:
        # sikl tool bilan tugadi — tool'siz yakuniy chaqiruv
        messages.append(Msg.user(FINAL_NUDGE))
        try:
            res = await client.chat(Task.TOOLS, messages, tools=None, max_tokens=FINAL_MAX_TOKENS)
        except LLMError as exc:
            log.warning(
                "agent.llm_failed", run_id=run.id, iteration=iterations, error=str(exc)[:200]
            )
            await _mark_failed(session, run, exc, tokens_in=tokens_in, tokens_out=tokens_out)
            raise
        tokens_in += res.usage.tokens_in
        tokens_out += res.usage.tokens_out
        model, provider, finish = res.model, res.provider, res.finish_reason
        final_text = res.text.strip()
        cost_usd = estimate_cost(provider, model, tokens_in, tokens_out)
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
        calls_executed=calls_executed,
        cache_hits=cache_hits,
        stop_reason=stop_reason or "answered",
        repeat_guard=repeat_guard,
        cost_usd=cost_usd,
    )
    log.info(
        "agent.done",
        run_id=run.id,
        iterations=iterations,
        tools=[c["tool"] for c in calls_log],
        calls_executed=calls_executed,
        cache_hits=cache_hits,
        stop_reason=outcome.stop_reason,
        repeat_guard=repeat_guard,
        cost_usd=cost_usd,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        result_tokens=used_result_tokens,
        latency_ms=outcome.latency_ms,
    )
    return outcome
