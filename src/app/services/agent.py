"""Read-only agent sikli (5-bosqich): LLM tool'larni chaqiradi, biz bajaramiz, audit yozamiz.

Xarajat nazorati:
  * `AGENT_MAX_ITERATIONS` (default 5) va har iteratsiyada ≤ `MAX_CALLS_PER_TURN` tool;
  * **yugurish bo'yicha** bajarilgan chaqiruvlar qopqog'i `AGENT_MAX_EXECUTED_CALLS`
    va narx qopqog'i `AGENT_MAX_COST_USD` — yetilganda tool'lar o'chiriladi va
    "bor narsa bilan javob ber" turn'i qo'shiladi (`stop_reason`);
  * takror-qo'riqchisi (`repeat_guard`) raunddagi yagona chaqiruvni to'ssa — xuddi
    shunday darhol "javob ber" turn'i (`stop_reason="repeat_guard"`);
  * ikki ketma-ket natijasiz raund va yangi tool yo'q — `stop_reason="no_progress"`;
  * savolda nomlangan odam/chat kodda topiladi (`chat_hints`, runtime note'da ≤3 qator);
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
from app.services import untrusted as U
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
    # `answered` | `call_budget` | `cost_budget` | `repeat_guard` | `no_progress` | `iterations`
    stop_reason: str = ""
    #: Takror-qo'riqchisi ishladimi: keshdan kelgan qidiruvdan keyin kamida
    #: bitta server qidiruvi to'sildi (eval `econ-01`).
    repeat_guard: bool = False
    cost_usd: float | None = None  # taxminiy yig'ma narx (pricing.estimate_cost)
    #: Fikrlash (output narxida) va keshdan o'qilgan input tokenlari — yig'ma.
    tokens_thinking: int = 0
    tokens_cached: int = 0


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


def direct_intent_note(intent: str, locale: str) -> str:
    """Qidiruvsiz turn uchun runtime qator (`meta` da imkoniyatlar matni bilan).

    Nega `runtime_note` ichida, system'da emas: system prompt bayt darajasida
    barqaror bo'lishi kerak (kesh prefiksi), bu qator esa faqat darvoza
    ishlagan turn'da qo'shiladi. Imkoniyatlar ro'yxati **statik matndan**
    (`prompts_deep.capabilities_text`) — model o'zida yo'q funksiyani o'ylab
    topmasin.
    """
    from app.services.prompts_deep import capabilities_text

    if intent in ("meta", "meta_cost"):
        return (
            "Bu turn tizimning o'zi haqida: tool chaqirma, quyidagi ro'yxatdan "
            "foydalanuvchi tilida qisqa javob ber va ro'yxatda yo'q imkoniyatni "
            "qo'shma. Oxirida bitta aniq misol savol taklif qil.\n"
            f"<capabilities>\n{capabilities_text(locale)}\n</capabilities>"
        )
    return (
        "Bu turn salomlashuv/minnatdorchilik — tool chaqirma, bir-ikki qatorda "
        "iliq javob ber va nima qidirib bera olishingni qisqa taklif qil."
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


# ─── chat maslahatlari (nomlangan odam/chat → [id:N]) ─────────────────────────
#
# Jonli o'lchov (conv 284, "Otabek bilan oxirgi marta nima gaplashganmiz?"):
# `runtime_note` dagi indeks faqat eng faol `CHAT_INDEX_LIMIT` (25) chatni
# oladi, "Otabek" shaxsiy chatlari unda yo'q edi. Model `list_chats` →
# `search_global` → `get_user_info` → `list_members` x2 bilan 7 iteratsiya
# aylandi ($0.105, 94k in) va noto'g'ri javob berdi ("shaxsiy chat topilmadi").
# Shuning uchun moslik **akkauntning hamma** sinxronlangan chati bo'yicha
# kodda qidiriladi (bepul, deterministik) va ≤3 ta aniq maslahat qatori
# qo'shiladi — model chatni o'zi izlamaydi.

#: Qo'shiladigan maslahatlar soni (runtime_note har iteratsiyada qayta ketadi).
CHAT_HINT_MAX = 3
#: Savol so'zi va sarlavha so'zi o'xshashligi (`difflib`, uz_fold dan keyin).
CHAT_HINT_MIN_RATIO = 0.8
#: Shundan qisqa so'zlar solishtirilmaydi ("IT", "va" — tasodifiy moslik).
CHAT_HINT_MIN_CHARS = 3
#: Maslahat uchun o'qiladigan chatlar qopqog'i (akkaunt bo'yicha).
CHAT_HINT_SCAN_MAX = 2_000
_HINT_WORD_RE = re.compile(r"[a-z0-9]+")
#: Qo'shimchalar: "Otabekning" / "Otabekka" / "Otabekdan" → "otabek".
#: Kirill uz_fold'da lotinga o'tadi ("Отабеком" → "otabekom" — o'xshashlik yetadi).
_HINT_SUFFIXES = (
    "lardan",
    "larga",
    "larni",
    "ning",
    "dagi",
    "niki",
    "dan",
    "lar",
    "ga",
    "ka",
    "qa",
    "da",
    "ni",
)
#: Sarlavhada ham, savolda ham ko'p uchraydigan umumiy so'zlar — ular bo'yicha
#: moslik "shu chat" degani emas ("IT chat", "ochiq kanal", "bu hafta").
_HINT_STOPWORDS = frozenset(
    """
    chat chats chati chatda chatlar guruh guruhi gruppa group groups kanal kanali channel
    channels ochiq rasmiy official news yangiliklar yangilik uzbekistan ozbekiston toshkent
    tashkent bilan oxirgi marta nima nimalar haqida qachon kim kimlar bugun kecha hafta oy yil
    bu shu men meni menga mening biz bizning siz sizning uchun va yoki ham emas bor yoq
    gaplashganmiz gaplashdik yozgan yozdi dedi degan aytdi
    shuni shunga shuning shundan shunda buni bunga buning bundan bunda uni unga uning undan
    ularni ularga ularning
    the and for with from what when who last about this that
    chto kak kogda kto eto dlya ili posledniy poslednie
    """.split()
)


@dataclass(frozen=True, slots=True)
class ChatHint:
    chat_id: int
    title: str
    kind: str
    msgs: int
    last: datetime | None
    score: float

    def line(self) -> str:
        last = f" last={self.last.strftime('%Y-%m-%d')}" if self.last else ""
        return (
            f'chat_hint: "{self.title}" [id:{self.chat_id}] {self.kind} msgs={self.msgs}{last} '
            f'— use get_recent_messages / search_messages with chat="{self.chat_id}"'
        )


def _hint_words(text: str) -> list[str]:
    return _HINT_WORD_RE.findall(uz_fold(text or "").lower())


def _word_forms(word: str) -> set[str]:
    forms = {word}
    for suffix in _HINT_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= CHAT_HINT_MIN_CHARS:
            forms.add(word[: -len(suffix)])
    return forms


def _similar(a: str, b: str) -> float:
    if a == b:
        return 1.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def chat_hints(question: str, chats: list[Any], *, limit: int = CHAT_HINT_MAX) -> list[ChatHint]:
    """Savoldagi nom/sarlavha bilan mos keladigan chatlar — **deterministik**, LLM'siz.

    Moslik: savol so'zi (qo'shimchasi olib tashlangan shakllari bilan) va
    sarlavhaning "o'ziga xos" so'zi (umumiy so'z emas, ≥3 belgi) `uz_fold`
    dan keyin `difflib` o'xshashligi ≥ `CHAT_HINT_MIN_RATIO`; `@username`
    ham xuddi shunday. Tartib: sarlavha so'zlarining qancha qismi mos kelgani
    (to'liq nom birinchi), keyin shaxsiy chat, oxirgi faollik, xabarlar soni.
    Chat nomi begona matn — qator `untrusted.attr` dan o'tadi.
    """
    q_forms: set[str] = set()
    for word in _hint_words(question):
        if len(word) < CHAT_HINT_MIN_CHARS or word in _HINT_STOPWORDS:
            continue
        q_forms |= _word_forms(word)
    if not q_forms:
        return []
    scored: list[tuple[tuple[Any, ...], ChatHint]] = []
    for c in chats:
        title = str(getattr(c, "title", "") or "")
        words = [
            w
            for w in dict.fromkeys(_hint_words(title))
            if len(w) >= CHAT_HINT_MIN_CHARS and w not in _HINT_STOPWORDS
        ]
        uname = (getattr(c, "username", "") or "").lower()
        matched = sum(
            1 for w in words if max(_similar(f, w) for f in q_forms) >= CHAT_HINT_MIN_RATIO
        )
        by_user = bool(uname) and (max(_similar(f, uname) for f in q_forms) >= CHAT_HINT_MIN_RATIO)
        if not matched and not by_user:
            continue
        frac = 1.0 if by_user else (matched / len(words) if words else 0.0)
        kind = str(getattr(getattr(c, "type", ""), "value", getattr(c, "type", "")) or "")
        last = getattr(c, "last_message_at", None)
        msgs = int(getattr(c, "synced_total", 0) or 0)
        hint = ChatHint(
            chat_id=int(c.id),
            title=U.attr(title) or f"chat {c.id}",
            kind=kind,
            msgs=msgs,
            last=last if isinstance(last, datetime) else None,
            score=round(frac, 3),
        )
        key = (
            -frac,
            0 if kind == "private" else 1,
            -(last.timestamp() if isinstance(last, datetime) else 0.0),
            -msgs,
            int(c.id),
        )
        scored.append((key, hint))
    scored.sort(key=lambda p: p[0])
    return [h for _k, h in scored[: max(0, limit)]]


async def load_chat_hints(session: AsyncSession, account_id: int, question: str) -> list[ChatHint]:
    """Akkauntning **hamma** sinxronlangan chati bo'yicha `chat_hints` (bitta so'rov)."""
    from sqlalchemy import select

    rows = await session.execute(
        select(Chat)
        .where(Chat.account_id == account_id, Chat.synced_total > 0)
        .order_by(Chat.last_message_at.desc().nulls_last())
        .limit(CHAT_HINT_SCAN_MAX)
    )
    return chat_hints(question, list(rows.scalars().all()))


# ─── "oldinga siljish yo'q" to'xtashi ─────────────────────────────────────────
#
# Ikki ketma-ket raundda bajarilgan chaqiruvlar hech narsa bermagan (xato /
# bo'sh / faqat keshdan) va model yangi tool ham sinamayapti — uchinchi raund
# natijani o'zgartirmaydi, faqat prefiksni (~8k token) yana bir marta yeydi.
NO_PROGRESS_ROUNDS = 2


def unproductive(result: T.ToolResult, *, cached: bool) -> bool:
    """Chaqiruv yangi ma'lumot bermadi: xato, to'silgan, bo'sh yoki keshdan."""
    if cached or not result.ok:
        return True
    meta = result.meta or {}
    for key in ("hits", "n"):
        if key in meta and not meta.get(key):
            return True
    head = (result.text or "").lstrip().lower()
    return head.startswith(("no ", "note:", "error"))


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
    # Salom / "nimalar qila olasan" — tool kerak emas. Darvoza deep quvuridagi
    # bilan **bitta** (`deep_search.direct_intent`): rejim qopqoq, buyruq emas.
    from app.services.deep_search import direct_intent

    gate_intent = direct_intent(question)
    # Savolda nomlangan odam/chat — indeksga sig'magan bo'lsa ham kodda topiladi.
    hints = [] if gate_intent else await load_chat_hints(session, account_id, question)
    if hints:
        note = f"{note}\n\n" + "\n".join(h.line() for h in hints)
    if gate_intent:
        note = f"{note}\n\n{direct_intent_note(gate_intent, locale)}"
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
    tokens_thinking = tokens_cached = 0
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
    # "Oldinga siljish yo'q": ketma-ket natijasiz raundlar va sinalgan tool nomlari.
    idle_rounds = 0
    tried_tools: set[str] = set()
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
        tokens_thinking += res.usage.thinking_tokens
        tokens_cached += res.usage.cached_tokens
        model, provider, finish = res.model, res.provider, res.finish_reason
        cost_usd = estimate_cost(
            provider,
            model,
            tokens_in,
            tokens_out,
            tokens_thinking=tokens_thinking,
            tokens_cached=tokens_cached,
        )

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
        executed_before = calls_executed
        guard_hit = False
        round_progress = False
        round_names = {c.name for c in res.tool_calls[:MAX_CALLS_PER_TURN]}
        new_tool_tried = bool(round_names - tried_tools)
        tried_tools |= round_names
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
                guard_hit = True
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
            if not unproductive(result, cached=cached_text is not None):
                round_progress = True
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
        # ── takror-qo'riqchisi: to'silgan chaqiruv = darhol javob ────────────
        # Shu raundda qo'riqchi chaqiruvni to'sgan va **hech narsa bajarilmagan**
        # bo'lsa, model yangi ma'lumot olmaydi — keyingi iteratsiyalar faqat
        # to'silgan chaqiruvlarni qayta yozib, shiftgacha yeyardi (o'lchov: 7 iter,
        # $0.10+ bitta takroriy savolga). Byudjet tugagandagi bilan bir xil yo'l:
        # tool'lar o'chadi va bitta "javob ber" turn'i.
        if not tools_off and guard_hit and calls_executed == executed_before:
            tools_off, stop_reason = True, "repeat_guard"
            log.info("agent.repeat_guard_stop", run_id=run.id, iteration=it)
            messages.append(Msg.user(FINAL_NUDGE))
        # ── oldinga siljish yo'q: ikki raund bo'sh, yangi tool ham yo'q ─────────
        # Qopqoqlar (`call_cap`, `cost_cap`, iteratsiya) o'zgarmaydi — bu faqat
        # ulardan **oldin** to'xtaydi: uchinchi bo'sh raund javobni o'zgartirmaydi.
        idle_rounds = 0 if round_progress else idle_rounds + 1
        if not tools_off and idle_rounds >= NO_PROGRESS_ROUNDS and not new_tool_tried:
            tools_off, stop_reason = True, "no_progress"
            log.info("agent.no_progress_stop", run_id=run.id, iteration=it)
            messages.append(Msg.user(FINAL_NUDGE))
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
        tokens_thinking += res.usage.thinking_tokens
        tokens_cached += res.usage.cached_tokens
        model, provider, finish = res.model, res.provider, res.finish_reason
        final_text = res.text.strip()
        cost_usd = estimate_cost(
            provider,
            model,
            tokens_in,
            tokens_out,
            tokens_thinking=tokens_thinking,
            tokens_cached=tokens_cached,
        )
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
        tokens_thinking=tokens_thinking,
        tokens_cached=tokens_cached,
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
        tokens_thinking=tokens_thinking,
        tokens_cached=tokens_cached,
        result_tokens=used_result_tokens,
        latency_ms=outcome.latency_ms,
    )
    return outcome
