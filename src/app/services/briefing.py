"""Shaxsiy kunlik briefing (L3) — kross-chat, bitta `Task.SEARCH` (arxitektura v2, §9).

Nima uchun: kun digestlari chat bo'yicha; odam esa **kuni** bo'yicha o'qiydi.
L3 kun tugunlarini (`retrieval_nodes`, level 2) bitta konvertga yig'adi va
bitta sintez chaqiruvi bilan "ertalabki gazeta" yasaydi.

Uch qaror bu modulni boshqarib turadi:

1. **Salience kodda, sintez modelda.** Qaysi chat muhimligi — statistika
   (`msg_count` ning 28 kunlik medianadan z-og'ishi), egasining mention
   qilinishi (+2) va ochiq savollar (+1 har biri). Model tartib tanlamaydi:
   u faqat allaqachon tanlangan chatlar haqida yozadi. Top
   `BRIEFING_TOP_CHATS` to'liq, qolgani bir qatordan.
2. **Sitatalar mexanik tekshiriladi.** Model qaytargan har `#<id>` manba
   tugunlari matnidagi id'lar to'plamida bo'lishi shart; bo'lmasa havola
   o'chiriladi, havolasiz qolgan qator esa **tashlanadi**. "Hallyutsinatsiya
   qilingan sitata" chiqishga yeta olmaydi.
3. **Kim ko'p yozgani — kodda sanaladi**, modeldan so'ralmaydi: son
   generatsiya qilinadigan narsa emas.

Xavfsizlik (invariant 3): kun tugunlari matni — begona odamlar yozgani.
U **bitta** `<untrusted_data kind="digests">` konvertida, `untrusted.neutralize`
dan o'tgan holda beriladi. Chain-of-Density raundlari ham shu konvertdan
foydalanadi (arzon `Task.ROUTE`), ya'ni asosiy sintez baribir **bitta**
`Task.SEARCH` chaqiruvi bo'lib qoladi.

Yetkazish — `bot/push.py` orqali **faqat akkaunt egasiga**; MTProto bu yerda
umuman ishlatilmaymi (hech narsa Telegram akkaunt nomidan yuborilmaydi).
"""

from __future__ import annotations

import html
import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (
    Account,
    Briefing,
    Chat,
    ChatDigest,
    Conversation,
    ConversationMessage,
    Message,
    RetrievalNode,
    User,
)
from app.llm import LLM, LLMError, Msg, Task
from app.llm.pricing import estimate_cost
from app.logging import get_logger
from app.services import untrusted as U
from app.services.nodes import LEVEL_DAY
from app.services.prompts_retrieval import BRIEFING_COD_PROMPT, BRIEFING_PROMPT

log = get_logger(__name__)

#: Salience tarixi oynasi (kun) — 28 kunlik mediana (§9).
HISTORY_DAYS = 28
#: Egasining nomi tilga olingani uchun ball.
MENTION_BONUS = 2.0
#: Har bir ochiq savol uchun ball (`chat_digests.structured.open_questions`).
QUESTION_BONUS = 1.0
#: Bitta chat tuguni matnining konvertdagi qopqog'i (narx).
CHAT_TEXT_MAX_CHARS = 3_000
#: Butun konvert qopqog'i.
ENVELOPE_MAX_CHARS = 30_000
#: Sintez javobi (token).
BRIEFING_MAX_TOKENS = 1_400
#: "Kim ko'p yozdi" ro'yxati uzunligi.
TOP_SENDERS = 5
#: `#<id>` sitata regexi — manba matnida ham, javobda ham bir xil.
CITE_RE = re.compile(r"#(\d{1,12})\b")

DEFAULT_TZ = "Asia/Tashkent"


# ─── vaqt / jim soatlar ──────────────────────────────────────────────────────


def zone(tz: str | None) -> ZoneInfo:
    """Nomi buzuq bo'lsa ham yiqilmaydi — `DEFAULT_TZ` ga tushadi, keyin UTC."""
    for name in (tz or DEFAULT_TZ, DEFAULT_TZ, "UTC"):
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            continue
    return ZoneInfo("UTC")


def local_now(tz: str | None, *, now: datetime | None = None) -> datetime:
    base = now or datetime.now(UTC)
    if base.tzinfo is None:
        base = base.replace(tzinfo=UTC)
    return base.astimezone(zone(tz))


def in_quiet_hours(now: time, quiet_from: time | None, quiet_to: time | None) -> bool:
    """Lokal vaqt jim oyna ichidami. Yarim tun orqali o'ralishni qo'llaydi.

    * `22:00 → 07:00` — kechasi (o'ralgan): `now >= 22:00` **yoki** `now < 07:00`.
    * `01:00 → 06:00` — oddiy oyna.
    * chegaralardan biri `None` yoki ikkalasi teng — jim oyna **yo'q**.
    """
    if quiet_from is None or quiet_to is None or quiet_from == quiet_to:
        return False
    if quiet_from < quiet_to:
        return quiet_from <= now < quiet_to
    return now >= quiet_from or now < quiet_to


def edition_day(local: datetime) -> date:
    """Joriy "nashr" kuni — **kechagi** kun.

    Briefing ertalab yetkaziladi va o'tgan kunni qamraydi: 02:00 da bugungi
    kun hali bo'sh, shuning uchun qurish ham, yetkazish ham `local - 1 kun`
    ustida ishlaydi. Shu tufayli qurish va yetkazish cron'lari bir xil
    kalitni ko'radi va `uq_briefing_account_day` idempotentlikni kafolatlaydi.
    """
    return local.date() - timedelta(days=1)


@dataclass(frozen=True, slots=True)
class DeliveryCheck:
    ok: bool
    reason: str
    day: date
    local: datetime


def delivery_check(account: Account, *, now: datetime | None = None) -> DeliveryCheck:
    """Shu akkauntga hozir push qilish mumkinmi (sozlama + soat + jim oyna)."""
    local = local_now(account.tz, now=now)
    day = edition_day(local)
    if not account.briefing_enabled:
        return DeliveryCheck(False, "disabled", day, local)
    if local.hour < int(account.briefing_hour or 0):
        return DeliveryCheck(False, "too_early", day, local)
    if in_quiet_hours(local.timetz().replace(tzinfo=None), account.quiet_from, account.quiet_to):
        return DeliveryCheck(False, "quiet_hours", day, local)
    return DeliveryCheck(True, "ok", day, local)


# ─── salience ────────────────────────────────────────────────────────────────


def z_score(count: int, history: list[int]) -> float:
    """`count` ning tarix **medianasidan** og'ishi, tarqalishga bo'lingan.

    O'rtacha emas, mediana: bitta "portlagan" kun butun bazani siljitmasin.
    Tarqalish sifatida `pstdev` olinadi; u nol bo'lsa (masalan har kuni
    bir xil son) mediana (yoki 1) ishlatiladi — nolga bo'linish yo'q.
    """
    if not history:
        return 0.0
    med = statistics.median(history)
    spread = statistics.pstdev(history) if len(history) > 1 else 0.0
    if spread <= 0:
        spread = float(med) if med > 0 else 1.0
    return round((count - med) / spread, 4)


@dataclass(slots=True)
class ChatSalience:
    chat_id: int
    title: str
    msg_count: int
    z: float = 0.0
    mentioned: bool = False
    open_questions: int = 0
    text: str = ""

    @property
    def score(self) -> float:
        return round(
            self.z
            + (MENTION_BONUS if self.mentioned else 0.0)
            + QUESTION_BONUS * self.open_questions,
            4,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "title": self.title,
            "msg_count": self.msg_count,
            "z": self.z,
            "mentioned": self.mentioned,
            "open_questions": self.open_questions,
            "score": self.score,
        }


def owner_names(user: User | None, account: Account | None) -> list[str]:
    """Egasini matnda tanish uchun qidiriladigan satrlar (bo'shlar tashlanadi)."""
    out: list[str] = []
    # Uzunlik **`@` siz** o'lchanadi: ikki harfli nom har xil matnda tasodifan
    # uchrab, salience'ni bezovta qiladi.
    for item in ((user.username if user else None), (account.label if account else None)):
        bare = (item or "").strip().lstrip("@").lower()
        if len(bare) < 3:
            continue
        for variant in (bare, f"@{bare}"):
            if variant not in out:
                out.append(variant)
    return out


def is_mentioned(text: str, names: list[str]) -> bool:
    low = (text or "").lower()
    return any(n in low for n in names)


# ─── ma'lumot yig'ish ────────────────────────────────────────────────────────


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return start, start + timedelta(days=1)


async def _selected_chat_ids(session: AsyncSession, account: Account) -> list[int]:
    ids = (
        (await session.execute(select(Chat.id).where(Chat.account_id == account.id)))
        .scalars()
        .all()
    )
    allowed = set(int(x) for x in (account.briefing_chats or []) if isinstance(x, int | str))
    out = [int(i) for i in ids]
    if allowed:
        out = [i for i in out if i in allowed]
    return out


async def _history_counts(
    session: AsyncSession, chat_ids: list[int], day: date
) -> dict[int, list[int]]:
    """Har chat uchun `day` dan oldingi `HISTORY_DAYS` kunlik xabar sonlari.

    Xabar bo'lmagan kunlar **nol** bilan to'ldiriladi: jim chatning bitta
    faol kuni haqiqatan ham anomaliya, uni mediana yashirmasligi kerak.
    """
    if not chat_ids:
        return {}
    start, _ = _day_bounds(day - timedelta(days=HISTORY_DAYS))
    end, _ = _day_bounds(day)
    bucket = func.date_trunc("day", Message.published_at).label("d")
    rows = (
        await session.execute(
            select(Message.chat_id, bucket, func.count(Message.id))
            .where(
                Message.chat_id.in_(chat_ids),
                Message.published_at >= start,
                Message.published_at < end,
            )
            .group_by(Message.chat_id, bucket)
        )
    ).all()
    per: dict[int, dict[date, int]] = {}
    for chat_id, bucket_day, n in rows:
        key = bucket_day.date() if hasattr(bucket_day, "date") else bucket_day
        per.setdefault(int(chat_id), {})[key] = int(n)
    days = [day - timedelta(days=i) for i in range(1, HISTORY_DAYS + 1)]
    return {cid: [per.get(cid, {}).get(d, 0) for d in days] for cid in chat_ids}


async def _open_questions(session: AsyncSession, chat_ids: list[int], day: date) -> dict[int, int]:
    """`chat_digests.structured.open_questions` uzunligi (bor bo'lsa)."""
    if not chat_ids:
        return {}
    start, end = _day_bounds(day)
    rows = (
        await session.execute(
            select(ChatDigest.chat_id, ChatDigest.structured).where(
                ChatDigest.chat_id.in_(chat_ids),
                ChatDigest.day >= start,
                ChatDigest.day < end,
            )
        )
    ).all()
    out: dict[int, int] = {}
    for chat_id, structured in rows:
        items = (structured or {}).get("open_questions") if isinstance(structured, dict) else None
        out[int(chat_id)] = len(items) if isinstance(items, list) else 0
    return out


async def _most_active(
    session: AsyncSession, chat_ids: list[int], day: date
) -> list[dict[str, Any]]:
    """Kim ko'p yozgani — **kodda** sanaladi, modeldan so'ralmaydi."""
    if not chat_ids:
        return []
    start, end = _day_bounds(day)
    rows = (
        await session.execute(
            select(Message.sender_id, func.count(Message.id).label("n"))
            .where(
                Message.chat_id.in_(chat_ids),
                Message.published_at >= start,
                Message.published_at < end,
                Message.sender_id.is_not(None),
            )
            .group_by(Message.sender_id)
            .order_by(func.count(Message.id).desc())
            .limit(TOP_SENDERS)
        )
    ).all()
    return [{"sender_id": int(sid), "count": int(n)} for sid, n in rows]


async def collect_salience(
    session: AsyncSession, account: Account, day: date, *, user: User | None = None
) -> list[ChatSalience]:
    """Kun tugunlari + statistika → salience bo'yicha tartiblangan chatlar."""
    chat_ids = await _selected_chat_ids(session, account)
    if not chat_ids:
        return []
    start, _ = _day_bounds(day)
    nodes = (
        await session.execute(
            select(RetrievalNode, Chat.title)
            .join(Chat, Chat.id == RetrievalNode.chat_id)
            .where(
                RetrievalNode.account_id == account.id,
                RetrievalNode.level == LEVEL_DAY,
                RetrievalNode.starts_at == start,
                RetrievalNode.chat_id.in_(chat_ids),
            )
        )
    ).all()
    if not nodes:
        return []

    present = [int(n.chat_id) for n, _ in nodes]
    history = await _history_counts(session, present, day)
    questions = await _open_questions(session, present, day)
    names = owner_names(user, account)

    out: list[ChatSalience] = []
    for node, title in nodes:
        text = (node.text or "").strip()
        if not text:
            continue
        out.append(
            ChatSalience(
                chat_id=int(node.chat_id),
                title=title or "",
                msg_count=int(node.msg_count or 0),
                z=z_score(int(node.msg_count or 0), history.get(int(node.chat_id), [])),
                mentioned=is_mentioned(text, names),
                open_questions=questions.get(int(node.chat_id), 0),
                text=text,
            )
        )
    out.sort(key=lambda c: (-c.score, -c.msg_count, c.chat_id))
    return out[: get_settings().briefing_max_chats]


# ─── konvert + sintez ────────────────────────────────────────────────────────


def build_envelope(items: list[ChatSalience], day: date, *, top: int) -> str:
    """Barcha kun tugunlari — **bitta** `<untrusted_data kind="digests">` konverti."""
    body: list[str] = []
    used = 0
    for i, c in enumerate(items):
        rank = "full" if i < top else "brief"
        head = f'[chat i={i} title="{U.attr(c.title)}" rank={rank}]'
        block = f"{head}\n{U.neutralize(c.text[:CHAT_TEXT_MAX_CHARS])}"
        if used + len(block) > ENVELOPE_MAX_CHARS and body:
            break
        body.append(block)
        used += len(block) + 1
    head = f'<untrusted_data kind="digests" source="telegram" day="{day.isoformat()}">'
    return f"{head}\n" + "\n\n".join(body) + "\n</untrusted_data>"


def allowed_citations(items: list[ChatSalience]) -> set[str]:
    """Manba matnlaridagi `#<id>` lar — javobda faqat shular bo'lishi mumkin."""
    out: set[str] = set()
    for c in items:
        out.update(CITE_RE.findall(c.text))
    return out


def validate_line(line: str, allowed: set[str]) -> str:
    """Noma'lum sitatalarni o'chiradi; sitatasiz qolgan qatorni tashlaydi ("").

    §9 mexanik nazorati: har bullet'da ≥1 `#msg_id` bo'lishi shart va har id
    oynadagi id'lar to'plamida bo'lishi shart.
    """
    raw = (line or "").strip()
    if not raw:
        return ""
    found = CITE_RE.findall(raw)
    if not found:
        return ""
    kept = [f for f in found if f in allowed]
    if not kept:
        return ""
    cleaned = CITE_RE.sub(lambda m: m.group(0) if m.group(1) in allowed else "", raw)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip(" ,;")
    return cleaned if CITE_RE.search(cleaned) else ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Fence/preambulani kechiradigan JSON o'qish (Claude'da alohida json mode yo'q)."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw[:4].lower() == "json" else raw
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _as_lines(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x or "").strip()]


def validate_draft(draft: dict[str, Any], items: list[ChatSalience], *, top: int) -> dict[str, Any]:
    """Model javobini mexanik tekshiradi va bizning tuzilmaga aylantiradi."""
    allowed = allowed_citations(items)
    by_index: dict[int, list[str]] = {}
    for entry in draft.get("chats") or []:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("i", -1))
        except (TypeError, ValueError):
            continue
        by_index[idx] = _as_lines(entry.get("bullets"))

    chats: list[dict[str, Any]] = []
    for i, c in enumerate(items):
        limit = 3 if i < top else 1
        bullets = [b for b in (validate_line(x, allowed) for x in by_index.get(i, [])) if b]
        chats.append({**c.to_dict(), "full": i < top, "bullets": bullets[:limit]})

    def checked(value: Any) -> list[str]:
        return [x for x in (validate_line(v, allowed) for v in _as_lines(value)) if x][:6]

    return {
        "tldr": str(draft.get("tldr") or "").strip(),
        "chats": chats,
        "decisions": checked(draft.get("decisions")),
        "open_questions": checked(draft.get("open_questions")),
    }


@dataclass(slots=True)
class _Cost:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: str = ""
    provider: str = ""
    calls: list[str] = field(default_factory=list)

    def add(self, res: Any, task: str) -> None:
        self.tokens_in += int(res.usage.tokens_in)
        self.tokens_out += int(res.usage.tokens_out)
        self.model = self.model or res.model
        self.provider = self.provider or res.provider
        c = estimate_cost(res.provider, res.model, res.usage.tokens_in, res.usage.tokens_out)
        if c:
            self.cost_usd += c
        self.calls.append(task)


async def synthesize(
    items: list[ChatSalience],
    day: date,
    locale: str,
    *,
    top: int,
    llm: LLM | None = None,
    cod_rounds: int | None = None,
) -> tuple[dict[str, Any], _Cost] | None:
    """Konvert → **bitta** `Task.SEARCH` → (ixtiyoriy) CoD raundlari.

    DB'ga tegmaydi — shuning uchun testda soxta LLM bilan to'liq qamrab
    olinadi. Xato bo'lsa `None` (briefing yozilmaydi).
    """
    s = get_settings()
    envelope = build_envelope(items, day, top=top)
    cost = _Cost()
    user_turn = (
        f"Briefing language: {locale}\n"
        f"Day: {day.isoformat()}\n"
        f"Chats: {len(items)} (first {top} are rank=full)\n\n{envelope}"
    )
    try:
        res = await (llm or LLM()).chat(
            Task.SEARCH,
            [Msg.system(BRIEFING_PROMPT), Msg.user(user_turn)],
            json_mode=True,
            max_tokens=BRIEFING_MAX_TOKENS,
        )
    except LLMError as exc:
        log.warning("briefing.llm_failed", day=str(day), error=str(exc)[:200])
        return None
    cost.add(res, "search")
    draft = parse_json_object(res.text) or {}

    # Chain-of-Density — TL;DR va mavzu bulletlari ustida, arzon `Task.ROUTE`.
    rounds = max(0, int(s.briefing_cod_rounds if cod_rounds is None else cod_rounds))
    for _ in range(rounds):
        draft = await _densify(draft, envelope, locale, llm=llm, cost=cost) or draft
    return draft, cost


async def build_briefing(
    session: AsyncSession,
    account_id: int,
    day: date,
    *,
    llm: LLM | None = None,
    force: bool = False,
) -> Briefing | None:
    """Bir akkauntning bir kuni uchun briefing quradi va `briefings` ga yozadi.

    Args:
        session: DB sessiyasi.
        account_id: `accounts.id`.
        day: qamraladigan **lokal** kun.
        llm: LLM fasadi (testda soxta).
        force: mavjud qator bo'lsa ham qayta hisoblash.

    Returns:
        Yozilgan `Briefing`; kun tugunlari bo'lmasa `None`.
    """
    account = await session.get(Account, account_id)
    if account is None:
        return None
    existing = (
        await session.execute(
            select(Briefing).where(Briefing.account_id == account_id, Briefing.day == day)
        )
    ).scalar_one_or_none()
    if existing is not None and not force:
        return existing

    user = await session.get(User, account.user_id)
    items = await collect_salience(session, account, day, user=user)
    if not items:
        log.info("briefing.no_nodes", account_id=account_id, day=str(day))
        return None

    s = get_settings()
    top = min(s.briefing_top_chats, len(items))
    locale = (user.locale if user else None) or s.default_locale
    synth = await synthesize(items, day, locale, top=top, llm=llm)
    if synth is None:
        return None
    draft, cost = synth

    structured = validate_draft(draft, items, top=top)
    structured["most_active"] = await _most_active(session, [c.chat_id for c in items], day)
    structured["day"] = day.isoformat()
    structured["locale"] = locale
    text = render_markdown(structured)

    row = existing or Briefing(account_id=account_id, day=day)
    if existing is None:
        session.add(row)
    row.text = text
    row.structured = structured
    row.model = cost.model
    row.tokens_in = cost.tokens_in
    row.tokens_out = cost.tokens_out
    row.cost_usd = round(cost.cost_usd, 6) or None
    row.delivered_at = None
    row.delivery_error = None
    await session.flush()
    await record_usage(session, account, cost, text)
    log.info(
        "briefing.built",
        account_id=account_id,
        day=str(day),
        chats=len(items),
        calls=len(cost.calls),
        tokens=cost.tokens_in + cost.tokens_out,
    )
    return row


async def _densify(
    draft: dict[str, Any],
    envelope: str,
    locale: str,
    *,
    llm: LLM | None,
    cost: _Cost,
) -> dict[str, Any] | None:
    """Bitta Chain-of-Density raundi. Xato bo'lsa `None` — oldingi qoralama qoladi."""
    payload = json.dumps(
        {k: draft.get(k) for k in ("tldr", "chats", "decisions", "open_questions")},
        ensure_ascii=False,
    )
    try:
        res = await (llm or LLM()).chat(
            Task.ROUTE,
            [
                Msg.system(BRIEFING_COD_PROMPT),
                Msg.user(f"Briefing language: {locale}\nDraft:\n{payload}\n\n{envelope}"),
            ],
            json_mode=True,
            max_tokens=BRIEFING_MAX_TOKENS,
        )
    except LLMError as exc:
        log.warning("briefing.cod_failed", error=str(exc)[:200])
        return None
    cost.add(res, "cod")
    return parse_json_object(res.text)


async def record_usage(session: AsyncSession, account: Account, cost: _Cost, text: str) -> None:
    """Briefing tokenlarini kunlik byudjet yo'liga qo'shadi (digest'lar kabi).

    `services/limits.daily_usage` faqat `conversation_messages` ni ko'radi,
    shuning uchun briefing ham o'sha yerga tushadi: akkaunt uchun bitta
    "Briefing" suhbati ochiladi va har qurilgan briefing unga `assistant`
    qatori bo'lib qo'shiladi. Aks holda fon xarajati limitdan ko'rinmay
    qolardi.
    """
    if not (cost.tokens_in or cost.tokens_out):
        return
    conv = (
        await session.execute(
            select(Conversation)
            .where(
                Conversation.user_id == account.user_id,
                Conversation.account_id == account.id,
                Conversation.title == "Briefing",
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    if conv is None:
        conv = Conversation(user_id=account.user_id, account_id=account.id, title="Briefing")
        session.add(conv)
        await session.flush()
    session.add(
        ConversationMessage(
            conversation_id=conv.id,
            role="assistant",
            content=text[:4000],
            model=cost.model,
            provider=cost.provider,
            tokens_in=cost.tokens_in,
            tokens_out=cost.tokens_out,
            task="briefing",
            cost_usd=round(cost.cost_usd, 6) or None,
        )
    )
    await session.flush()


# ─── render ──────────────────────────────────────────────────────────────────


def render_markdown(structured: dict[str, Any]) -> str:
    """To'liq briefing — web uchun markdown (`md.js` render qiladi)."""
    out: list[str] = []
    if structured.get("tldr"):
        out.append(str(structured["tldr"]))
    for c in structured.get("chats") or []:
        bullets = c.get("bullets") or []
        if not bullets:
            continue
        if c.get("full"):
            out.append(f"\n**{c.get('title') or '—'}**")
            out.extend(f"- {b}" for b in bullets)
        else:
            out.append(f"- **{c.get('title') or '—'}**: {bullets[0]}")
    for key, head in (("decisions", "Decisions"), ("open_questions", "Open questions")):
        lines = structured.get(key) or []
        if lines:
            out.append(f"\n**{head}**")
            out.extend(f"- {x}" for x in lines)
    active = structured.get("most_active") or []
    if active:
        out.append("\n**Most active**")
        out.append(", ".join(f"user:{a['sender_id']} ({a['count']})" for a in active))
    return "\n".join(out).strip()


def render_html(
    structured: dict[str, Any], *, title: str, link: str = "", link_label: str = ""
) -> str:
    """Bot push uchun HTML — TL;DR + top chatlar + "to'liq briefing" havolasi.

    Har teg **bitta qator** ichida ochilib yopiladi: `bot/push.split_html`
    qatorlar chegarasida bo'ladi, shuning uchun bo'lakning HTML'i buzilmaydi.
    """
    esc = html.escape
    lines: list[str] = [f"<b>{esc(title)}</b>"]
    if structured.get("tldr"):
        lines.append(esc(str(structured["tldr"])))
    for c in structured.get("chats") or []:
        if not c.get("full") or not (c.get("bullets") or []):
            continue
        lines.append("")
        lines.append(f"<b>{esc(str(c.get('title') or '—'))}</b>")
        lines.extend(f"• {esc(b)}" for b in c["bullets"])
    brief = [c for c in (structured.get("chats") or []) if not c.get("full") and c.get("bullets")]
    if brief:
        lines.append("")
        lines.extend(
            f"• <b>{esc(str(c.get('title') or '—'))}</b>: {esc(c['bullets'][0])}" for c in brief
        )
    active = structured.get("most_active") or []
    if active:
        lines.append("")
        top = active[0]
        lines.append(f"<i>user:{top['sender_id']} — {top['count']}</i>")
    if link:
        lines.append("")
        lines.append(f'<a href="{esc(link, quote=True)}">{esc(link_label or link)}</a>')
    return "\n".join(lines)


# ─── yetkazish ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    account_id: int
    day: date
    status: str  # sent | skipped | error
    reason: str = ""
    message_id: int | None = None


async def deliver_account(
    session: AsyncSession,
    account: Account,
    *,
    now: datetime | None = None,
    send: Any = None,
    title: str = "Briefing",
    link_label: str = "",
) -> DeliveryResult:
    """Bitta akkauntning joriy nashrini push qiladi (sozlama/jim oyna hisobga olinadi)."""
    check = delivery_check(account, now=now)
    if not check.ok:
        return DeliveryResult(account.id, check.day, "skipped", check.reason)

    row = (
        await session.execute(
            select(Briefing).where(Briefing.account_id == account.id, Briefing.day == check.day)
        )
    ).scalar_one_or_none()
    if row is None:
        return DeliveryResult(account.id, check.day, "skipped", "not_built")
    if row.delivered_at is not None:
        return DeliveryResult(account.id, check.day, "skipped", "already_delivered")

    user = await session.get(User, account.user_id)
    if user is None or not user.tg_user_id:
        return DeliveryResult(account.id, check.day, "skipped", "no_owner")

    from app.bot.push import mini_app_link, send_html

    sender = send or send_html
    body = render_html(
        row.structured or {},
        title=f"{title} · {check.day.isoformat()}",
        link=mini_app_link(f"/briefings/{check.day.isoformat()}"),
        link_label=link_label or "full briefing",
    )
    try:
        msg_id = await sender(int(user.tg_user_id), body)
    except Exception as exc:  # sender o'zi yutadi, bu — ikkinchi chiziq
        row.delivery_error = str(exc)[:500]
        await session.flush()
        return DeliveryResult(account.id, check.day, "error", str(exc)[:200])

    if msg_id is None:
        row.delivery_error = "push_failed"
        await session.flush()
        return DeliveryResult(account.id, check.day, "error", "push_failed")

    row.delivered_at = datetime.now(UTC)
    row.delivery_error = None
    await session.flush()
    log.info("briefing.delivered", account_id=account.id, day=str(check.day), msg_id=msg_id)
    return DeliveryResult(account.id, check.day, "sent", "ok", int(msg_id))


async def enabled_accounts(session: AsyncSession) -> list[Account]:
    return list(
        (
            await session.execute(
                select(Account).where(Account.briefing_enabled.is_(True)).order_by(Account.id)
            )
        )
        .scalars()
        .all()
    )


__all__ = [
    "BRIEFING_MAX_TOKENS",
    "CITE_RE",
    "HISTORY_DAYS",
    "MENTION_BONUS",
    "QUESTION_BONUS",
    "ChatSalience",
    "DeliveryCheck",
    "DeliveryResult",
    "allowed_citations",
    "build_briefing",
    "build_envelope",
    "collect_salience",
    "deliver_account",
    "delivery_check",
    "edition_day",
    "enabled_accounts",
    "in_quiet_hours",
    "is_mentioned",
    "local_now",
    "owner_names",
    "parse_json_object",
    "render_html",
    "render_markdown",
    "synthesize",
    "validate_draft",
    "validate_line",
    "z_score",
    "zone",
]
