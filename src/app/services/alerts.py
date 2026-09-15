"""Alertlar (3-bosqich/C) — kuzatuv qoidalari va boshqaruv botiga push.

Ikki turdagi qoida:

``keyword``
    Vergul bilan ajratilgan so'zlar. Moslik **fold fazosida** (`uz_fold`)
    `messages.tsv_uz` ustidan, va **xom** so'zlar bilan `messages.tsv_ru`
    ustidan qidiriladi. Ikki kanal kerak: `tsv_uz` `simple` konfiguratsiya
    (stemming yo'q, lekin apostrof/kirill bitta fazoda), `tsv_ru` esa ruscha
    stemming beradi ("платеж" → "платежи" ni ham topadi). Fold qilingan
    ruscha so'z `tsv_ru` ga tushmaydi (u lotinga o'giriladi), shuning uchun
    ruscha kanalga aynan xom so'z beriladi.

``semantic``
    Tabiiy tildagi tavsif → bitta `RETRIEVAL_QUERY` embedding → cosine.
    Vektor `alert_rules.embedding` da **keshlanadi**: skaner har 5 daqiqada
    yuguradi, qoida matni esa deyarli o'zgarmaydi — har yugurishda qayta
    embed qilish bekor xarajat. `pattern` yangilanganda router `embedding`
    ni `NULL` ga tushiradi.

Qopqoqlar (o'lchov bilan emas, xavfsizlik bilan qo'yilgan):

  * **Kursor** `last_checked_msg_id` — har skaner faqat yangi xabarlarni
    ko'radi. Usiz har yugurish butun tarixni qayta tekshirardi.
  * **Cooldown** (`cooldown_minutes`, `(rule, chat)` bo'yicha) — gapiradigan
    chatda "to'lov" so'zi kuniga 40 marta uchraydi; usiz bot bir soatda
    40 ta push yuborardi va foydalanuvchi alertni o'chirib qo'yardi.
  * **Dedup bazada** (`uq_alert_event`) — parallel skaner ham ikkinchi
    hodisani yarata olmaydi.
  * **≤5 hodisa bitta push'da**, qolgani "va yana N ta".

Xavfsizlik:

  * Push **faqat** egasining boshqaruv boti chatiga ketadi
    (`accounts.user_id → users.tg_user_id`), ixtiyoriy peer'ga emas.
  * Xabar matni push'ga tushishdan oldin `untrusted.neutralize` dan, keyin
    `html.escape` dan o'tadi: konvert tegi ham, HTML tegi ham yasalmaydi.
  * Telefon raqami hech qachon qaytarilmaydi — qoida ham, hodisa ham
    faqat chat/xabar identifikatorlari bilan ishlaydi.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Account, AlertEvent, AlertKind, AlertRule, Chat, Message, User
from app.llm import EMBED_QUERY, LLM, LLMError
from app.logging import get_logger
from app.services import untrusted as U
from app.services.search import fold_terms
from app.services.textnorm import uz_fold

log = get_logger(__name__)

__all__ = [
    "MAX_EVENTS_PER_PUSH",
    "PATTERN_MAX_CHARS",
    "AlertError",
    "compile_keyword_query",
    "deliver_alerts",
    "is_quiet",
    "render_push",
    "scan_rule",
]

# ── qopqoqlar ────────────────────────────────────────────────────────────────
PATTERN_MAX_CHARS = 300
MAX_TERMS = 12
MAX_EVENTS_PER_PUSH = 5  # qolgani "va yana N ta"
MAX_EVENTS_PER_RUN = 200  # bitta deliver yugurishida ko'riladigan hodisalar
SNIPPET_CHARS = 200
PUSH_MAX_CHARS = 3800  # Telegram chegarasi 4096 — sarlavha/havolalarga zapas
SCAN_LIMIT = 500

THRESHOLD_MIN = 0.5
THRESHOLD_MAX = 0.95
COOLDOWN_MAX = 1440

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_QUOTED = re.compile(r'^\s*"(.*)"\s*$|^\s*«(.*)»\s*$', re.DOTALL)


class AlertError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


# ─── sof funksiyalar ─────────────────────────────────────────────────────────


def is_quiet(
    now_utc: datetime, tz: str | None, quiet_from: time | None, quiet_to: time | None
) -> bool:
    """Akkauntning **lokal** vaqti jim soatlar oralig'idami?

    `quiet_from`/`quiet_to` — `tz` dagi lokal vaqt, yarim tun orqali o'ralishi
    mumkin (22:00 → 07:00). Ikkisidan biri `None` bo'lsa jim soat yo'q; teng
    bo'lsa ham yo'q (bo'sh oraliq, "butun sutka jim" degani emas).

    Noma'lum `tz` — UTC (jim soat butunlay yo'qolib ketgandan ko'ra siljigani
    yaxshi; `zoneinfo` ma'lumot bazasi yo'q konteynerda ham yiqilmaydi).
    """
    if quiet_from is None or quiet_to is None or quiet_from == quiet_to:
        return False
    try:
        zone = ZoneInfo(tz) if tz else UTC
    except (ZoneInfoNotFoundError, ValueError):
        zone = UTC
    local = now_utc.astimezone(zone).time()
    if quiet_from < quiet_to:
        return quiet_from <= local < quiet_to
    return local >= quiet_from or local < quiet_to  # tun oralig'i


def split_pattern(pattern: str) -> list[str]:
    """Vergul bilan ajratilgan qismlar (qo'shtirnoq ichidagi vergul saqlanadi?).

    Yo'q — ataylab oddiy: vergul har doim ajratadi. Ibora qo'shtirnoq bilan
    beriladi (`"yangi ofis"`), ibora ichida vergul kerak bo'lgan holat
    uchramadi va uni qo'llab-quvvatlash parserni kutilmagan qiladi.
    """
    return [p.strip() for p in (pattern or "").split(",") if p.strip()]


def _tokens(part: str, *, fold: bool) -> list[str]:
    """Bitta qism → tsquery tokenlari (fold fazosida yoki xom).

    Fold yo'li **`search.fold_terms`** orqali o'tadi — `messages.tsv_uz`
    generated ustuni aynan o'sha `uz_fold` bilan qurilgan, ikki tomon
    ajralib ketsa qidiruv jimgina hech narsa topmay qo'yadi.

    Keyin har token `to_tsquery` operatorlaridan tozalanadi: `!`, `&`, `|`,
    `:` va qavslar naqshda qolib ketsa tsquery sintaksisi buziladi
    (Postgres xato qaytaradi, skaner esa har 5 daqiqada qayta uriladi).
    """
    raws = fold_terms([part], max_terms=MAX_TERMS) if fold else part.lower().split()
    out: list[str] = []
    for raw in raws:
        tok = _NON_WORD.sub("", raw)
        if len(tok) >= 2 and tok not in out:
            out.append(tok)
    return out[:MAX_TERMS]


def compile_keyword_query(pattern: str, *, fold: bool = True) -> str:
    """`to_tsquery` satri: qismlar **OR**, qo'shtirnoqli ibora — `<->` qo'shnilik.

    `fold=True` (sukut) — `messages.tsv_uz` uchun: `search.fold_terms` bilan
    aynan indeks qurilgan fazoga tushiriladi (apostrof va kirill variantlari bitta tokenga tushadi).
    `fold=False` — `messages.tsv_ru` uchun: xom so'zlar, `russian`
    konfiguratsiyasi ularni o'zi stemming qiladi.

    Prefiks `:*` faqat **oxirgi** tokenda: "to'lo" → "tolo:*" bir necha
    shaklni tutadi, ibora ichidagi har tokenni prefiksga aylantirish esa
    qo'shnilikni ma'nosiz kengaytiradi.

    Bo'sh yoki faqat belgi bo'lgan naqsh → bo'sh satr (chaqiruvchi so'rov
    qurmaydi; bo'sh tsquery Postgres'da xato beradi).

    Examples:
        >>> compile_keyword_query("to'lov, shartnoma")
        'tolov:* | shartnoma:*'
        >>> compile_keyword_query('"yangi ofis"')
        '(yangi <-> ofis:*)'
    """
    groups: list[str] = []
    for part in split_pattern(pattern)[:MAX_TERMS]:
        m = _QUOTED.match(part)
        phrase = m is not None
        body = next((g for g in (m.groups() if m else ()) if g is not None), part)
        toks = _tokens(body, fold=fold)
        if not toks:
            continue
        if len(toks) == 1:
            groups.append(f"{toks[0]}:*")
            continue
        # Ibora — qo'shnilik (`<->`); qo'shtirnoqsiz ko'p so'zli qism — `&`
        # (so'zlar xabarda bo'lsin, tartibi muhim emas).
        op = " <-> " if phrase else " & "
        groups.append("(" + op.join([*toks[:-1], f"{toks[-1]}:*"]) + ")")
    return " | ".join(dict.fromkeys(groups))


def snippet(text: str, *, max_chars: int = SNIPPET_CHARS) -> str:
    """Push uchun xavfsiz parcha: neutralize → bir qatorga → kesish → escape.

    Tartib muhim: `neutralize` konvert tegini o'ldiradi (`<untrusted_data>`),
    `html.escape` esa Telegram HTML tegini (`<b>`). Ikkalasi ham kerak —
    biri ikkinchisini almashtirmaydi.
    """
    clean = U.neutralize(text or "").replace("\n", " ").strip()
    clean = re.sub(r"\s{2,}", " ", clean)
    if len(clean) > max_chars:
        clean = clean[: max_chars - 1].rstrip() + "…"
    return html.escape(clean, quote=False)


def rule_label(rule: Any) -> str:
    raw = (getattr(rule, "label", "") or "").strip() or (rule.pattern or "").strip()
    return html.escape(raw[:64], quote=False) or "alert"


def message_link(chat: Any, tg_msg_id: int) -> str:
    prefix = getattr(chat, "link_prefix", None)
    return f"{prefix}{int(tg_msg_id)}" if prefix else ""


# ─── skaner ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Candidate:
    message_id: int
    chat_id: int
    tg_msg_id: int
    score: float


def _base_where(rule: Any, owner_tg_id: int | None) -> list[Any]:
    """Har ikki yo'l uchun umumiy filtr: yangi, o'chirilmagan, o'zimizniki emas."""
    conds: list[Any] = [
        Message.id > int(rule.last_checked_msg_id or 0),
        Message.deleted_at.is_(None),
    ]
    if rule.chat_id is not None:
        conds.append(Message.chat_id == int(rule.chat_id))
    else:
        conds.append(Chat.account_id == int(rule.account_id))
    if owner_tg_id is not None:
        # O'z xabaringiz haqida sizga xabar berish ma'nosiz (va cooldown'ni yeydi).
        conds.append(or_(Message.sender_id.is_(None), Message.sender_id != int(owner_tg_id)))
    return conds


def _keyword_stmt(rule: Any, owner_tg_id: int | None, limit: int) -> Any:
    from sqlalchemy import func as sa_func

    q_uz = compile_keyword_query(rule.pattern, fold=True)
    q_ru = compile_keyword_query(rule.pattern, fold=False)
    if not q_uz and not q_ru:
        return None
    match: list[Any] = []
    if q_uz:
        match.append(Message.tsv_uz.op("@@")(sa_func.to_tsquery("simple", q_uz)))
    if q_ru:
        match.append(Message.tsv_ru.op("@@")(sa_func.to_tsquery("russian", q_ru)))
    return (
        select(Message.id, Message.chat_id, Message.tg_msg_id)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*_base_where(rule, owner_tg_id), or_(*match))
        .order_by(Message.id)
        .limit(limit)
    )


def _semantic_stmt(rule: Any, owner_tg_id: int | None, qvec: Any, limit: int) -> Any:
    from app.db.models import MessageEmbedding

    sim = (1.0 - MessageEmbedding.vector.cosine_distance(qvec)).label("sim")
    threshold = float(rule.threshold if rule.threshold is not None else 0.80)
    return (
        select(Message.id, Message.chat_id, Message.tg_msg_id, sim)
        .join(Chat, Chat.id == Message.chat_id)
        .join(MessageEmbedding, MessageEmbedding.message_id == Message.id)
        .where(*_base_where(rule, owner_tg_id), sim >= threshold)
        .order_by(Message.id)
        .limit(limit)
    )


async def _owner_tg_id(session: AsyncSession, account_id: int) -> int | None:
    acc = await session.get(Account, int(account_id))
    return int(acc.tg_account_id) if acc is not None and acc.tg_account_id else None


async def _rule_vector(session: AsyncSession, rule: Any, llm: LLM | None) -> list[float] | None:
    """Keshlangan vektor yoki bitta `RETRIEVAL_QUERY` embedding (kesh yoziladi)."""
    cached = getattr(rule, "embedding", None)
    if cached is not None:
        return [float(x) for x in cached]
    text_ = (rule.pattern or "").strip()
    if not text_:
        return None
    try:
        res = await (llm or LLM()).embed([text_], task_type=EMBED_QUERY)
    except LLMError as exc:
        log.warning("alerts.embed_failed", rule_id=getattr(rule, "id", None), error=str(exc)[:200])
        return None
    if not res.vectors:
        return None
    vec = [float(x) for x in res.vectors[0]]
    rule.embedding = vec  # kesh — keyingi skanerda embedding chaqirilmaydi
    return vec


async def _last_event_per_chat(session: AsyncSession, rule_id: int) -> dict[int, datetime]:
    from sqlalchemy import func as sa_func

    rows = (
        await session.execute(
            select(AlertEvent.chat_id, sa_func.max(AlertEvent.created_at))
            .where(AlertEvent.rule_id == int(rule_id))
            .group_by(AlertEvent.chat_id)
        )
    ).all()
    out: dict[int, datetime] = {}
    for chat_id, at in rows:
        if at is not None:
            out[int(chat_id)] = at
    return out


def _apply_cooldown(
    cands: list[Candidate], last_at: dict[int, datetime], cooldown_minutes: int, now: datetime
) -> list[Candidate]:
    """`(rule, chat)` bo'yicha shift: bitta chatdan cooldown ichida bitta hodisa."""
    if cooldown_minutes <= 0:
        return cands
    window = timedelta(minutes=int(cooldown_minutes))
    out: list[Candidate] = []
    seen: set[int] = set()
    for c in cands:
        prev = last_at.get(c.chat_id)
        if c.chat_id in seen or (prev is not None and now - prev < window):
            continue
        seen.add(c.chat_id)
        out.append(c)
    return out


async def scan_rule(
    session: AsyncSession, rule: Any, *, llm: LLM | None = None, limit: int = SCAN_LIMIT
) -> list[AlertEvent]:
    """Qoidani yangi xabarlarga qo'llaydi va `alert_events` yaratadi.

    Kursor (`last_checked_msg_id`) **har doim** oldinga siljiydi: mos keluvchi
    topilmasa ham, aks holda har yugurish bir xil oynani qayta skanerlardi.
    Agar natija `limit` ga tiqilib qolsa — kursor faqat oxirgi ko'rilgan
    xabargacha siljiydi, qolgani keyingi yugurishda ko'riladi.
    """
    now = datetime.now(UTC)
    owner = await _owner_tg_id(session, rule.account_id)
    kind = str(getattr(rule, "kind", AlertKind.KEYWORD))

    if kind == AlertKind.SEMANTIC:
        qvec = await _rule_vector(session, rule, llm)
        if qvec is None:
            return []
        stmt = _semantic_stmt(rule, owner, qvec, limit)
    else:
        kw = _keyword_stmt(rule, owner, limit)
        if kw is None:
            return []
        stmt = kw

    rows = (await session.execute(stmt)).all()
    cands = [
        Candidate(
            message_id=int(r[0]),
            chat_id=int(r[1]),
            tg_msg_id=int(r[2]),
            score=float(r[3]) if len(r) > 3 and r[3] is not None else 1.0,
        )
        for r in rows
    ]

    # Kursor: to'liq oyna ko'rilgan bo'lsa eng katta id'gacha, aks holda
    # oxirgi qaytgan xabargacha (limit bilan kesilgan qolgan qismi yo'qolmasin).
    max_seen = max((c.message_id for c in cands), default=0)
    if len(cands) < limit:
        max_seen = max(max_seen, await _scope_max_id(session, rule))
    if max_seen > int(rule.last_checked_msg_id or 0):
        rule.last_checked_msg_id = max_seen

    if not cands:
        return []

    fresh = _apply_cooldown(
        cands, await _last_event_per_chat(session, rule.id), int(rule.cooldown_minutes or 0), now
    )
    if not fresh:
        return []
    return await _insert_events(session, rule.id, fresh)


async def _scope_max_id(session: AsyncSession, rule: Any) -> int:
    """Qoida qamrovidagi eng katta `messages.id` — bo'sh skanerdan keyingi kursor."""
    from sqlalchemy import func as sa_func

    stmt = select(sa_func.max(Message.id)).join(Chat, Chat.id == Message.chat_id)
    if rule.chat_id is not None:
        stmt = stmt.where(Message.chat_id == int(rule.chat_id))
    else:
        stmt = stmt.where(Chat.account_id == int(rule.account_id))
    val = (await session.execute(stmt)).scalar()
    return int(val or 0)


async def _insert_events(
    session: AsyncSession, rule_id: int, cands: list[Candidate]
) -> list[AlertEvent]:
    """`ON CONFLICT DO NOTHING` — takror hodisa jimgina tushib qoladi."""
    stmt = (
        pg_insert(AlertEvent)
        .values(
            [
                {
                    "rule_id": int(rule_id),
                    "message_id": c.message_id,
                    "chat_id": c.chat_id,
                    "tg_msg_id": c.tg_msg_id,
                    "score": c.score,
                }
                for c in cands
            ]
        )
        .on_conflict_do_nothing(index_elements=["rule_id", "message_id"])
        .returning(AlertEvent.id, AlertEvent.message_id)
    )
    rows = (await session.execute(stmt)).all()
    by_msg = {c.message_id: c for c in cands}
    out: list[AlertEvent] = []
    for ev_id, msg_id in rows:
        c = by_msg.get(int(msg_id))
        if c is None:
            continue
        out.append(
            AlertEvent(
                id=int(ev_id),
                rule_id=int(rule_id),
                message_id=c.message_id,
                chat_id=c.chat_id,
                tg_msg_id=c.tg_msg_id,
                score=c.score,
            )
        )
    return out


# ─── yetkazish ───────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PushItem:
    chat_title: str
    date: datetime | None
    text: str
    link: str
    tg_msg_id: int


def render_push(label: str, items: list[PushItem], extra: int = 0) -> str:
    """Bitta qoida uchun HTML push (`parse_mode=HTML`, ≤ `PUSH_MAX_CHARS`).

    Barcha o'zgaruvchi qismlar allaqachon `snippet()` / `html.escape` dan
    o'tgan bo'lishi shart — bu funksiya yangi escaping qilmaydi, faqat
    sarlavha va sanani qo'shadi.
    """
    lines = [f"🔔 <b>{label}</b>"]
    for i, it in enumerate(items, 1):
        when = it.date.strftime("%Y-%m-%d %H:%M") if it.date else "—"
        head = f"{i}. <b>{it.chat_title}</b> · {when}"
        if it.link:
            head += f' · <a href="{html.escape(it.link, quote=True)}">#{it.tg_msg_id}</a>'
        lines.append(head)
        if it.text:
            lines.append(it.text)
    if extra > 0:
        lines.append(f"<i>+{extra}</i>")
    out = "\n".join(lines)
    return out if len(out) <= PUSH_MAX_CHARS else out[: PUSH_MAX_CHARS - 1] + "…"


async def _send_html(user_tg_id: int, text: str) -> int | None:
    """`bot.push.send_html` ustidagi yupqa qatlam — testlar shu yerni almashtiradi."""
    from app.bot.push import send_html

    return await send_html(user_tg_id, text)


async def _owner_chat_id(session: AsyncSession, account_id: int) -> int | None:
    """Push manzili: **faqat** akkaunt egasining boshqaruv boti chati."""
    row = (
        await session.execute(
            select(User.tg_user_id)
            .join(Account, Account.user_id == User.id)
            .where(Account.id == int(account_id))
        )
    ).first()
    return int(row[0]) if row and row[0] else None


async def deliver_alerts(
    session: AsyncSession, account_id: int, now: datetime | None = None
) -> dict[str, Any]:
    """Yetkazilmagan hodisalarni qoida bo'yicha guruhlab push qiladi.

    Jim soatlarda **hech narsa yuborilmaydi va hech narsa yo'qolmaydi** —
    hodisalar `delivered_at IS NULL` bo'lib qoladi va keyingi yugurishda
    (jim soatlar tugagach) bitta guruhlangan push bo'lib ketadi.
    """
    now = now or datetime.now(UTC)
    acc = await session.get(Account, int(account_id))
    if acc is None:
        return {"skipped": "no_account"}
    if is_quiet(
        now,
        getattr(acc, "tz", None),
        getattr(acc, "quiet_from", None),
        getattr(acc, "quiet_to", None),
    ):
        return {"skipped": "quiet"}

    rows = (
        await session.execute(
            select(AlertEvent, AlertRule, Chat, Message)
            .join(AlertRule, AlertRule.id == AlertEvent.rule_id)
            .join(Chat, Chat.id == AlertEvent.chat_id)
            .join(Message, Message.id == AlertEvent.message_id)
            .where(AlertRule.account_id == int(account_id), AlertEvent.delivered_at.is_(None))
            .order_by(AlertEvent.rule_id, AlertEvent.created_at)
            .limit(MAX_EVENTS_PER_RUN)
        )
    ).all()
    if not rows:
        return {"pushes": 0, "events": 0}

    target = await _owner_chat_id(session, account_id)
    if target is None:
        return {"skipped": "no_owner"}

    groups: dict[int, list[tuple[Any, Any, Any, Any]]] = {}
    for ev, rule, chat, msg in rows:
        groups.setdefault(int(ev.rule_id), []).append((ev, rule, chat, msg))

    pushes = 0
    delivered = 0
    failed = 0
    for grp in groups.values():
        shown = grp[:MAX_EVENTS_PER_PUSH]
        items = [
            PushItem(
                chat_title=U.attr(chat.title or ""),
                date=msg.published_at,
                text=snippet(msg.text or ""),
                link=message_link(chat, ev.tg_msg_id),
                tg_msg_id=int(ev.tg_msg_id),
            )
            for ev, _rule, chat, msg in shown
        ]
        body = render_push(rule_label(grp[0][1]), items, extra=max(0, len(grp) - len(shown)))
        try:
            # `bot.push.send_html` xato ko'tarmaydi — muvaffaqiyatsizlikda `None`.
            # Ikkala signal ham bir xil ishlanadi: hodisa yetkazilmagan bo'lib qoladi.
            if await _send_html(target, body) is None:
                raise AlertError("push_failed")
        except Exception as exc:  # bitta qoida xatosi qolganlarini to'xtatmasin
            err = str(exc)[:500]
            log.warning("alerts.push_failed", rule_id=grp[0][0].rule_id, error=err[:200])
            for ev, *_ in grp:
                ev.delivery_error = err
            failed += len(grp)
            continue
        pushes += 1
        for ev, *_ in grp:
            ev.delivered_at = now
            ev.delivery_error = None
            delivered += 1
    log.info("alerts.delivered", account_id=account_id, pushes=pushes, events=delivered)
    return {"pushes": pushes, "events": delivered, "failed": failed}


# ─── qoida CRUD yordamchilari (router ishlatadi) ─────────────────────────────


def validate(kind: str, pattern: str, threshold: float, cooldown_minutes: int) -> None:
    """Router validatsiyasining **mazmuniy** qismi (pydantic shakl qismini bajaradi)."""
    if kind not in {AlertKind.KEYWORD, AlertKind.SEMANTIC}:
        raise AlertError("bad_kind")
    p = (pattern or "").strip()
    if not p:
        raise AlertError("empty_pattern")
    if len(p) > PATTERN_MAX_CHARS:
        raise AlertError("pattern_too_long")
    if not (THRESHOLD_MIN <= float(threshold) <= THRESHOLD_MAX):
        raise AlertError("bad_threshold")
    if not (0 <= int(cooldown_minutes) <= COOLDOWN_MAX):
        raise AlertError("bad_cooldown")
    if kind == AlertKind.KEYWORD and not compile_keyword_query(p):
        raise AlertError("empty_pattern")


def rule_dict(rule: Any) -> dict[str, Any]:
    """API javobi — vektor va boshqa ichki maydonlar chiqmaydi."""
    return {
        "id": int(rule.id),
        "account_id": int(rule.account_id),
        "chat_id": int(rule.chat_id) if rule.chat_id is not None else None,
        "kind": str(rule.kind),
        "label": rule.label or "",
        "pattern": rule.pattern or "",
        "threshold": float(rule.threshold or 0),
        "enabled": bool(rule.enabled),
        "cooldown_minutes": int(rule.cooldown_minutes or 0),
        "last_checked_msg_id": int(rule.last_checked_msg_id or 0),
        "created_at": rule.created_at.isoformat() if getattr(rule, "created_at", None) else None,
    }


def event_dict(ev: Any, chat: Any = None, msg: Any = None) -> dict[str, Any]:
    return {
        "id": int(ev.id),
        "rule_id": int(ev.rule_id),
        "chat_id": int(ev.chat_id),
        "chat_title": U.attr(getattr(chat, "title", "") or "") if chat is not None else "",
        "tg_msg_id": int(ev.tg_msg_id),
        "score": round(float(ev.score or 0), 4),
        "text": snippet(getattr(msg, "text", "") or "") if msg is not None else "",
        "link": message_link(chat, ev.tg_msg_id) if chat is not None else "",
        "created_at": ev.created_at.isoformat() if getattr(ev, "created_at", None) else None,
        "delivered_at": ev.delivered_at.isoformat() if ev.delivered_at else None,
        "delivery_error": ev.delivery_error,
    }


async def enabled_rule_ids(session: AsyncSession, account_id: int | None = None) -> list[int]:
    stmt = select(AlertRule.id).where(AlertRule.enabled.is_(True))
    if account_id is not None:
        stmt = stmt.where(AlertRule.account_id == int(account_id))
    return [int(r[0]) for r in (await session.execute(stmt.order_by(AlertRule.id))).all()]


async def accounts_with_rules(session: AsyncSession) -> list[int]:
    rows = (
        await session.execute(
            select(AlertRule.account_id).where(AlertRule.enabled.is_(True)).distinct()
        )
    ).all()
    return [int(r[0]) for r in rows]


def fold_preview(pattern: str) -> list[str]:
    """UI uchun: naqsh qanday fold bo'lishini ko'rsatadi (debug/tushuntirish)."""
    return [uz_fold(p) for p in split_pattern(pattern)]
