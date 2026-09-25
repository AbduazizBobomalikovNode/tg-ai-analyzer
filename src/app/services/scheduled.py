"""Eslatmalar — `scheduled_jobs(type="reminder")` va ularni yetkazish.

Oqim (bitta, boshqasi yo'q):

    "ertaga 10 da eslat: hisobotni yubor"
      → skills_ext `remind` → `parse_due` (kodda, akkaunt `tz` sida)
      → `write_tools.propose_automation` → `agent_actions(proposed)`
      → foydalanuvchi tasdig'i (`actions.confirm_action`)
      → `automation.execute` → `create_reminder` (INSERT, `status="pending"`)
      → cron `worker.tasks.deliver_reminders` (har daqiqa) → `deliver_due`
      → `bot.push.send_html` — **faqat** akkaunt egasining boshqaruv boti
        chatiga (`accounts.user_id → users.tg_user_id`). MTProto yo'q, hech
        kimga Telegram akkaunt nomidan hech narsa yuborilmaydi.

Yetkazish briefing/alert naqshida: push'dan **oldin** shartli UPDATE
(`status='pending' → 'sending'`, `delivered_at=now`, `attempts+1`,
`RETURNING`) — ikki cron tiki bitta eslatmani ikki marta yubora olmaydi.
Xato bo'lsa `pending` ga qaytadi; `REMINDER_MAX_DELIVERY_ATTEMPTS` dan keyin
`failed` bo'lib yopiladi. Jarayon push va yakuniy yozuv orasida o'lsa qator
`sending` da qoladi — ya'ni **ko'pi bilan bir marta** (takroriy push emas).

Jim soatlar eslatmaga **tegishli emas**: foydalanuvchi aniq vaqtni o'zi
aytgan ("ertaga 7 da eslat" — jim oyna ichida bo'lsa ham shuni xohlagan).

Bekor qilish — `status='cancelled'` (UPDATE), qator o'chirilmaydi.
"""

from __future__ import annotations

import html
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Account, ScheduledJob, User
from app.logging import get_logger
from app.services import timewin as TW
from app.services import untrusted as U

log = get_logger(__name__)

REMINDER_TYPE = "reminder"
REMINDER_MAX_DELIVERY_ATTEMPTS = 3
REMINDER_TEXT_MAX = 1000
#: Juda uzoq kelajak — ehtimol noto'g'ri tahlil; rad etiladi.
REMINDER_MAX_AHEAD = timedelta(days=366)
#: Bitta cron yugurishida ko'riladigan eslatmalar.
DELIVER_BATCH = 100
#: Kun aytilib, soat aytilmasa.
DEFAULT_HOUR = 9
#: Akkauntda bir vaqtda kutayotgan eslatmalar qopqog'i.
MAX_PENDING_PER_ACCOUNT = 50

STATUS_PENDING = "pending"
STATUS_SENDING = "sending"
STATUS_SENT = "sent"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


class ReminderError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


# ─── vaqt ifodasi → UTC lahza (sof funksiya) ─────────────────────────────────

_NUM = TW._NUM_ALT
_UNIT_SHORT = (
    r"daqiqa\w*|daqqa\w*|minut\w*|мин\w*|minutes|minute|mins|min"
    r"|soat\w*|час\w*|hours|hour|hrs|hr"
    r"|kun\w*|дн\w*|день|days|day"
    r"|hafta\w*|недел\w*|weeks|week"
)
#: "2 soatdan keyin", "30 daqiqadan so'ng", "bir kundan keyin".
_REL_UZ = re.compile(rf"\b({_NUM})\s*({_UNIT_SHORT})(?:dan|ga)?\s+(?:keyin|so'ng|so'nggi)\b")
#: "через 2 часа", "через час", "через 15 минут".
_REL_RU = re.compile(rf"\bчерез\s+(?:({_NUM})\s+)?({_UNIT_SHORT})")
#: "in 2 hours", "in an hour", "2 hours from now", "in 10 min".
_REL_EN = re.compile(rf"\bin\s+({_NUM}|an?)\s+({_UNIT_SHORT})\b")
_REL_EN2 = re.compile(rf"\b({_NUM})\s+({_UNIT_SHORT})\s+(?:from\s+now|later)\b")

_DAY_WORDS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"\b(indinga|indin|послезавтра|day\s+after\s+tomorrow)\b"), 2),
    (re.compile(r"\b(ertaga|ertangi|ertasiga|завтра|tomorrow)\b"), 1),
    (re.compile(r"\b(bugun|сегодня|today|tonight)\b"), 0),
)
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
#: Nuqtali sana faqat yil bilan ("30.09.2026") — "10.30" soat bo'lishi mumkin.
_DOTTED = re.compile(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})[-\s]\s*({TW._MONTH_ALT})\b(?:\s+(\d{{4}}))?")

#: Soat: "10:30", "10.30", "soat 10", "10 da", "в 10", "at 10", "at 10pm", "10 ga".
_HHMM = re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b")
_SOAT = re.compile(r"\bsoat\s+([01]?\d|2[0-3])(?:[:.]([0-5]\d))?\b")
_UZ_DA = re.compile(r"\b([01]?\d|2[0-3])\s*(?:-?\s*)(?:da|ga|larda)\b")
_RU_V = re.compile(r"\bв\s+([01]?\d|2[0-3])(?:[:.]([0-5]\d))?(?:\s*час\w*)?\b")
_EN_AT = re.compile(r"\bat\s+([01]?\d|2[0-3])(?:[:.]([0-5]\d))?\s*(am|pm|a\.m\.|p\.m\.)?")
_AMPM = re.compile(r"\b([01]?\d)\s*(am|pm)\b")

_EVENING = re.compile(r"\b(kechqurun|kechki|oqshom|tushdan\s+keyin|вечер\w*|evening|tonight)\b")
_MORNING = re.compile(r"\b(ertalab|tong\w*|утр\w*|morning)\b")
_NIGHT = re.compile(r"\b(kechasi|ночью|at\s+night)\b")


def _n(raw: str | None, *, default: int = 1) -> int:
    if not raw:
        return default
    raw = raw.strip()
    if raw in ("a", "an"):
        return 1
    n = TW._num(raw)
    return n if n > 0 else default


def _delta(count: int, unit_raw: str) -> timedelta | None:
    unit = TW._unit_of(unit_raw)
    if unit == "minute":
        return timedelta(minutes=count)
    if unit == "hour":
        return timedelta(hours=count)
    if unit == "day":
        return timedelta(days=count)
    if unit == "week":
        return timedelta(weeks=count)
    return None


def _relative(folded: str) -> timedelta | None:
    for rx in (_REL_UZ, _REL_EN2):
        m = rx.search(folded)
        if m:
            return _delta(_n(m.group(1)), m.group(2))
    m = _REL_RU.search(folded)
    if m:
        return _delta(_n(m.group(1)), m.group(2))
    m = _REL_EN.search(folded)
    if m:
        return _delta(_n(m.group(1)), m.group(2))
    return None


def _explicit_date(folded: str, local_now: datetime) -> date | None:
    m = _ISO_DATE.search(folded)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = _DAY_MONTH.search(folded)
    if m:
        month = TW._MONTHS.get(m.group(2))
        year = int(m.group(3)) if m.group(3) else local_now.year
        if month:
            try:
                d = date(year, month, int(m.group(1)))
            except ValueError:
                return None
            if not m.group(3) and d < local_now.date():
                d = d.replace(year=year + 1)
            return d
    m = _DOTTED.search(folded)
    if m:
        year = int(m.group(3)) if m.group(3) else local_now.year
        try:
            d = date(year, int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
        if not m.group(3) and d < local_now.date():
            d = d.replace(year=year + 1)
        return d
    return None


def _clock(folded: str) -> tuple[int, int] | None:
    """Soat:daqiqa (24 soatlik). Kun qismi so'zi ("kechqurun 7") hisobga olinadi."""
    hour: int | None = None
    minute = 0
    pm: bool | None = None
    m = _HHMM.search(folded)
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        for rx in (_SOAT, _RU_V):
            m = rx.search(folded)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2) or 0)
                break
        if hour is None:
            m = _EN_AT.search(folded)
            if m:
                hour = int(m.group(1))
                minute = int(m.group(2) or 0)
                if m.group(3):
                    pm = m.group(3).startswith("p")
        if hour is None:
            m = _AMPM.search(folded)
            if m:
                hour = int(m.group(1))
                pm = m.group(2) == "pm"
        if hour is None:
            m = _UZ_DA.search(folded)
            if m:
                hour = int(m.group(1))
    if hour is None:
        return None
    if pm is True and hour < 12:
        hour += 12
    elif pm is False and hour == 12:
        hour = 0
    elif pm is None and hour < 12:
        # "kechqurun 7" = 19:00; "kechasi 11" = 23:00, lekin "kechasi 2" = 02:00.
        if _EVENING.search(folded) or (_NIGHT.search(folded) and hour >= 6):
            hour += 12
    return hour, minute


def parse_due(
    text: str, *, tz: str | None = TW.DEFAULT_TZ, now: datetime | None = None
) -> datetime | None:
    """Eslatma vaqti (UTC) yoki `None` (vaqt topilmadi / o'tmishda / juda uzoq).

    Qoidalar: nisbiy ("2 soatdan keyin") birinchi; keyin kun (bugun/ertaga/
    indinga/aniq sana) + soat. Kun bor, soat yo'q → `DEFAULT_HOUR`. Soat bor,
    kun yo'q → bugun, o'tib ketgan bo'lsa ertaga. Hammasi akkaunt `tz` sida.
    """
    zone = TW.zone(tz)
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    local_now = now_utc.astimezone(zone)
    folded = TW.fold(text or "")
    rel = _relative(folded)
    if rel is not None:
        due = now_utc + rel
    else:
        day: date | None = _explicit_date(folded, local_now)
        if day is None:
            for rx, offset in _DAY_WORDS:
                if rx.search(folded):
                    day = local_now.date() + timedelta(days=offset)
                    break
        # Sana matni soat deb o'qilmasin ("12.10.2026" → 12:10 emas).
        clock = _clock(_ISO_DATE.sub(" ", _DOTTED.sub(" ", _DAY_MONTH.sub(" ", folded))))
        if day is None and clock is None:
            return None
        hour, minute = clock if clock is not None else (DEFAULT_HOUR, 0)
        if clock is None and _EVENING.search(folded):
            hour = 19
        if day is None:
            day = local_now.date()
            candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
            if candidate <= local_now:
                day = day + timedelta(days=1)
        due = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone).astimezone(UTC)
    if due <= now_utc or due - now_utc > REMINDER_MAX_AHEAD:
        return None
    return due


def parse_iso_due(raw: str, *, tz: str | None, now: datetime | None = None) -> datetime | None:
    """Planner argumenti ISO bo'lsa (`2026-09-27T10:00`) — naive = akkaunt `tz`."""
    try:
        dt = datetime.fromisoformat(str(raw).strip())
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TW.zone(tz))
    dt = dt.astimezone(UTC)
    now_utc = (now or datetime.now(UTC)).astimezone(UTC)
    if dt <= now_utc or dt - now_utc > REMINDER_MAX_AHEAD:
        return None
    return dt


#: Eslatma matnini ajratishda tashlanadigan buyruq so'zlari.
_INTENT_WORDS = re.compile(
    r"\b(eslat(ib)?(\s*(qo'y|ber))?\w*|eslatma\w*|напомни\w*|напоминани\w*|"
    r"remind(\s+me)?(\s+to|\s+about)?|reminder|iltimos|пожалуйста|please)\b",
    re.I,
)


#: Ajratuvchi ikki nuqta — "10:30" dagi emas (ikki tomonida ham raqam bo'lmasin).
_SEP_COLON_RE = re.compile(r"(?<!\d):|:(?!\d)")
_TEXT_STRIP_RES: tuple[re.Pattern[str], ...] = (
    _REL_UZ,
    _REL_RU,
    _REL_EN,
    _REL_EN2,
    _ISO_DATE,
    _DOTTED,
    _DAY_MONTH,
    _HHMM,
    _SOAT,
    _RU_V,
    _EN_AT,
    _AMPM,
    _UZ_DA,
    _EVENING,
    _MORNING,
    *(r for r, _ in _DAY_WORDS),
    _INTENT_WORDS,
)
_LEAD_FILLER_RE = re.compile(r"^(?:to|about|that|ki|что|чтобы|о|про)\s+", re.I)


def extract_text(text: str) -> str:
    """Eslatma matni: ajratuvchi `:` dan keyingi qism; bo'lmasa buyruq/vaqt so'zlarisiz qoldiq.

    Naqshlar fold qilingan matnda qidiriladi, kesish esa **asl** matndan
    (harf registri saqlansin) — fold uzunlikni o'zgartirsa, fold matni olinadi.
    """
    raw = (text or "").strip()
    m = _SEP_COLON_RE.search(raw)
    if m:
        tail = raw[m.end() :].strip()
        if tail:
            return tail[:REMINDER_TEXT_MAX]
    folded = TW.fold(raw)
    source = raw if len(folded) == len(raw) else folded
    keep = [True] * len(folded)
    for rx in _TEXT_STRIP_RES:
        for hit in rx.finditer(folded):
            for i in range(hit.start(), hit.end()):
                keep[i] = False
    cut = "".join(ch if keep[i] else " " for i, ch in enumerate(source))
    cut = re.sub(r"\s+", " ", cut).strip(" ,.;:-—")
    cut = _LEAD_FILLER_RE.sub("", cut).strip(" ,.;:-—")
    return cut[:REMINDER_TEXT_MAX]


def format_local(due: datetime, tz: str | None) -> str:
    return due.astimezone(TW.zone(tz)).strftime("%Y-%m-%d %H:%M")


# ─── DB: yaratish / bekor qilish (faqat `automation.execute` chaqiradi) ─────


async def create_reminder(
    session: AsyncSession,
    *,
    account_id: int,
    due_at: datetime,
    text: str,
    tz: str,
    action_id: int | None = None,
    now: datetime | None = None,
) -> ScheduledJob:
    now_utc = now or datetime.now(UTC)
    if due_at.tzinfo is None:
        raise ReminderError("bad_due")
    if due_at <= now_utc:
        raise ReminderError("due_passed")
    body = (text or "").strip()
    if not body:
        raise ReminderError("empty_text")
    pending = await pending_reminders(session, account_id)
    if len(pending) >= MAX_PENDING_PER_ACCOUNT:
        raise ReminderError("too_many")
    job = ScheduledJob(
        account_id=int(account_id),
        chat_id=None,
        type=REMINDER_TYPE,
        payload={"text": body[:REMINDER_TEXT_MAX], "tz": tz, "action_id": action_id},
        run_at=due_at,
        status=STATUS_PENDING,
        attempts=0,
    )
    session.add(job)
    await session.flush()
    log.info("reminder.created", job_id=job.id, account_id=account_id, due=due_at.isoformat())
    return job


async def pending_reminders(session: AsyncSession, account_id: int) -> list[ScheduledJob]:
    rows = await session.execute(
        select(ScheduledJob)
        .where(
            ScheduledJob.account_id == int(account_id),
            ScheduledJob.type == REMINDER_TYPE,
            ScheduledJob.status == STATUS_PENDING,
        )
        .order_by(ScheduledJob.run_at, ScheduledJob.id)
    )
    return list(rows.scalars().all())


async def cancel_reminders(session: AsyncSession, *, account_id: int, job_ids: list[int]) -> int:
    """`pending` → `cancelled` (faqat shu akkauntniki). Qaytaradi: nechta."""
    if not job_ids:
        return 0
    res = await session.execute(
        update(ScheduledJob)
        .where(
            ScheduledJob.id.in_([int(j) for j in job_ids]),
            ScheduledJob.account_id == int(account_id),
            ScheduledJob.type == REMINDER_TYPE,
            ScheduledJob.status == STATUS_PENDING,
        )
        .values(status=STATUS_CANCELLED)
        .execution_options(synchronize_session=False)
    )
    n = int(getattr(res, "rowcount", 0) or 0)
    log.info("reminder.cancelled", account_id=account_id, n=n)
    return n


# ─── yetkazish (cron) ────────────────────────────────────────────────────────

_TITLE = {"uz": "Eslatma", "ru": "Напоминание", "en": "Reminder"}


def render_html(text: str, *, locale: str = "uz") -> str:
    """Push matni — foydalanuvchining o'z matni, baribir `neutralize` + `escape`."""
    title = _TITLE.get(locale, _TITLE["uz"])
    body = html.escape(U.neutralize(text or "")[:REMINDER_TEXT_MAX])
    return f"⏰ <b>{title}</b>\n{body}"


@dataclass(frozen=True, slots=True)
class DeliveryOutcome:
    job_id: int
    status: str  # sent | retry | failed | skipped
    reason: str = ""


SendFn = Callable[[int, str], Awaitable[int | None]]


async def _owner(session: AsyncSession, account_id: int) -> tuple[int | None, str]:
    """Push manzili: **faqat** akkaunt egasining boshqaruv boti chati (+ til)."""
    row = (
        await session.execute(
            select(User.tg_user_id, User.locale)
            .join(Account, Account.user_id == User.id)
            .where(Account.id == int(account_id))
        )
    ).first()
    if not row or not row[0]:
        return None, "uz"
    return int(row[0]), str(row[1] or "uz")


async def due_ids(session: AsyncSession, now: datetime, *, limit: int = DELIVER_BATCH) -> list[int]:
    rows = await session.execute(
        select(ScheduledJob.id)
        .where(
            ScheduledJob.type == REMINDER_TYPE,
            ScheduledJob.status == STATUS_PENDING,
            ScheduledJob.run_at <= now,
        )
        .order_by(ScheduledJob.run_at, ScheduledJob.id)
        .limit(limit)
    )
    return [int(r[0]) for r in rows.all()]


async def claim(session: AsyncSession, job_id: int, now: datetime) -> int | None:
    """Push'dan **oldin** band qilish: shartli UPDATE, `RETURNING attempts`.

    `None` — qatorni boshqa tik oldi (yoki bekor qilindi) — hech narsa yuborilmaydi.
    """
    got = (
        await session.execute(
            update(ScheduledJob)
            .where(
                ScheduledJob.id == int(job_id),
                ScheduledJob.type == REMINDER_TYPE,
                ScheduledJob.status == STATUS_PENDING,
                ScheduledJob.run_at <= now,
            )
            .values(
                status=STATUS_SENDING,
                delivered_at=now,
                attempts=ScheduledJob.attempts + 1,
            )
            .returning(ScheduledJob.attempts)
            .execution_options(synchronize_session=False)
        )
    ).scalar_one_or_none()
    return int(got) if got is not None else None


async def _finish(
    session: AsyncSession, job_id: int, *, status: str, error: str | None, clear_delivered: bool
) -> None:
    values: dict[str, Any] = {"status": status, "error": (error or None)}
    if clear_delivered:
        values["delivered_at"] = None
    await session.execute(
        update(ScheduledJob)
        .where(ScheduledJob.id == int(job_id))
        .values(**values)
        .execution_options(synchronize_session=False)
    )


async def deliver_one(
    session: AsyncSession, job_id: int, *, now: datetime, send: SendFn | None = None
) -> DeliveryOutcome:
    """Bitta eslatma: band qilish → push → yakun (yoki bo'shatish / voz kechish)."""
    attempt = await claim(session, job_id, now)
    if attempt is None:
        return DeliveryOutcome(job_id, "skipped", "claimed_elsewhere")
    job = await session.get(ScheduledJob, int(job_id))
    if job is None:  # pragma: no cover — claim hozirgina ko'rdi
        return DeliveryOutcome(job_id, "skipped", "gone")
    owner, locale = await _owner(session, int(job.account_id))
    if owner is None:
        await _finish(
            session, job_id, status=STATUS_FAILED, error="no_owner", clear_delivered=False
        )
        return DeliveryOutcome(job_id, "failed", "no_owner")
    payload = job.payload if isinstance(job.payload, dict) else {}
    if send is None:
        from app.bot.push import send_html

        send = send_html
    try:
        msg_id = await send(owner, render_html(str(payload.get("text") or ""), locale=locale))
        err = None if msg_id else "push_failed"
    except Exception as exc:  # send_html o'zi yutadi — bu ikkinchi chiziq
        err = str(exc)[:300] or type(exc).__name__
    if err is None:
        await _finish(session, job_id, status=STATUS_SENT, error=None, clear_delivered=False)
        log.info("reminder.delivered", job_id=job_id, account_id=job.account_id)
        return DeliveryOutcome(job_id, "sent")
    if attempt < REMINDER_MAX_DELIVERY_ATTEMPTS:
        await _finish(session, job_id, status=STATUS_PENDING, error=err, clear_delivered=True)
        return DeliveryOutcome(job_id, "retry", err)
    await _finish(
        session,
        job_id,
        status=STATUS_FAILED,
        error=f"gave_up_after_{attempt}: {err}"[:500],
        clear_delivered=False,
    )
    log.warning("reminder.gave_up", job_id=job_id, attempts=attempt, error=err)
    return DeliveryOutcome(job_id, "failed", err)


__all__ = [
    "REMINDER_MAX_DELIVERY_ATTEMPTS",
    "REMINDER_TYPE",
    "DeliveryOutcome",
    "ReminderError",
    "cancel_reminders",
    "claim",
    "create_reminder",
    "deliver_one",
    "due_ids",
    "extract_text",
    "format_local",
    "parse_due",
    "parse_iso_due",
    "pending_reminders",
    "render_html",
]
