"""Avtomatlashtirish amallari: alert qoidasi, briefing sozlamasi, eslatma.

Uchala amal ham **yozish amali** hisoblanadi (akkaunt holatini o'zgartiradi)
va yozish bilan **bir xil** tasdiq oqimidan o'tadi (CLAUDE.md invariant 4):

    skill (skills_ext) → `build_*` (kodda tahlil + tekshiruv, DB'ga yozmaydi)
      → `write_tools.propose_automation` → `agent_actions(proposed)`
      → foydalanuvchi Actions sahifasida tasdiqlaydi (`actions.confirm_action`)
      → `actions.execute_action` → **`execute`** (shu modul) → INSERT/UPDATE

* Tasdiqsiz hech narsa yaratilmaydi: `execute` faqat `confirmed=True` bilan
  chaqiriladi (`actions.execute_action` tekshiradi). `autonomous` chat rejimi
  bu amallarga **tegishli emas** — u faqat Telegram yozish tool'lari uchun.
* `agent_actions` append-only: taklif `args` i (nima yaratilishi) taklif
  paytida muzlatiladi; tasdiqda aynan shu bajariladi. Egalik (akkaunt →
  foydalanuvchi, chat → akkaunt) va qiymatlar bajarishda **qayta** tekshiriladi.
* Hech narsa o'chirilmaydi: eslatmani bekor qilish — `status='cancelled'`,
  briefingni o'chirish — `briefing_enabled=false`.
* Telefon raqami hech qayerda yo'q — faqat chat/akkaunt id va sarlavhalar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Account, AgentAction, AgentRun, AlertKind, Chat
from app.logging import get_logger
from app.services import timewin as TW
from app.services import untrusted as U

log = get_logger(__name__)

ALERT_DEFAULT_COOLDOWN = 30
ALERT_DEFAULT_THRESHOLD = 0.80


class AutomationError(Exception):
    """Bajarish/taklif xatosi. `code` → `action.err.<code>` / foydalanuvchiga matn."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


# ─── niyat darvozalari (foydalanuvchining **o'z** matni ustida) ─────────────

ALERT_INTENT_RE = re.compile(
    r"(xabar\s*ber|ogohlantir\w*|habar\s*ber|signal\s*ber|bildir\w*|alert\w*|"
    r"уведом\w*|сообщи\w*|оповест\w*|предупред\w*|дай\s+знать|"
    r"notify|let\s+me\s+know|alert\s+me|ping\s+me|watch\s+for)",
    re.I,
)
BRIEFING_INTENT_RE = re.compile(
    r"(briefing\w*|брифинг\w*|brifing\w*|сводк\w*|daily\s+digest)", re.I
)
_QUOTED_WORD_RE = re.compile(
    r"(?:(?<=\s)|^)[\"'«“‘`]([^\"«»“”‘’`\n]{2,80}?)[\"'»”’`](?=\s|$|[,.!?])"
)
_OFF_RE = re.compile(
    r"(o'chir\w*|to'xtat\w*|kerak\s+emas|yuborma\w*|bekor\s+qil\w*|"
    r"отключ\w*|выключ\w*|не\s+присыла\w*|останов\w*|отмен\w*|"
    r"disable|turn\s+off|stop|cancel|unsubscribe)",
    re.I,
)
_ABOUT_RE = re.compile(
    r"(.{3,120}?)\s+(?:haqida|to'g'risida|bo'yicha)\b"
    r"|\b(?:про|о|об|насчёт|насчет)\s+(.{3,120}?)(?:[,.!?]|\s+(?:пиш|говор|напиш|сообщ)|$)"
    r"|\babout\s+(.{3,120}?)(?:[,.!?]|\s+(?:is|are|gets?)\b|$)",
    re.I,
)
_ALERT_LEAD_RE = re.compile(
    r"^(agar|qachonki|kimdir|biror\s+kim|if|when|whenever|someone|если|когда|кто-то|кто\s+нибудь)\s+",
    re.I,
)


def parse_alert_request(text: str) -> tuple[str, str] | None:
    """`(kind, pattern)` yoki `None`. Qo'shtirnoqdagi so'z → `keyword`;
    "X haqida yozishsa" → `semantic` (tabiiy tildagi tavsif)."""
    raw = (text or "").strip()
    # Apostrof o'zbek so'zining bir qismi ("so'z") — shuning uchun tirnoq
    # sifatida faqat so'z chegarasidagi juftni olamiz.
    for q in _QUOTED_WORD_RE.finditer(raw):
        word = q.group(1).strip()
        if word:
            return AlertKind.KEYWORD.value, word
    m = _ABOUT_RE.search(raw)
    if m:
        topic = next((g for g in m.groups() if g), "").strip(" ,.;:")
        for _ in range(3):  # "agar kimdir X haqida" → "X"
            topic = _ALERT_LEAD_RE.sub("", topic).strip()
        if len(topic) >= 3:
            return AlertKind.SEMANTIC.value, topic
    return None


def parse_briefing_request(text: str) -> tuple[bool, int | None]:
    """`(enabled, hour)` — o'chirish so'zi bo'lsa `enabled=False`; soat kodda."""
    from app.services import scheduled as SCH

    folded = TW.fold(text or "")
    enabled = not bool(_OFF_RE.search(folded))
    clock = SCH._clock(folded)
    hour = clock[0] if clock else None
    if hour is None and enabled:
        if SCH._MORNING.search(folded):
            hour = 8
        elif SCH._EVENING.search(folded):
            hour = 20
    return enabled, hour


# ─── taklif matni (Actions kartasida ko'rinadi, taklif paytida muzlaydi) ─────

_T = {
    "uz": {
        "alert_kw": "Alert qoidasi: «{p}» so'zi {scope} uchrasa botga xabar (cooldown {c} daq).",
        "alert_sem": "Alert qoidasi: {scope} «{p}» mavzusida yozishsa botga xabar "
        "(semantik, cooldown {c} daq).",
        "all_chats": "barcha chatlarda",
        "in_chat": "«{t}» chatida",
        "brief_on": "Kunlik briefing: yoqiladi, har kuni soat {h}:00 da ({tz}){chats}.",
        "brief_on_keep": "Kunlik briefing: yoqiladi (soat {h}:00, {tz}){chats}.",
        "brief_off": "Kunlik briefing: o'chiriladi.",
        "brief_chats": "; chatlar: {n} ta",
        "remind": "Eslatma {when} ({tz}) da botga keladi: «{t}»",
        "cancel": "Bekor qilinadigan eslatmalar ({n}):",
    },
    "ru": {
        "alert_kw": "Правило алерта: если слово «{p}» появится {scope}, бот сообщит "
        "(пауза {c} мин).",
        "alert_sem": "Правило алерта: если {scope} напишут про «{p}», бот сообщит "
        "(семантическое, пауза {c} мин).",
        "all_chats": "во всех чатах",
        "in_chat": "в чате «{t}»",
        "brief_on": "Ежедневный брифинг: включить, каждый день в {h}:00 ({tz}){chats}.",
        "brief_on_keep": "Ежедневный брифинг: включить ({h}:00, {tz}){chats}.",
        "brief_off": "Ежедневный брифинг: выключить.",
        "brief_chats": "; чатов: {n}",
        "remind": "Напоминание придёт в бот {when} ({tz}): «{t}»",
        "cancel": "Будут отменены напоминания ({n}):",
    },
    "en": {
        "alert_kw": "Alert rule: when the word «{p}» appears {scope}, the bot notifies you "
        "(cooldown {c} min).",
        "alert_sem": "Alert rule: when someone writes about «{p}» {scope}, the bot notifies you "
        "(semantic, cooldown {c} min).",
        "all_chats": "in any chat",
        "in_chat": "in «{t}»",
        "brief_on": "Daily briefing: turn on, every day at {h}:00 ({tz}){chats}.",
        "brief_on_keep": "Daily briefing: turn on ({h}:00, {tz}){chats}.",
        "brief_off": "Daily briefing: turn off.",
        "brief_chats": "; chats: {n}",
        "remind": "A reminder will arrive in the bot at {when} ({tz}): «{t}»",
        "cancel": "Reminders to cancel ({n}):",
    },
}


def _t(locale: str, key: str, **kw: Any) -> str:
    return _T.get(locale, _T["uz"])[key].format(**kw)


# ─── takliflar (DB'ga faqat o'qish; yozuv — `propose_automation`) ───────────


@dataclass(slots=True)
class Draft:
    """Tekshirilgan taklif: `tool`, muzlatiladigan `args`, odam o'qiydigan `summary`."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    summary: str = ""


async def build_alert(
    session: AsyncSession,
    *,
    account_id: int,
    kind: str,
    pattern: str,
    chat_id: int | None,
    cooldown_minutes: int | None,
    locale: str,
) -> Draft:
    from app.services import alerts as AL

    pattern = U.neutralize(pattern or "").strip()[: AL.PATTERN_MAX_CHARS]
    kind = kind if kind in (AlertKind.KEYWORD.value, AlertKind.SEMANTIC.value) else "keyword"
    cooldown = ALERT_DEFAULT_COOLDOWN if cooldown_minutes is None else int(cooldown_minutes)
    try:
        AL.validate(kind, pattern, ALERT_DEFAULT_THRESHOLD, cooldown)
    except AL.AlertError as exc:
        raise AutomationError(exc.code) from exc
    title = ""
    if chat_id is not None:
        chat = await session.get(Chat, int(chat_id))
        if chat is None or int(chat.account_id) != int(account_id):
            raise AutomationError("unknown_chat")
        title = U.attr(chat.title or "")
    scope = _t(locale, "in_chat", t=title) if chat_id is not None else _t(locale, "all_chats")
    key = "alert_kw" if kind == AlertKind.KEYWORD.value else "alert_sem"
    summary = _t(locale, key, p=pattern, scope=scope, c=cooldown)
    args = {
        "account_id": int(account_id),
        "kind": kind,
        "pattern": pattern,
        "chat_id": int(chat_id) if chat_id is not None else None,
        "chat_title": title or None,
        "cooldown_minutes": cooldown,
        "threshold": ALERT_DEFAULT_THRESHOLD,
        "label": pattern[:64],
        "summary": summary,
    }
    return Draft("create_alert", args, summary)


async def build_briefing(
    session: AsyncSession,
    *,
    account_id: int,
    enabled: bool,
    hour: int | None,
    chat_ids: list[int] | None,
    locale: str,
) -> Draft:
    acc = await session.get(Account, int(account_id))
    if acc is None:
        raise AutomationError("not_found")
    if hour is not None and not (0 <= int(hour) <= 23):
        raise AutomationError("bad_hour")
    if chat_ids:
        for cid in chat_ids:
            chat = await session.get(Chat, int(cid))
            if chat is None or int(chat.account_id) != int(account_id):
                raise AutomationError("unknown_chat")
    tz = acc.tz or TW.DEFAULT_TZ
    if not enabled:
        summary = _t(locale, "brief_off")
    else:
        h = int(hour) if hour is not None else int(acc.briefing_hour or 8)
        chats = _t(locale, "brief_chats", n=len(chat_ids)) if chat_ids else ""
        key = "brief_on" if hour is not None else "brief_on_keep"
        summary = _t(locale, key, h=f"{h:02d}", tz=tz, chats=chats)
    args: dict[str, Any] = {
        "account_id": int(account_id),
        "enabled": bool(enabled),
        "hour": int(hour) if hour is not None else None,
        "summary": summary,
    }
    if chat_ids:
        args["chats"] = sorted({int(c) for c in chat_ids})
    return Draft("set_briefing", args, summary)


def build_reminder(*, account_id: int, due: datetime, text: str, tz: str, locale: str) -> Draft:
    from app.services import scheduled as SCH

    body = U.neutralize(text or "").strip()[: SCH.REMINDER_TEXT_MAX]
    if not body:
        raise AutomationError("empty_text")
    when = SCH.format_local(due, tz)
    summary = _t(locale, "remind", when=when, tz=tz, t=body)
    args = {
        "account_id": int(account_id),
        "due_at": due.astimezone(UTC).isoformat(),
        "due_local": when,
        "tz": tz,
        "text": body,
        "summary": summary,
    }
    return Draft("remind", args, summary)


def build_cancel(
    *, account_id: int, jobs: list[tuple[int, str, str]], tz: str, locale: str
) -> Draft:
    """`jobs` — `(id, due_local, text)`; bo'sh bo'lsa chaqiruvchi taklif qilmaydi."""
    lines = [_t(locale, "cancel", n=len(jobs))]
    lines += [f"• #{jid} {when} — {U.neutralize(text)[:80]}" for jid, when, text in jobs]
    summary = "\n".join(lines)
    args = {
        "account_id": int(account_id),
        "job_ids": [int(j[0]) for j in jobs],
        "tz": tz,
        "summary": summary,
    }
    return Draft("cancel_reminder", args, summary)


# ─── bajarish (faqat tasdiqdan keyin, `actions.execute_action` chaqiradi) ────


async def execute(session: AsyncSession, action: AgentAction, run: AgentRun) -> int | None:
    """Tasdiqlangan avtomatlashtirish amali. Qaytaradi: yaratilgan qator id (yoki `None`).

    `run.account_id` — yagona haqiqat: taklifdagi `account_id` unga teng
    bo'lmasa rad etiladi (append-only args'ni boshqa akkauntga burib bo'lmaydi).
    """
    args = action.args or {}
    account_id = run.account_id
    if account_id is None or int(args.get("account_id") or 0) != int(account_id):
        raise AutomationError("not_found")
    acc = await session.get(Account, int(account_id))
    if acc is None or int(acc.user_id) != int(run.user_id):
        raise AutomationError("not_found")
    if action.tool == "create_alert":
        return await execute_create_alert(session, args, account_id=int(account_id))
    if action.tool == "set_briefing":
        return await execute_set_briefing(session, args, account=acc)
    if action.tool == "remind":
        return await execute_remind(session, args, account_id=int(account_id), action=action)
    if action.tool == "cancel_reminder":
        return await execute_cancel_reminder(session, args, account_id=int(account_id))
    raise AutomationError("unknown_tool", action.tool)


async def execute_create_alert(
    session: AsyncSession, args: dict[str, Any], *, account_id: int
) -> int:
    from app.services import alerts as AL

    try:
        rule = await AL.create_rule(
            session,
            account_id=account_id,
            kind=str(args.get("kind") or "keyword"),
            pattern=str(args.get("pattern") or ""),
            chat_id=int(args["chat_id"]) if args.get("chat_id") is not None else None,
            label=str(args.get("label") or ""),
            threshold=float(args.get("threshold") or ALERT_DEFAULT_THRESHOLD),
            cooldown_minutes=int(args.get("cooldown_minutes") or 0),
        )
    except AL.AlertError as exc:
        raise AutomationError(exc.code) from exc
    return int(rule.id)


async def execute_set_briefing(
    session: AsyncSession, args: dict[str, Any], *, account: Account
) -> int | None:
    from app.services import briefing as BR

    hour = args.get("hour")
    try:
        await BR.apply_settings(
            session,
            account,
            enabled=bool(args.get("enabled")),
            hour=int(hour) if hour is not None else None,
            chats=list(args["chats"]) if "chats" in args else None,
            set_chats="chats" in args,
        )
    except BR.BriefingSettingsError as exc:
        raise AutomationError(exc.code) from exc
    return None


async def execute_remind(
    session: AsyncSession, args: dict[str, Any], *, account_id: int, action: AgentAction
) -> int:
    from app.services import scheduled as SCH

    try:
        due = datetime.fromisoformat(str(args.get("due_at") or ""))
    except ValueError as exc:
        raise AutomationError("bad_due") from exc
    try:
        job = await SCH.create_reminder(
            session,
            account_id=account_id,
            due_at=due,
            text=str(args.get("text") or ""),
            tz=str(args.get("tz") or TW.DEFAULT_TZ),
            action_id=int(action.id) if action.id else None,
        )
    except SCH.ReminderError as exc:
        raise AutomationError(exc.code) from exc
    return int(job.id)


async def execute_cancel_reminder(
    session: AsyncSession, args: dict[str, Any], *, account_id: int
) -> int | None:
    from app.services import scheduled as SCH

    ids = [int(x) for x in (args.get("job_ids") or []) if isinstance(x, int)]
    n = await SCH.cancel_reminders(session, account_id=account_id, job_ids=ids)
    if n == 0:
        raise AutomationError("nothing_to_cancel")
    return None


__all__ = [
    "ALERT_INTENT_RE",
    "BRIEFING_INTENT_RE",
    "AutomationError",
    "Draft",
    "build_alert",
    "build_briefing",
    "build_cancel",
    "build_reminder",
    "execute",
    "parse_alert_request",
    "parse_briefing_request",
]
