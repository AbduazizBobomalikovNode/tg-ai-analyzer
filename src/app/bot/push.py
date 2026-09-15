"""Boshqaruv boti orqali **egasiga** push (3-bosqich).

Invariant: bot faqat o'z foydalanuvchisining shaxsiy chatiga yozadi —
`users.tg_user_id`. Ixtiyoriy peer yo'q, kanal yo'q, guruh yo'q. Bu modul
MTProto'ga umuman tegmaydi (u yerda `mtproto/guard.py` o'z deny-list'i bilan
ishlaydi); bu yerda Bot API va bitta manzil turi.

Nima uchun alohida modul: briefing yetkazish (`services/briefing.py`) ham,
alertlar (`services/alerts.py`) ham bir xil yo'ldan o'tsin — HTML
`parse_mode`, 4096 belgi qopqog'i, qator chegarasida bo'lish, xato
**ko'tarilmaydi** (fon vazifasi bitta xabar tufayli yiqilmasin).

Bo'lish qoidasi: matn qator chegarasida bo'linadi, shuning uchun renderer
har bir HTML tegini **bitta qator ichida** ochib-yopishi shart (bizning
rendererlar shunday qiladi) — aks holda bo'lakning HTML'i buzilardi.
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

#: Telegram Bot API `sendMessage` matn chegarasi.
TG_MAX_CHARS = 4096
#: Bitta push'da yuboriladigan maksimal bo'lak (spam qopqog'i).
MAX_CHUNKS = 6
#: Yozib bo'lmaydigan maxsus peer'lar (Telegram xizmat akkauntlari).
_FORBIDDEN_IDS = frozenset({777000, 42777})
#: Eng uzun HTML entity (`&thetasym;`) — kesish shu oynada tekshiriladi.
_ENTITY_MAX_CHARS = 12


class PushError(RuntimeError):
    """Faqat `assert_pushable` ko'taradi — `send_html` uni ham yutadi."""


def assert_pushable(user_tg_id: int) -> int:
    """Manzil haqiqiy foydalanuvchi id'simi (invariant: faqat egasiga).

    Bot id'si, xizmat akkauntlari va manfiy (chat/kanal) id'lar rad etiladi.
    """
    uid = int(user_tg_id)
    if uid <= 0:
        raise PushError("push target must be a positive user id")
    if uid in _FORBIDDEN_IDS:
        raise PushError("push target is a Telegram service account")
    if uid == get_settings().control_bot_id:
        raise PushError("push target is the control bot itself")
    return uid


def mini_app_link(path: str = "") -> str:
    """Mini App chuqur havolasi: `WEBAPP_BASE_URL` + `path`.

    `path` bo'sh bo'lsa — ildiz. Tashqi (`http`/`https` bilan boshlanadigan)
    qiymat berilsa o'zi qaytadi, shunda chaqiruvchi to'liq URL ham bera oladi.
    """
    if path.startswith(("http://", "https://")):
        return path
    base = get_settings().webapp_base_url.rstrip("/")
    return f"{base}/{path.lstrip('/')}" if path else base or "/"


def split_html(html: str, *, limit: int = TG_MAX_CHARS) -> list[str]:
    """Matnni ≤ `limit` bo'laklarga **qator chegarasida** bo'ladi.

    Bitta qatorning o'zi limitdan uzun bo'lsa — qattiq kesiladi, lekin
    ochilgan `<...>` tegining o'rtasidan emas: oxirgi yopilmagan `<` dan
    oldin kesiladi.
    """
    text = (html or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    buf: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal size
        if buf:
            chunks.append("\n".join(buf).strip("\n"))
            buf.clear()
            size = 0

    for raw_line in text.split("\n"):
        for piece in _hard_split(raw_line, limit):
            add = len(piece) + (1 if buf else 0)
            if size + add > limit:
                flush()
                add = len(piece)
            buf.append(piece)
            size += add
    flush()
    return [c for c in chunks if c]


def _hard_split(line: str, limit: int) -> list[str]:
    """Limitdan uzun bitta qator — teg **va** entity o'rtasidan kesmasdan.

    Ikki qopqoq, ikkalasi ham Telegram parseri uchun:

    * yopilmagan `<` dan oldin kesiladi — `<a href="…` ning yarmi keyingi
      bo'lakka o'tsa, xabar "can't parse entities" bilan rad etiladi;
    * yopilmagan `&` dan oldin kesiladi — `&amp;` ni `&am` + `p;` ga
      bo'lish ham aynan shu xatoni beradi (renderer'lar `html.escape`
      ishlatadi, ya'ni entity'lar **har doim** bor).
    """
    if len(line) <= limit:
        return [line]
    out: list[str] = []
    rest = line
    while len(rest) > limit:
        cut = limit
        head = rest[:cut]
        open_lt = head.rfind("<")
        if open_lt > 0 and head.rfind(">") < open_lt:
            cut = open_lt  # teg ichida kesmaymiz
        amp = rest.rfind("&", 0, cut)
        if amp > 0 and cut - amp <= _ENTITY_MAX_CHARS and ";" not in rest[amp:cut]:
            cut = amp  # entity ichida ham kesmaymiz
        if cut <= 0:  # bo'lakka joy qolmadi — qattiq kesamiz (cheksiz sikl yo'q)
            cut = limit
        out.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        out.append(rest)
    return out


async def send_html(
    user_tg_id: int,
    html: str,
    *,
    reply_markup: Any = None,
    bot: Any = None,
    disable_preview: bool = True,
) -> int | None:
    """Foydalanuvchiga HTML xabar yuboradi. **Hech qachon xato ko'tarmaydi.**

    Args:
        user_tg_id: `users.tg_user_id` — egasining shaxsiy chati.
        html: HTML `parse_mode` uchun tayyor matn (renderer escape qilgan).
        reply_markup: aiogram klaviaturasi (faqat **oxirgi** bo'lakka).
        bot: tayyor `aiogram.Bot` (test/qayta ishlatish uchun). Berilmasa
            shu chaqiruv uchun yaratiladi va oxirida yopiladi.
        disable_preview: havola ko'rinishini o'chirish.

    Returns:
        Birinchi yuborilgan xabar `message_id` si; yuborilmasa `None`.
    """
    try:
        uid = assert_pushable(user_tg_id)
    except PushError as exc:
        log.warning("push.blocked", reason=str(exc), tg_user_id=user_tg_id)
        return None

    chunks = split_html(html)
    if not chunks:
        return None
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
        log.warning("push.truncated", tg_user_id=uid, kept=MAX_CHUNKS)

    own_bot = bot is None
    if own_bot:
        from app.bot.main import build_bot

        bot = build_bot()

    first_id: int | None = None
    try:
        for i, chunk in enumerate(chunks):
            last = i == len(chunks) - 1
            msg = await bot.send_message(
                chat_id=uid,
                text=chunk,
                parse_mode="HTML",
                disable_web_page_preview=disable_preview,
                reply_markup=reply_markup if last else None,
            )
            if first_id is None:
                first_id = int(getattr(msg, "message_id", 0)) or None
        log.info("push.sent", tg_user_id=uid, chunks=len(chunks))
    except Exception as exc:  # fon vazifasi bitta xabar tufayli yiqilmasin
        log.warning("push.failed", tg_user_id=uid, error=str(exc)[:200])
    finally:
        if own_bot:
            try:
                await bot.session.close()
            except Exception as exc:
                log.debug("push.session_close_failed", error=str(exc)[:100])
    return first_id


__all__ = [
    "MAX_CHUNKS",
    "TG_MAX_CHARS",
    "PushError",
    "assert_pushable",
    "mini_app_link",
    "send_html",
    "split_html",
]
