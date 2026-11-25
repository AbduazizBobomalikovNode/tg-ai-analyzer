"""Yagona sender-nom hal qiluvchi.

Nega alohida modul: akkaunt egasining o'z xabari bir necha joyda
(`services/tools.py`, `services/window_search.py`, `services/scale_search.py`,
va boshqalar) mustaqil ravishda `f"user:{sender_id}"` bo'lib chiqib ketardi —
lokal DB'da egasi ham oddiy jo'natuvchi kabi ko'rinardi, chunki `senders`
jadvali faqat **boshqalar** uchun to'ldiriladi (`ingestion.upsert_senders`).

Bu yerda ikkita narsa bitta joyda: (1) egasini aniqlash — `Message.out=True`
(asosiy, Telegram'ning o'zi belgilaydi) **yoki** `sender_id ==
account.tg_account_id` (zaxira — eski qatorlarda `out` ishonchsiz bo'lishi
mumkin, CLAUDE.md); (2) ko'rinadigan yorliq — uz "Men" / ru "Я" / en "Me",
akkaunt `label`i bo'lsa `"Men (<label>)"`.

Boshqa (notanish) jo'natuvchi hamon `user:<id>` bilan qaytadi — bu
**hujjatlashtirilgan fallback**, `senders` jadvalida ism topilmagan holat
uchun; testlar shuni qo'riqlaydi (`tests/test_senders.py`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Account

OWNER_LABELS: dict[str, str] = {"uz": "Men", "ru": "Я", "en": "Me"}
DEFAULT_LOCALE = "uz"

__all__ = [
    "DEFAULT_LOCALE",
    "OWNER_LABELS",
    "is_owner_message",
    "owner_info",
    "owner_label",
    "owner_marker",
    "relabel_owner_hits",
    "sender_label",
]


def _norm_locale(locale: str | None) -> str:
    short = (locale or "").split("-")[0].strip().lower()
    return short if short in OWNER_LABELS else DEFAULT_LOCALE


def is_owner_message(*, sender_id: int | None, out: bool, owner_tg_id: int | None) -> bool:
    """`out=True` — asosiy belgi (Telegram beradi). `sender_id == owner_tg_id`
    — zaxira: `out` bo'lmagan/eski qatorlar uchun ham egasi tanilsin.
    """
    if out:
        return True
    if sender_id is None or owner_tg_id is None:
        return False
    return int(sender_id) == int(owner_tg_id)


def owner_label(locale: str | None = DEFAULT_LOCALE, *, display_name: str | None = None) -> str:
    """ "Men" / "Я" / "Me" — `display_name` (akkaunt `label`i) bo'lsa qo'shiladi."""
    base = OWNER_LABELS[_norm_locale(locale)]
    name = (display_name or "").strip()
    return f"{base} ({name})" if name else base


def sender_label(
    sender_id: int | None,
    *,
    out: bool = False,
    owner_tg_id: int | None = None,
    owner_display_name: str | None = None,
    names: Mapping[int, str] | None = None,
    locale: str | None = DEFAULT_LOCALE,
) -> str:
    """`sender_id` → ko'rsatiladigan ism.

    Kanal posti (`sender_id` yo'q, `chat.type == "channel"`) bu yerga
    kirmaydi — chaqiruvchi o'zi ko'rib chat sarlavhasini beradi.
    """
    if is_owner_message(sender_id=sender_id, out=out, owner_tg_id=owner_tg_id):
        return owner_label(locale, display_name=owner_display_name)
    if not sender_id:
        return "—"
    resolved = (names or {}).get(int(sender_id))
    return resolved or f"user:{sender_id}"


async def owner_info(session: AsyncSession | None, account_id: int) -> tuple[int | None, str]:
    """`(tg_account_id, label)` — akkaunt egasi. Bitta arzon PK so'rov.

    `.all()` — bu modul bo'ylab (masalan `tools.sender_names`) ishlatiladigan
    naqsh, testlardagi soxta sessiyalar ham shuni qo'llab-quvvatlaydi.
    `session=None` — ba'zi tool testlarida DB umuman ulanmaydi (server
    qidiruvi lokalga tegmaydi); bunda egasi shunchaki aniqlanmaydi, xato
    emas — `sender_label`/`_hit_line` baribir eski `user:<id>` bilan ishlaydi.
    """
    if session is None:
        return None, ""
    rows = (
        await session.execute(
            select(Account.tg_account_id, Account.label).where(Account.id == account_id)
        )
    ).all()
    if not rows:
        return None, ""
    tg_id, label = rows[0]
    return (int(tg_id) if tg_id is not None else None), (label or "").strip()


def owner_marker(owner_tg_id: int | None) -> str:
    """`f"user:{id}"` — hydratsiya bosqichlari hali shu ko'rinishda yozadigan
    joylarda (masalan `deep_search._hydrate`, `tg_search.who_of`) render
    vaqtida almashtirish uchun matn shakli.
    """
    return f"user:{owner_tg_id}" if owner_tg_id is not None else ""


def relabel_owner_hits(
    hits: Iterable[Any],
    *,
    owner_tg_id: int | None,
    owner_display_name: str | None = None,
    locale: str | None = DEFAULT_LOCALE,
) -> None:
    """`.sender` allaqachon matn bo'lib hal qilingan ob'ektlarda (`S.Hit`,
    `tg_search.ServerHit`) egasini **render vaqtida** qayta belgilaydi.

    Ba'zi hydratsiya bosqichlari (`deep_search._hydrate`) `sender_id`/`out`
    ni emas, tayyor `f"user:{id}"` matnini yozadi — asl maydonlar bu yerga
    yetib kelmaydi. Shu sabab **aniq marker matni** bo'yicha almashtiriladi;
    boshqa foydalanuvchi hech qachon noto'g'ri belgilanmaydi.
    """
    marker = owner_marker(owner_tg_id)
    if not marker:
        return
    label = owner_label(locale, display_name=owner_display_name)
    for hit in hits:
        if getattr(hit, "sender", None) == marker:
            hit.sender = label
