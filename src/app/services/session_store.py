"""MTProto session'ni DB'ga shifrlab yozish / o'qish.

Faqat shu modul `Account.session_*` ustunlariga tegadi. Session string RAM'dan
tashqariga faqat `envelope.seal()` dan o'tib chiqadi; log'ga hech qachon
tushmaydi (`app.logging._redact` — `session`, `session_string` kalitlari).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.crypto import SealedSecret, account_aad, seal, unseal_str
from app.db.models import Account, AccountStatus
from app.logging import get_logger

log = get_logger(__name__)


def phone_hash(phone: str) -> str:
    """Xom raqam saqlanmaydi — faqat sha256 (rejaning 9-bo'limi)."""
    digits = "".join(ch for ch in phone if ch.isdigit())
    return hashlib.sha256(digits.encode()).hexdigest()


def store_session(account: Account, session_string: str) -> None:
    """Session'ni akkauntga bog'lab shifrlaydi. `account.id` bo'lishi shart (flush qilingan)."""
    if account.id is None:
        raise ValueError("account.id yo'q — avval flush qiling (AAD id'ga bog'lanadi)")
    sealed = seal(get_settings().master_key, session_string, aad=account_aad(account.id))
    account.session_ciphertext = sealed.ciphertext
    account.session_wrapped_dek = sealed.wrapped_dek
    account.status = AccountStatus.ACTIVE
    account.last_seen_at = datetime.now(UTC)
    log.info("session.stored", account_id=account.id)


def load_session(account: Account) -> str | None:
    """Shifrlangan session'ni ochadi. Yo'q bo'lsa None."""
    if not account.session_ciphertext or not account.session_wrapped_dek:
        return None
    sealed = SealedSecret(
        wrapped_dek=account.session_wrapped_dek,
        ciphertext=account.session_ciphertext,
        aad=account_aad(account.id),
    )
    return unseal_str(get_settings().master_key, sealed)


async def revoke_session(session: AsyncSession, account: Account, *, status: str) -> None:
    """Session'ni o'chiradi (masalan AuthKeyUnregistered). Akkaunt yozuvi qoladi."""
    account.session_ciphertext = None
    account.session_wrapped_dek = None
    account.status = status
    await session.flush()
    log.warning("session.revoked", account_id=account.id, status=status)
