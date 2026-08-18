"""User ↔ Account bog'lash (login natijasini DB'ga tushirish)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Account, AccountStatus, User
from app.logging import get_logger
from app.services.auth_flow import AuthError, LoggedIn
from app.services.session_store import store_session

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Linked:
    user_id: int
    account_id: int
    created_user: bool


async def get_or_create_user(session: AsyncSession, tg_user_id: int, username: str | None) -> User:
    user = (
        await session.execute(select(User).where(User.tg_user_id == tg_user_id))
    ).scalar_one_or_none()
    if user is None:
        user = User(tg_user_id=tg_user_id, username=username)
        session.add(user)
        await session.flush()
        log.info("user.created", user_id=user.id, tg_user_id=tg_user_id)
    elif username and user.username != username:
        user.username = username
    return user


async def link_account(
    session: AsyncSession,
    result: LoggedIn,
    *,
    owner_user_id: int | None = None,
    label: str = "",
) -> Linked:
    """Login natijasini akkaunt sifatida bog'laydi.

    * `owner_user_id` berilsa (web'da allaqachon kirgan user) — akkaunt shu userga
      qo'shiladi (multi-account). Aks holda user `tg_user_id` bo'yicha topiladi/yaratiladi.
    * `ALLOWED_USER_IDS` bo'sh bo'lmasa — faqat ro'yxatdagilar (bot bilan bir xil qoida).
    * Bir user uchun bir xil `tg_account_id` — mavjud yozuv yangilanadi (re-login).
    * `MAX_ACCOUNTS` limiti.
    """
    s = get_settings()
    allowed = s.allowed_users
    if allowed and result.tg_user_id not in allowed and owner_user_id is None:
        log.warning("auth.not_allowed", tg_user_id=result.tg_user_id)
        raise AuthError("not_allowed")

    created_user = False
    if owner_user_id is not None:
        user = await session.get(User, owner_user_id)
        if user is None:
            raise AuthError("flow_expired")
    else:
        existing = (
            await session.execute(select(User).where(User.tg_user_id == result.tg_user_id))
        ).scalar_one_or_none()
        created_user = existing is None
        user = existing or await get_or_create_user(session, result.tg_user_id, result.username)

    account = (
        await session.execute(
            select(Account).where(
                Account.user_id == user.id, Account.tg_account_id == result.tg_user_id
            )
        )
    ).scalar_one_or_none()

    if account is None:
        count = (
            (await session.execute(select(Account.id).where(Account.user_id == user.id)))
            .scalars()
            .all()
        )
        if len(count) >= s.max_accounts:
            raise AuthError("max_accounts")
        account = Account(
            user_id=user.id,
            tg_account_id=result.tg_user_id,
            label=label or (f"@{result.username}" if result.username else result.first_name),
            phone_hash=result.phone_hash,
            status=AccountStatus.PENDING,
        )
        session.add(account)
        await session.flush()  # id kerak — AAD shunga bog'lanadi

    account.phone_hash = result.phone_hash
    if label:
        account.label = label
    store_session(account, result.session_string)
    await session.flush()

    log.info(
        "account.linked",
        user_id=user.id,
        account_id=account.id,
        tg_account_id=result.tg_user_id,
        created_user=created_user,
    )
    return Linked(user_id=user.id, account_id=account.id, created_user=created_user)


async def list_accounts(session: AsyncSession, user_id: int) -> list[Account]:
    return list(
        (
            await session.execute(
                select(Account).where(Account.user_id == user_id).order_by(Account.id)
            )
        )
        .scalars()
        .all()
    )
