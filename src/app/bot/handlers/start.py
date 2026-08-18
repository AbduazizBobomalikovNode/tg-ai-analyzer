from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy import select

from app.db.base import session_scope
from app.db.models import Account, User
from app.i18n import SUPPORTED, Translator, t

router = Router(name="start")


def main_menu(_: Translator) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=_("btn.connect_account"), callback_data="acc:connect")],
            [InlineKeyboardButton(text=_("btn.my_accounts"), callback_data="acc:list")],
            [
                InlineKeyboardButton(text=_("btn.language"), callback_data="lang:choose"),
                InlineKeyboardButton(text=_("btn.help"), callback_data="help"),
            ],
        ]
    )


def lang_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=t("lang.name", loc), callback_data=f"lang:set:{loc}")]
            for loc in SUPPORTED
        ]
    )


@router.message(CommandStart())
async def cmd_start(message: Message, user: User, _: Translator) -> None:
    async with session_scope() as session:
        count = len(
            (await session.execute(select(Account.id).where(Account.user_id == user.id)))
            .scalars()
            .all()
        )

    text = _("start.greeting", name=message.from_user.full_name if message.from_user else "")
    if count == 0:
        text += "\n\n" + _("start.no_accounts")
    await message.answer(text, reply_markup=main_menu(_))


@router.message(Command("help"))
@router.callback_query(F.data == "help")
async def cmd_help(event: Message | CallbackQuery, _: Translator) -> None:
    text = _("help.text")
    if isinstance(event, CallbackQuery):
        await event.answer()
        if event.message:
            await event.message.answer(text)
    else:
        await event.answer(text)


@router.message(Command("lang"))
@router.callback_query(F.data == "lang:choose")
async def cmd_lang(event: Message | CallbackQuery, _: Translator) -> None:
    text = _("lang.choose")
    if isinstance(event, CallbackQuery):
        await event.answer()
        if event.message:
            await event.message.answer(text, reply_markup=lang_menu())
    else:
        await event.answer(text, reply_markup=lang_menu())


@router.callback_query(F.data.startswith("lang:set:"))
async def set_lang(cb: CallbackQuery, user: User) -> None:
    locale = (cb.data or "").rsplit(":", 1)[-1]
    async with session_scope() as session:
        db_user = await session.get(User, user.id)
        if db_user:
            db_user.locale = locale

    _ = Translator(locale)
    await cb.answer()
    if cb.message:
        await cb.message.answer(
            _("lang.changed", lang=t("lang.name", locale)), reply_markup=main_menu(_)
        )
