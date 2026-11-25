"""`bot/main.py::run_bot` — auth muvaffaqiyatsizligida qulay chiqish.

Qo'riqlaydi:
  * noto'g'ri token (`TelegramUnauthorizedError`) yoki tarmoq xatosi
    (`TelegramNetworkError`) da process yiqilmaydi — exponential backoff
    (30s -> 300s cap) bilan `get_me()` qayta uriniladi;
  * muvaffaqiyatdan keyin polling avvalgidek boshlanadi (mavjud xatti-harakat
    o'zgarmagan);
  * to'g'ri token yo'li o'zgarmagan (birinchi urinishda muvaffaqiyat, sleep yo'q).
"""

from __future__ import annotations

from typing import Any

import pytest
from aiogram.exceptions import TelegramNetworkError, TelegramUnauthorizedError

from app.bot import main as bot_main


class _Me:
    def __init__(self, *, id: int, username: str) -> None:
        self.id = id
        self.username = username


class FakeBot:
    """Soxta `aiogram.Bot` — `get_me()` berilgan effektlar ro'yxatini ketma-ket ishlatadi."""

    def __init__(self, effects: list[Any]) -> None:
        self._effects = list(effects)
        self.get_me_calls = 0
        self.webhook_deleted = False

    async def get_me(self) -> _Me:
        self.get_me_calls += 1
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    async def delete_webhook(self, *, drop_pending_updates: bool) -> None:
        self.webhook_deleted = True


class FakeDispatcher:
    def __init__(self) -> None:
        self.polling_started_with: Any = None

    async def start_polling(self, bot: Any) -> None:
        self.polling_started_with = bot


@pytest.fixture
def fake_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """`asyncio.sleep` ni yozib oladi va darhol qaytadi — testda haqiqiy kutish yo'q."""
    calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        calls.append(delay)

    monkeypatch.setattr(bot_main.asyncio, "sleep", _fake_sleep)
    return calls


def _patch_builders(monkeypatch: pytest.MonkeyPatch, bot: FakeBot, dp: FakeDispatcher) -> None:
    monkeypatch.setattr(bot_main, "build_bot", lambda: bot)
    monkeypatch.setattr(bot_main, "build_dispatcher", lambda: dp)


# ─── auth muvaffaqiyatsizligi — backoff va tiklanish ────────────────────────


@pytest.mark.asyncio
async def test_wait_for_bot_ready_retries_with_exponential_backoff(
    fake_sleep: list[float],
) -> None:
    me = _Me(id=123456, username="test_bot")
    bot = FakeBot(
        [
            TelegramUnauthorizedError(method=None, message="Unauthorized"),  # type: ignore[arg-type]
            TelegramNetworkError(method=None, message="network down"),  # type: ignore[arg-type]
            me,
        ]
    )

    result = await bot_main._wait_for_bot_ready(bot)  # type: ignore[arg-type]

    assert result is me
    assert bot.get_me_calls == 3
    assert fake_sleep == [30.0, 60.0]


@pytest.mark.asyncio
async def test_run_bot_starts_polling_after_transient_auth_failures(
    monkeypatch: pytest.MonkeyPatch, fake_sleep: list[float]
) -> None:
    me = _Me(id=123456, username="test_bot")  # conftest: CONTROL_BOT_ID=123456
    bot = FakeBot(
        [
            TelegramUnauthorizedError(method=None, message="Unauthorized"),  # type: ignore[arg-type]
            me,
        ]
    )
    dp = FakeDispatcher()
    _patch_builders(monkeypatch, bot, dp)

    await bot_main.run_bot()

    assert fake_sleep == [30.0]
    assert bot.webhook_deleted is True
    assert dp.polling_started_with is bot


# ─── to'g'ri token yo'li — o'zgarmagan ───────────────────────────────────────


@pytest.mark.asyncio
async def test_run_bot_valid_token_path_unchanged(
    monkeypatch: pytest.MonkeyPatch, fake_sleep: list[float]
) -> None:
    me = _Me(id=123456, username="test_bot")
    bot = FakeBot([me])
    dp = FakeDispatcher()
    _patch_builders(monkeypatch, bot, dp)

    await bot_main.run_bot()

    assert bot.get_me_calls == 1
    assert fake_sleep == []  # hech qanday kutish bo'lmagan
    assert bot.webhook_deleted is True
    assert dp.polling_started_with is bot
