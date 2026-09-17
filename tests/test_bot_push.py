"""`bot/push.py` — bo'lish, manzil invarianti, soxta Bot bilan yuborish.

Qo'riqlaydi:
  * push **faqat** foydalanuvchi id'siga ketadi (bot, xizmat akkaunt, manfiy id — yo'q);
  * har bo'lak ≤ 4096 va bo'linish **qator chegarasida**;
  * xato ko'tarilmaydi (fon vazifasi bitta xabar tufayli yiqilmasin);
  * `mini_app_link` — `WEBAPP_BASE_URL` ustida, ikkilangan `/` siz.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.bot import push
from app.config import get_settings


class _Msg:
    def __init__(self, mid: int) -> None:
        self.message_id = mid


class FakeBot:
    """Minimal aiogram Bot — tarmoq yo'q."""

    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.fail = fail
        self.closed = False
        self.session = type("S", (), {"close": self._close})()

    async def _close(self) -> None:
        self.closed = True

    async def send_message(self, **kwargs: Any) -> _Msg:
        if self.fail:
            raise RuntimeError("telegram is down")
        self.sent.append(kwargs)
        return _Msg(1000 + len(self.sent))


# ─── bo'lish ─────────────────────────────────────────────────────────────────


def test_split_short_text_is_one_chunk() -> None:
    assert push.split_html("hello") == ["hello"]
    assert push.split_html("   ") == []


def test_split_breaks_at_line_boundaries_under_limit() -> None:
    lines = [f"<b>line {i}</b> " + "x" * 200 for i in range(60)]
    chunks = push.split_html("\n".join(lines))
    assert len(chunks) > 1
    assert all(len(c) <= push.TG_MAX_CHARS for c in chunks)
    # hech bir qator ikkiga bo'linmagan: har bo'lakning qatorlari asl qatorlar
    rejoined = "\n".join(chunks).split("\n")
    assert rejoined == lines


def test_split_long_single_line_never_cuts_inside_a_tag() -> None:
    line = "y" * 4090 + '<a href="https://example.com/very/long/path">link</a>'
    chunks = push.split_html(line, limit=4096)
    assert len(chunks) > 1
    assert all(len(c) <= 4096 for c in chunks)
    for c in chunks:
        # ochilgan `<` yopilmagan holda bo'lak oxirida qolmasin
        assert c.rfind("<") <= c.rfind(">")


def test_split_long_single_line_never_cuts_inside_an_entity() -> None:
    """F2/H5: `&amp;` ni `&am` + `p;` ga bo'lish ham "can't parse entities" beradi.

    Renderer'lar `html.escape` ishlatadi, ya'ni uzun qatorda entity'lar
    deyarli har doim bor — kesish nuqtasi ularning ustiga tushishi mumkin.
    """
    line = "z" * 4090 + "&amp;" * 40
    chunks = push.split_html(line, limit=4096)
    assert len(chunks) > 1
    assert all(len(c) <= 4096 for c in chunks)
    for c in chunks:
        tail = c[c.rfind("&") :] if "&" in c else ""
        assert not tail or ";" in tail  # yarim qolgan entity yo'q
        assert not c.startswith(";")
    assert "".join(chunks) == line  # kontent butun


def test_hard_split_terminates_on_pathological_input() -> None:
    """Har belgi `&` bo'lsa ham sikl tugaydi (cut > 0 kafolati)."""
    out = push._hard_split("&" * 50, 10)
    assert "".join(out) == "&" * 50
    assert all(len(c) <= 10 for c in out)


def test_split_preserves_all_content() -> None:
    text = "\n".join("a" * 100 for _ in range(200))
    assert "".join(push.split_html(text)).replace("\n", "") == text.replace("\n", "")


# ─── manzil invarianti ───────────────────────────────────────────────────────


def test_assert_pushable_rejects_non_user_targets() -> None:
    assert push.assert_pushable(12345) == 12345
    for bad in (0, -100123456789, 777000, get_settings().control_bot_id):
        with pytest.raises(push.PushError):
            push.assert_pushable(bad)


async def test_send_html_blocked_target_sends_nothing() -> None:
    bot = FakeBot()
    assert await push.send_html(-500, "<b>x</b>", bot=bot) is None
    assert bot.sent == []


# ─── yuborish ────────────────────────────────────────────────────────────────


async def test_send_html_uses_html_parse_mode_and_returns_first_id() -> None:
    bot = FakeBot()
    mid = await push.send_html(42, "<b>hi</b>", bot=bot)
    assert mid == 1001
    assert bot.sent[0]["chat_id"] == 42
    assert bot.sent[0]["parse_mode"] == "HTML"
    assert bot.sent[0]["text"] == "<b>hi</b>"
    assert not bot.closed  # tashqaridan berilgan bot yopilmaydi


async def test_send_html_splits_and_attaches_markup_to_last_chunk_only() -> None:
    bot = FakeBot()
    markup = object()
    body = "\n".join("z" * 500 for _ in range(30))
    await push.send_html(42, body, bot=bot, reply_markup=markup)
    assert len(bot.sent) > 1
    assert [s["reply_markup"] for s in bot.sent[:-1]] == [None] * (len(bot.sent) - 1)
    assert bot.sent[-1]["reply_markup"] is markup


async def test_send_html_never_raises_on_api_error() -> None:
    bot = FakeBot(fail=True)
    assert await push.send_html(42, "<b>x</b>", bot=bot) is None


async def test_send_html_caps_chunk_count() -> None:
    bot = FakeBot()
    body = "\n".join("q" * 4000 for _ in range(30))
    await push.send_html(42, body, bot=bot)
    assert len(bot.sent) == push.MAX_CHUNKS


# ─── mini app havolasi ───────────────────────────────────────────────────────


def test_mini_app_link_joins_without_double_slash(monkeypatch: pytest.MonkeyPatch) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "webapp_base_url", "https://app.example.com/", raising=False)
    assert push.mini_app_link("/briefings/2026-09-14") == (
        "https://app.example.com/briefings/2026-09-14"
    )
    assert push.mini_app_link("briefings") == "https://app.example.com/briefings"
    assert push.mini_app_link("https://other/x") == "https://other/x"
