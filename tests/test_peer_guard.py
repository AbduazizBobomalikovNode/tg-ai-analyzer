from __future__ import annotations

import pytest

from app.mtproto.guard import (
    TELEGRAM_SERVICE_ID,
    PeerBlocked,
    assert_writable,
    filter_visible,
    is_protected,
)

CONTROL_BOT_ID = 123456  # tests/conftest.py dagi qiymat


def test_control_bot_is_protected() -> None:
    assert is_protected(CONTROL_BOT_ID)


def test_agent_cannot_write_to_control_bot() -> None:
    with pytest.raises(PeerBlocked, match="boshqaruv boti"):
        assert_writable(CONTROL_BOT_ID)


def test_negative_peer_id_also_blocked() -> None:
    """Kanal/guruh id'lari manfiy keladi — abs() bilan solishtiriladi."""
    with pytest.raises(PeerBlocked):
        assert_writable(-CONTROL_BOT_ID)


def test_telegram_service_blocked() -> None:
    with pytest.raises(PeerBlocked, match="servis"):
        assert_writable(TELEGRAM_SERVICE_ID)


def test_normal_peer_allowed() -> None:
    assert_writable(-1001234567890)  # istisno tashlamasligi kerak


def test_extra_protected_peers() -> None:
    with pytest.raises(PeerBlocked, match="himoyalangan"):
        assert_writable(999888, extra=frozenset({999888}))


def test_filter_visible_hides_control_bot() -> None:
    dialogs = [
        {"id": CONTROL_BOT_ID, "title": "control bot"},
        {"id": -1001234567890, "title": "my channel"},
        {"id": TELEGRAM_SERVICE_ID, "title": "Telegram"},
    ]
    visible = filter_visible(dialogs, lambda d: int(d["id"]))
    assert [d["title"] for d in visible] == ["my channel"]
