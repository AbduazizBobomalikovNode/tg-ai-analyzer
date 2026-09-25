"""Ingest normalizatsiyasi (1-bosqich): sanitizatsiya, forward, topic, til, link.

DB va Telegram'siz — hammasi sof funksiyalar ustida.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from telethon.tl.types import Message as TlMessage
from telethon.tl.types import PeerChannel

from app.db.models import ChatType
from app.services import ingestion as ing


def _tl_message(**kw: Any) -> TlMessage:
    base: dict[str, Any] = {
        "id": 42,
        "peer_id": PeerChannel(channel_id=100),
        "date": datetime(2026, 8, 18, 10, 0, tzinfo=UTC),
        "message": "salom",
    }
    base.update(kw)
    return TlMessage(**base)


# ─── sanitizatsiya ───────────────────────────────────────────────────────────


def test_text_is_sanitized_before_storage() -> None:
    """Ko'rinmas belgilar DB'ga umuman yozilmaydi (saqlash chegarasi)."""
    hidden = "to​lov‮ kerak" + chr(0xE0041)
    row = ing.message_to_row(1, _tl_message(message=hidden))
    assert row is not None
    assert row["text"] == "tolov kerak"


def test_message_with_only_invisible_text_and_no_media_is_skipped() -> None:
    assert ing.message_to_row(1, _tl_message(message="​﻿")) is None


# ─── forward manbasi ─────────────────────────────────────────────────────────


def test_forward_fields_are_filled_without_extra_rpc() -> None:
    fwd = SimpleNamespace(
        from_id=PeerChannel(channel_id=1234567890),
        from_name=None,
        channel_post=777,
        date=datetime(2026, 8, 17, 9, 0, tzinfo=UTC),
    )
    row = ing.message_to_row(1, _tl_message(fwd_from=fwd))
    assert row is not None
    assert row["fwd_from_peer_id"] == -1001234567890  # to'liq peer id
    assert row["fwd_from_msg_id"] == 777
    assert row["fwd_date"] == fwd.date


def test_forward_from_name_only_leaves_peer_empty() -> None:
    """Maxfiylik yopiq bo'lsa Telegram faqat `from_name` beradi — u saqlanmaydi."""
    fwd = SimpleNamespace(from_id=None, from_name="Nurbek", channel_post=None, date=None)
    row = ing.message_to_row(1, _tl_message(fwd_from=fwd))
    assert row is not None
    assert row["fwd_from_peer_id"] is None
    assert "Nurbek" not in str(row)


def test_no_forward_means_null_columns() -> None:
    row = ing.message_to_row(1, _tl_message())
    assert row is not None
    assert row["fwd_from_peer_id"] is None and row["fwd_date"] is None


# ─── forum topic / thread ────────────────────────────────────────────────────


def test_top_msg_id_from_reply_to_top_id() -> None:
    reply_to = SimpleNamespace(reply_to_msg_id=55, reply_to_top_id=10, forum_topic=True)
    row = ing.message_to_row(1, _tl_message(reply_to=reply_to))
    assert row is not None
    assert row["top_msg_id"] == 10
    assert row["reply_to_msg_id"] == 55


def test_top_msg_id_falls_back_to_topic_root() -> None:
    reply_to = SimpleNamespace(reply_to_msg_id=10, reply_to_top_id=None, forum_topic=True)
    row = ing.message_to_row(1, _tl_message(reply_to=reply_to))
    assert row is not None and row["top_msg_id"] == 10


def test_plain_reply_is_not_a_topic() -> None:
    reply_to = SimpleNamespace(reply_to_msg_id=10, reply_to_top_id=None, forum_topic=False)
    row = ing.message_to_row(1, _tl_message(reply_to=reply_to))
    assert row is not None and row["top_msg_id"] is None


# ─── til aniqlash ────────────────────────────────────────────────────────────


def test_short_text_skips_detection() -> None:
    assert ing.detect_lang("salom") == (None, None)
    assert ing.detect_lang("") == (None, None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Toʻlov qachon boʻladi, kelishib olaylik", "uz"),
        ("Бугун ишга бормадим, касал бўлиб қолдим", "uz"),
        ("Оплата будет завтра, я уже отправил документы", "ru"),
        ("The payment will be sent tomorrow morning", "en"),
    ],
)
def test_detect_lang_main_cases(text: str, expected: str) -> None:
    lang, conf = ing.detect_lang(text)
    assert lang == expected
    assert conf is not None and 0.0 <= conf <= 1.0


def test_detect_lang_is_written_into_the_row() -> None:
    row = ing.message_to_row(1, _tl_message(message="Toʻlov qachon boʻladi, kelishamizmi?"))
    assert row is not None
    assert row["lang"] == "uz" and row["lang_conf"] is not None


def test_detect_lang_never_raises() -> None:
    samples = ("👍👍👍👍👍👍👍👍", "1234567890123", "!!!!!!!!!!!!!!", "\n" * 12)
    for sample in samples:
        lang, conf = ing.detect_lang(sample)
        assert lang in (None, "uz", "ru", "en", "other")
        assert conf is None or 0.0 <= conf <= 1.0


# ─── link_prefix ─────────────────────────────────────────────────────────────


def test_link_prefix_public_username_wins() -> None:
    assert ing.link_prefix_for(ChatType.CHANNEL, -1001234567890, "uznews") == (
        "https://t.me/uznews/"
    )


def test_link_prefix_private_channel_strips_minus_100() -> None:
    assert ing.link_prefix_for(ChatType.CHANNEL, -1001234567890, None) == (
        "https://t.me/c/1234567890/"
    )
    # Telethon "xom" id bersa ham bir xil natija
    assert ing.link_prefix_for(ChatType.SUPERGROUP, 1234567890, None) == (
        "https://t.me/c/1234567890/"
    )


def test_link_prefix_none_for_private_and_basic_group() -> None:
    assert ing.link_prefix_for(ChatType.PRIVATE, 777, None) is None
    assert ing.link_prefix_for(ChatType.GROUP, 777, None) is None


def test_link_prefix_none_for_private_even_with_a_username() -> None:
    """Jonli regressiya: shaxsiy chat egasining `@username`si bo'lsa ham

    `t.me/<username>/<id>` havolasi ochilmaydi — faqat ochiq kanal/supergruh
    postiga ishlaydi. `link_prefix_for` bunday qatorga umuman yozmaydi.
    """
    assert ing.link_prefix_for(ChatType.PRIVATE, 777, "JOO571") is None


def test_link_prefix_fits_column() -> None:
    prefix = ing.link_prefix_for(ChatType.CHANNEL, -1001234567890, "a" * 64)
    assert prefix is not None and len(prefix) <= 128
