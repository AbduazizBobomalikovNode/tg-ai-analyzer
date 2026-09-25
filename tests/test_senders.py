"""`services/senders.py` — akkaunt egasining "Men" yorlig'i.

Nima qo'riqlanadi:
  * chiquvchi (`out=True`) xabar — istalgan renderer'da "Men" (`sender_label`,
    `tools._line`, `window_search.stats_window` per-sender);
  * `senders` jadvalida qatori bor jo'natuvchi — o'sha nom bilan;
  * notanish (`senders`da yo'q) boshqa jo'natuvchi — hujjatlashtirilgan
    fallback `user:<id>` bilan qoladi (regressiya emas);
  * `out` ishonchsiz bo'lgan holat (eski qator) — `sender_id ==
    account.tg_account_id` zaxira yo'li bilan hamon "Men".
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.db.models import Chat, ChatType, Message
from app.services import senders as SN
from app.services import tools as T
from app.services import window_search as WS

# `TEST_DATABASE_URL` berilmasa faqat DB'ga tegadigan test skip bo'ladi
# (`pg_engine` fixture'ining o'zi); sof funksiyalar har doim yuguradi.


# ─── sof funksiyalar ─────────────────────────────────────────────────────────


def test_owner_label_by_locale_falls_back_to_uz() -> None:
    assert SN.owner_label("uz") == "Men"
    assert SN.owner_label("ru") == "Я"
    assert SN.owner_label("en") == "Me"
    assert SN.owner_label(None) == "Men"
    assert SN.owner_label("fr") == "Men"  # noma'lum til — uz fallback


def test_owner_label_includes_the_account_display_name() -> None:
    assert SN.owner_label("uz", display_name="Aziz") == "Men (Aziz)"
    assert SN.owner_label("ru", display_name="  ") == "Я"  # bo'sh label — bazaviy nom


def test_is_owner_message_by_out_flag_or_id_fallback() -> None:
    assert SN.is_owner_message(sender_id=None, out=True, owner_tg_id=None)
    assert SN.is_owner_message(sender_id=999, out=False, owner_tg_id=999)
    assert not SN.is_owner_message(sender_id=999, out=False, owner_tg_id=1000)
    assert not SN.is_owner_message(sender_id=None, out=False, owner_tg_id=999)


def test_sender_label_owner_vs_known_vs_unknown() -> None:
    # Chiquvchi xabar — "Men".
    assert SN.sender_label(555, out=True, owner_tg_id=555) == "Men"
    # `out` yo'q, lekin id egasiniki — zaxira yo'l ham "Men" beradi.
    assert SN.sender_label(555, out=False, owner_tg_id=555) == "Men"
    # `senders` jadvalida topilgan — ism.
    assert SN.sender_label(7, names={7: "Nurbek (@nurbek)"}) == "Nurbek (@nurbek)"
    # Notanish jo'natuvchi — hujjatlashtirilgan fallback.
    assert SN.sender_label(42, names={7: "Nurbek"}) == "user:42"
    # Kanal posti / sender_id yo'q — bu funksiya chaqiruvchisi hal qiladi,
    # bu yerda faqat "—" fallback.
    assert SN.sender_label(None) == "—"


@pytest.mark.asyncio
async def test_owner_info_returns_none_when_account_missing() -> None:
    class _EmptySession:
        async def execute(self, *a: object, **kw: object) -> object:
            return SimpleNamespace(all=lambda: [])

    assert await SN.owner_info(_EmptySession(), 1) == (None, "")  # type: ignore[arg-type]


def test_relabel_owner_hits_only_matches_the_exact_marker() -> None:
    mine = SimpleNamespace(sender="user:555")
    other = SimpleNamespace(sender="user:9")
    channel_post = SimpleNamespace(sender="Kanal nomi")
    hits = [mine, other, channel_post]
    SN.relabel_owner_hits(hits, owner_tg_id=555, locale="uz")
    assert mine.sender == "Men"
    assert other.sender == "user:9"  # boshqasi tegilmaydi
    assert channel_post.sender == "Kanal nomi"


# ─── `tools._line` — evidence/tool natija qatori ─────────────────────────────


def _chat_row() -> Chat:
    return Chat(id=1, account_id=1, tg_peer_id=99, type=ChatType.SUPERGROUP, title="Guruh")


def _msg(mid: int, sender: int | None, *, out: bool = False) -> Message:
    return Message(
        id=mid,
        chat_id=1,
        tg_msg_id=1000 + mid,
        sender_id=sender,
        out=out,
        published_at=datetime(2026, 9, 1, tzinfo=UTC),
        text="salom",
    )


def test_line_labels_an_outgoing_message_as_men() -> None:
    chat = _chat_row()
    line = T._line(chat, _msg(1, 555, out=True), owner_tg_id=555)
    assert "Men" in line and "user:555" not in line


def test_line_labels_the_owner_even_without_the_out_flag() -> None:
    """`out` eski qatorda ishonchsiz bo'lishi mumkin — `sender_id` zaxira."""
    chat = _chat_row()
    line = T._line(chat, _msg(1, 555, out=False), owner_tg_id=555)
    assert "Men" in line


def test_line_uses_the_senders_table_name_for_a_known_sender() -> None:
    chat = _chat_row()
    line = T._line(chat, _msg(1, 7), names={7: "Nurbek (@nurbek)"}, owner_tg_id=555)
    assert "Nurbek (@nurbek)" in line and "user:7" not in line


def test_line_falls_back_to_the_raw_id_for_an_unknown_sender() -> None:
    """Boshqa (notanish) jo'natuvchi — hujjatlashtirilgan `user:<id>` fallback."""
    chat = _chat_row()
    line = T._line(chat, _msg(1, 42), names={7: "Nurbek"}, owner_tg_id=555)
    assert "user:42" in line


def test_line_localizes_the_owner_label() -> None:
    chat = _chat_row()
    ru = T._line(chat, _msg(1, 555, out=True), owner_tg_id=555, locale="ru")
    en = T._line(chat, _msg(1, 555, out=True), owner_tg_id=555, locale="en")
    assert "Я" in ru
    assert "Me" in en


# ─── `window_search.stats_window` — per-sender jadval ────────────────────────


async def test_stats_window_per_sender_labels_the_owner_defensively(pg_engine: AsyncEngine) -> None:
    """`out=False` bo'lsa ham `sender_id == account.tg_account_id` — hamon "Men".

    So'rov normal holatda `out=True` qatorlarni chiqarib tashlaydi (egasi
    reytingga kirmaydi), lekin CLAUDE.md ogohlantirganidek eski qatorlarda
    `out` ishonchsiz bo'lishi mumkin — shu holat uchun himoya qatlami.
    """
    async with pg_engine.begin() as conn:
        owner_tg = 900_100_001
        uid = int(
            (
                await conn.execute(
                    text(
                        "INSERT INTO users (tg_user_id, locale) VALUES "
                        "(floor(random() * 1e12)::bigint, 'uz') RETURNING id"
                    )
                )
            ).scalar_one()
        )
        aid = int(
            (
                await conn.execute(
                    text(
                        "INSERT INTO accounts (user_id, tg_account_id, label, status) "
                        "VALUES (:uid, :tg, 'Aziz', 'active') RETURNING id"
                    ),
                    {"uid": uid, "tg": owner_tg},
                )
            ).scalar_one()
        )
        chat_id = int(
            (
                await conn.execute(
                    text(
                        "INSERT INTO chats (account_id, tg_peer_id, type, title, write_mode, "
                        "sync_state, synced_total) VALUES (:aid, floor(random()*1e12)::bigint, "
                        "'supergroup', 'Guruh', 'read_only', 'idle', 0) RETURNING id"
                    ),
                    {"aid": aid},
                )
            ).scalar_one()
        )
        # Bitta "eski" qator: `out` yolg'on, lekin `sender_id` egasiniki.
        await conn.execute(
            text(
                "INSERT INTO messages (chat_id, tg_msg_id, published_at, text, out, sender_id) "
                "VALUES (:c, 1, :d, 'eski xabar', false, :s)"
            ),
            {"c": chat_id, "d": datetime(2026, 9, 1, tzinfo=UTC), "s": owner_tg},
        )
        # Va bitta chinakam boshqa odam.
        await conn.execute(
            text(
                "INSERT INTO messages (chat_id, tg_msg_id, published_at, text, out, sender_id) "
                "VALUES (:c, 2, :d, 'boshqa xabar', false, 42)"
            ),
            {"c": chat_id, "d": datetime(2026, 9, 2, tzinfo=UTC)},
        )

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as session:
        spec = WS.WindowSpec(
            account_id=aid,
            chats=None,
            since=datetime(2026, 1, 1, tzinfo=UTC),
            until=datetime(2026, 12, 31, tzinfo=UTC),
            direction="",
        )
        agg = await WS.stats_window(session, spec, locale="uz")
    labels = dict(agg.per_sender)
    assert labels.get("Men (Aziz)") == 1  # `accounts.label` bo'lsa qo'shiladi
    assert labels.get("user:42") == 1
