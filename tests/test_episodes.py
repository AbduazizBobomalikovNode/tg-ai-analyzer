"""Epizod segmentatsiyasi (phase2/A, arxitektura v2, 5.2-band).

Qo'riqlanadigan xulq:

  * uch chegara qoidasi — gap (chat 45 daqiqa / kanal 6 soat), 40 xabar, 6000 belgi;
  * **reply-zanjir hech qachon bo'linmaydi** (union-find), qopqoqdan katta
    komponent bitta epizod bo'lib qoladi;
  * `ctx_prefix` formati va "replying to" faqat epizoddan tashqaridagi otaga;
  * `content_hash` tahrirni sezadi, `simhash` belgili `bigint` oralig'ida;
  * 500 xabarlik sintetik oqimda median epizod uzunligi [5, 25] oralig'ida;
  * `build_episodes` idempotent: o'zgarmagan epizod qayta yozilmaydi,
    o'zgargani yangilanadi va vektori o'chiriladi.

DB yo'q — `_FakeSession` navbatdagi natijalarni qaytaradi va SQL'ni yozib boradi.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any

from app.db.models import Chat, Message, RetrievalNode
from app.services import episodes as EP

NOW = datetime(2026, 3, 14, 9, 0, tzinfo=UTC)


def _chat(kind: str = "supergroup") -> Chat:
    return Chat(id=1, account_id=7, tg_peer_id=99, type=kind, title="Ish guruhi")


def _msg(
    i: int,
    *,
    at: datetime,
    text: str = "salom",
    sender: int | None = 1,
    reply_to: int | None = None,
) -> Message:
    return Message(
        id=1000 + i,
        chat_id=1,
        tg_msg_id=i,
        published_at=at,
        text=text,
        sender_id=sender,
        reply_to_msg_id=reply_to,
        media_type=None,
        views=None,
    )


def _stream(n: int, *, step_s: int = 60, start: datetime = NOW) -> list[Message]:
    return [_msg(i, at=start + timedelta(seconds=step_s * i)) for i in range(1, n + 1)]


# ─── segmentatsiya qoidalari ─────────────────────────────────────────────────


def test_segment_cuts_on_time_gap() -> None:
    msgs = _stream(3)
    msgs.append(_msg(4, at=msgs[-1].published_at + timedelta(minutes=46)))
    msgs.append(_msg(5, at=msgs[-1].published_at + timedelta(minutes=1)))
    parts = EP.segment(msgs, gap_seconds=EP.GAP_SECONDS)
    assert [[m.tg_msg_id for m in p] for p in parts] == [[1, 2, 3], [4, 5]]


def test_segment_keeps_short_gaps_together() -> None:
    parts = EP.segment(_stream(10, step_s=120), gap_seconds=EP.GAP_SECONDS)
    assert len(parts) == 1


def test_segment_cuts_on_message_cap() -> None:
    parts = EP.segment(_stream(95, step_s=30), gap_seconds=EP.GAP_SECONDS)
    assert all(len(p) <= EP.MAX_EPISODE_MSGS for p in parts)
    assert len(parts) == 3  # 40 + 40 + 15


def test_segment_cuts_on_char_cap() -> None:
    msgs = [_msg(i, at=NOW + timedelta(seconds=30 * i), text="x" * 1000) for i in range(1, 16)]
    parts = EP.segment(msgs, gap_seconds=EP.GAP_SECONDS)
    assert len(parts) > 1
    assert all(sum(len(m.text) for m in p[:-1]) <= EP.MAX_EPISODE_CHARS for p in parts)


def test_channel_uses_six_hour_gap() -> None:
    chat = _chat("channel")
    msgs = [
        _msg(1, at=NOW, sender=None),
        _msg(2, at=NOW + timedelta(hours=2), sender=None),
        _msg(3, at=NOW + timedelta(hours=9), sender=None),
    ]
    parts = EP.build_from_messages(msgs, chat)
    assert [p.msg_lo for p in parts] == [1, 3]
    # bir xil oqim oddiy guruhda uchga bo'linadi (45 daqiqa)
    assert len(EP.build_from_messages(msgs, _chat())) == 3


# ─── reply-zanjir ────────────────────────────────────────────────────────────


def test_reply_chain_is_never_split_by_gap() -> None:
    """#1 ga javob #4 — orada 3 soatlik bo'shliq bo'lsa ham bitta epizod."""
    msgs = [
        _msg(1, at=NOW, text="to'lov qachon?"),
        _msg(2, at=NOW + timedelta(minutes=1)),
        _msg(3, at=NOW + timedelta(minutes=2)),
        _msg(4, at=NOW + timedelta(hours=3), reply_to=1, text="20-sanaga"),
        _msg(5, at=NOW + timedelta(hours=5)),
    ]
    parts = EP.segment(msgs, gap_seconds=EP.GAP_SECONDS)
    ids = [[m.tg_msg_id for m in p] for p in parts]
    assert ids == [[1, 2, 3, 4], [5]]


def test_component_larger_than_caps_stays_one_episode() -> None:
    """Zanjir 40 xabardan uzun bo'lsa ham bo'linmaydi (ataylab)."""
    msgs = [_msg(1, at=NOW)]
    for i in range(2, 61):
        msgs.append(_msg(i, at=NOW + timedelta(seconds=30 * i), reply_to=i - 1))
    parts = EP.segment(msgs, gap_seconds=EP.GAP_SECONDS)
    assert len(parts) == 1 and len(parts[0]) == 60


def test_allowed_cuts_blocks_inside_component_only() -> None:
    msgs = [
        _msg(1, at=NOW),
        _msg(2, at=NOW + timedelta(minutes=1)),
        _msg(3, at=NOW + timedelta(minutes=2), reply_to=1),
        _msg(4, at=NOW + timedelta(minutes=3)),
    ]
    # #3 → #1 komponenti 0..2 indekslarni egallaydi: ichidagi har ikki chegara
    # (1 va 2) taqiqlanadi, faqat komponentdan keyingi 3 ochiq qoladi.
    assert EP.allowed_cuts(msgs) == {3}
    roots = EP.reply_components(msgs)
    assert roots[0] == roots[2] and roots[1] != roots[0]


def test_reply_outside_window_does_not_extend_component() -> None:
    msgs = [_msg(9, at=NOW, reply_to=1), _msg(10, at=NOW + timedelta(hours=4))]
    assert EP.allowed_cuts(msgs) == {1}


# ─── ctx_prefix / hash ───────────────────────────────────────────────────────


def test_ctx_prefix_has_chat_day_and_speakers() -> None:
    msgs = [
        _msg(1, at=NOW, sender=11),
        _msg(2, at=NOW + timedelta(minutes=1), sender=22),
        _msg(3, at=NOW + timedelta(minutes=2), sender=11),
    ]
    prefix = EP.build_ctx_prefix(msgs, _chat())
    assert prefix.startswith("[chat: Ish guruhi (supergroup) | 2026-03-14 | speakers: ")
    assert "user:11, user:22" in prefix and prefix.count("user:11") == 1
    assert "replying to" not in prefix


def test_ctx_prefix_mentions_outside_parent_only() -> None:
    parent = _msg(1, at=NOW, sender=9, text="to'lovni 20-sanaga suramizmi " + "z" * 200)
    msgs = [_msg(5, at=NOW + timedelta(hours=2), sender=11, reply_to=1)]
    with_parent = EP.build_ctx_prefix(msgs, _chat(), reply_parent=parent)
    assert 'replying to user:9: "' in with_parent
    quoted = with_parent.split('replying to user:9: "')[1].rstrip('"]')
    assert len(quoted) == EP.REPLY_SNIPPET_CHARS

    # ota epizod ichida bo'lsa — prefiksga qo'shilmaydi
    inside = EP.build_from_messages([parent, _msg(2, at=NOW, reply_to=1)], _chat())
    assert "replying to" not in inside[0].ctx_prefix


def test_content_hash_reacts_to_edit_and_deletion() -> None:
    msgs = _stream(3)
    base = EP.content_hash(msgs)
    edited = [_msg(1, at=msgs[0].published_at, text="boshqa matn"), msgs[1], msgs[2]]
    assert EP.content_hash(edited) != base
    assert EP.content_hash(msgs[:2]) != base
    assert EP.content_hash(msgs) == base  # deterministik


def test_episode_text_uses_digest_line_format() -> None:
    ep = EP.make_episode([_msg(7, at=NOW, sender=11, text="salom")], _chat())
    assert ep.text == "[09:00] #7 user:11: salom"
    assert ep.msg_lo == ep.msg_hi == 7 and ep.speakers == [11]


def test_simhash_fits_signed_bigint() -> None:
    ep = EP.make_episode(_stream(5), _chat())
    assert -(2**63) <= ep.simhash < 2**63
    assert EP.to_signed64(2**64 - 1) == -1
    assert EP.to_signed64(5) == 5


# ─── median o'lchami (asosiy sifat mezoni) ───────────────────────────────────


def _synthetic(n: int = 500) -> list[Message]:
    """Realistik oqim: qisqa navbatlar, orada tanaffuslar, 30% javoblar."""
    rng = random.Random(7)  # noqa: S311 — test uchun takrorlanadigan oqim
    out: list[Message] = []
    t = NOW
    for i in range(1, n + 1):
        roll = rng.random()
        if roll < 0.70:
            t += timedelta(seconds=rng.randint(20, 180))
        elif roll < 0.90:
            t += timedelta(minutes=rng.randint(5, 20))
        else:
            t += timedelta(minutes=rng.randint(60, 240))
        reply_to = None
        if out and rng.random() < 0.3:
            reply_to = out[max(0, len(out) - rng.randint(1, 3))].tg_msg_id
        out.append(
            _msg(
                i,
                at=t,
                text="gap " * rng.randint(2, 30),
                sender=rng.randint(1, 4),
                reply_to=reply_to,
            )
        )
    return out


def test_median_episode_size_is_in_target_band() -> None:
    msgs = _synthetic()
    parts = EP.segment(msgs, gap_seconds=EP.GAP_SECONDS)
    sizes = [len(p) for p in parts]
    assert 5 <= median(sizes) <= 25, sizes
    assert sum(sizes) == len(msgs)  # bitta xabar ham yo'qolmadi


def test_reply_chains_stay_inside_one_episode() -> None:
    msgs = _synthetic()
    parts = EP.segment(msgs, gap_seconds=EP.GAP_SECONDS)
    where = {m.tg_msg_id: idx for idx, p in enumerate(parts) for m in p}
    for m in msgs:
        if m.reply_to_msg_id in where:
            assert where[m.tg_msg_id] == where[m.reply_to_msg_id]


# ─── build_episodes (fake session) ───────────────────────────────────────────


class _Result:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return list(self._rows)

    def first(self) -> Any:
        return self._rows[0] if self._rows else None

    def scalars(self) -> list[Any]:
        return list(self._rows)


class _FakeSession:
    """`build_episodes` uchun minimal sessiya: chat, xabarlar, mavjud tugunlar."""

    def __init__(self, chat: Chat, msgs: list[Message], nodes: list[RetrievalNode]) -> None:
        self.chat = chat
        self.msgs = msgs
        self.nodes = nodes
        self.added: list[Any] = []
        self.sql: list[str] = []

    async def get(self, _model: Any, _pk: int) -> Chat:
        return self.chat

    async def execute(self, stmt: Any) -> _Result:
        sql = str(stmt)
        self.sql.append(sql)
        if sql.lstrip().startswith("DELETE"):
            return _Result([])
        if "FROM messages" in sql:
            return _Result(self.msgs)
        if "retrieval_nodes.id" in sql:  # to'liq tugun qatorlari
            return _Result(self.nodes)
        return _Result([(n.msg_lo,) for n in self.nodes][-1:])  # _rebuild_from

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    async def flush(self) -> None:
        return None


async def test_build_episodes_creates_nodes_and_is_idempotent() -> None:
    chat = _chat()
    msgs = _stream(12, step_s=120)
    session = _FakeSession(chat, msgs, [])
    first = await EP.build_episodes(session, 1)  # type: ignore[arg-type]
    assert first.created == 1 and first.unchanged == 0 and first.messages == 12
    node = session.added[0]
    assert node.level == EP.LEVEL_EPISODE and node.account_id == 7
    assert node.msg_lo == 1 and node.msg_hi == 12 and node.msg_count == 12

    # Ikkinchi yugurish: aynan shu tugun bor → hech narsa yozilmaydi.
    again = _FakeSession(chat, msgs, [node])
    second = await EP.build_episodes(again, 1)  # type: ignore[arg-type]
    assert (second.created, second.updated, second.unchanged) == (0, 0, 1)
    assert not again.added
    assert not any(s.lstrip().startswith("DELETE") for s in again.sql)


async def test_build_episodes_rewrites_changed_window_and_drops_vector() -> None:
    chat = _chat()
    msgs = _stream(12, step_s=120)
    stale = EP.make_episode(msgs, chat)
    node = RetrievalNode(
        id=55,
        account_id=7,
        chat_id=1,
        level=EP.LEVEL_EPISODE,
        starts_at=stale.starts_at,
        ends_at=stale.ends_at,
        msg_lo=stale.msg_lo,
        msg_hi=stale.msg_hi,
        msg_count=stale.msg_count,
        text=stale.text,
        ctx_prefix=stale.ctx_prefix,
        content_hash="eskirgan-hash",
        simhash=0,
    )
    session = _FakeSession(chat, msgs, [node])
    rep = await EP.build_episodes(session, 1)  # type: ignore[arg-type]
    assert (rep.created, rep.updated, rep.unchanged) == (0, 1, 0)
    assert node.content_hash == stale.content_hash and node.text == stale.text
    deletes = [s for s in session.sql if s.lstrip().startswith("DELETE")]
    assert any("retrieval_node_embeddings" in s for s in deletes)


async def test_build_episodes_excludes_deleted_messages() -> None:
    session = _FakeSession(_chat(), _stream(3), [])
    await EP.build_episodes(session, 1)  # type: ignore[arg-type]
    msg_sql = next(s for s in session.sql if "FROM messages" in s)
    assert "deleted_at IS NULL" in msg_sql
    assert "tg_msg_id >=" in msg_sql  # faqat oxirgi barqaror chegaradan


async def test_build_episodes_full_reads_from_zero() -> None:
    node = RetrievalNode(id=1, account_id=7, chat_id=1, level=1, msg_lo=500)
    session = _FakeSession(_chat(), _stream(3), [node])
    await EP.build_episodes(session, 1, full=True)  # type: ignore[arg-type]
    # `full=True` da oxirgi chegara so'ralmaydi — birinchi so'rov darhol xabarlar
    assert "FROM messages" in session.sql[0]
