"""`/api/stats/overview` — dashboard (ui/R4) uchun shakl va oldingi davr matematikasi.

Sahifa to'rtta bosh ko'rsatkichni **delta bilan** ko'rsatadi, shuning uchun
`prev` bloki aynan oldingi teng uzunlikdagi oynani sanashi shart: agar u
joriy davrni ham qamrab olsa, delta har doim ~0 bo'lib qoladi va karta
ma'nosini yo'qotadi.

DB'siz qismi (`_mean`, `REVIEW_QUESTION_CHARS`) har doim yuguradi; qolgani
`TEST_DATABASE_URL` bo'lsa (CI'da `pgvector/pgvector:pg16`).
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.db.models import Account, Chat, Conversation, ConversationMessage, User
from app.services import stats

# ─── DB'siz ──────────────────────────────────────────────────────────────────


def test_mean_ignores_missing_values() -> None:
    assert stats._mean(None, None) is None
    assert stats._mean(4, None) == 4.0
    assert stats._mean(4, 5) == 4.5


def test_review_question_is_clipped_for_the_card() -> None:
    """Karta ikki-uch qatordan oshmasin — savol serverda kesiladi."""
    assert stats.REVIEW_QUESTION_CHARS == 120


# ─── Postgres ustida ─────────────────────────────────────────────────────────

_counter = itertools.count(1)


def _uniq() -> int:
    """Testlar bir bazada yuguradi — `users.tg_user_id` unikal bo'lishi shart."""
    return 900_000_000 + next(_counter) * 7 + int(datetime.now(UTC).timestamp()) % 100_000 * 100


def _session_factory(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def _seed(engine: AsyncEngine) -> tuple[int, int]:
    """User + account + chat + ikki davrga tarqalgan suhbat qatorlari.

    Qaytaradi: `(user_id, chat_id)`.
    """
    now = datetime.now(UTC)
    async with _session_factory(engine)() as db:
        user = User(tg_user_id=_uniq(), locale="uz")
        db.add(user)
        await db.flush()
        acc = Account(user_id=user.id, tg_account_id=user.tg_user_id, status="active")
        db.add(acc)
        await db.flush()
        chat = Chat(account_id=acc.id, tg_peer_id=424242, type="group", title="Ish chati")
        db.add(chat)
        await db.flush()
        conv = Conversation(user_id=user.id, title="test")
        db.add(conv)
        await db.flush()

        ctx = {"account_id": acc.id, "peer_id": chat.tg_peer_id, "title": chat.title}

        def pair(offset_days: float, *, cost: float, latency: int, rating: int | None) -> None:
            at = now - timedelta(days=offset_days)
            db.add(
                ConversationMessage(
                    conversation_id=conv.id, role="user", content="Q" * 300, created_at=at
                )
            )
            db.add(
                ConversationMessage(
                    conversation_id=conv.id,
                    role="assistant",
                    content="A",
                    model="claude-opus-5",
                    provider="claude",
                    tokens_in=100,
                    tokens_out=20,
                    cost_usd=cost,
                    latency_ms=latency,
                    rating=rating,
                    auto_relevance=2 if rating == -1 else 5,
                    auto_usefulness=2 if rating == -1 else 4,
                    auto_grounded=True,
                    context=ctx,
                    created_at=at,
                )
            )

        # joriy davr (oxirgi 7 kun): 3 ta savol
        pair(0.5, cost=0.10, latency=1000, rating=1)
        pair(1.5, cost=0.20, latency=3000, rating=-1)
        pair(2.5, cost=0.30, latency=5000, rating=None)
        # oldingi davr (8-14 kun): 1 ta savol
        pair(9.0, cost=1.00, latency=9000, rating=1)
        # davrdan tashqarida (20 kun) — hech qayerda sanalmasin
        pair(20.0, cost=5.00, latency=100, rating=1)
        await db.commit()
        return user.id, chat.id


async def test_overview_shape_and_prev_period_math(pg_engine: AsyncEngine) -> None:
    user_id, chat_id = await _seed(pg_engine)
    async with _session_factory(pg_engine)() as db:
        ov = await stats.overview(db, user_id, days=7)

    # ── shakl: sahifa o'qiydigan hamma bloklar joyida ──
    for key in ("prev", "series", "latency", "quality", "budget", "models", "top_chats", "review"):
        assert key in ov, key
    # eski kalitlar ham qoladi (tashqi iste'molchi buzilmasin)
    for key in ("totals", "daily", "strategies", "actions"):
        assert key in ov, key

    assert ov["totals"]["questions"] == 3
    assert ov["latency"]["p50_ms"] == 3000
    assert ov["latency"]["p95_ms"] is not None
    assert ov["latency"]["p95_ms"] >= ov["latency"]["p50_ms"]

    # ── oldingi teng uzunlikdagi davr: faqat 8-14 kun oralig'i ──
    assert ov["prev"]["questions"] == 1
    assert ov["prev"]["cost_usd"] == pytest.approx(1.0, abs=1e-6)
    assert ov["prev"]["p50_ms"] == 9000
    assert ov["prev"]["satisfaction"] == 100.0

    # ── sifat: `rated` — faqat qo'lda baholanganlar ──
    assert ov["quality"]["rated"] == 2
    assert ov["quality"]["up"] == 1 and ov["quality"]["down"] == 1
    assert ov["quality"]["auto_n"] == 3
    assert 1 <= float(ov["quality"]["auto_avg"]) <= 5

    # ── seriya: har kun uchun bitta qator, kunlik p50/p95 bilan ──
    assert len(ov["series"]) == 7
    assert {"day", "questions", "cost_usd", "p50_ms", "p95_ms"} <= set(ov["series"][0])
    assert sum(d["questions"] for d in ov["series"]) == 3
    assert sum(d["cost_usd"] for d in ov["series"]) == pytest.approx(0.6, abs=1e-6)

    # ── byudjet: kalitlari bor (limit qo'yilmagan bo'lsa `daily_usd` — None) ──
    assert set(ov["budget"]) == {"daily_usd", "used_usd"}

    # ── modellar ──
    assert ov["models"][0]["model"] == "claude-opus-5"
    assert ov["models"][0]["questions"] == 3
    assert ov["models"][0]["avg_ms"] == 3000

    # ── top chatlar: kontekstdagi (account_id, peer_id) → chat qatori ──
    top = ov["top_chats"][0]
    assert top["questions"] == 3
    assert top["chat_id"] == chat_id
    assert "avatar_url" in top

    # ── ko'rib chiqish: **savol** (javob emas), 120 belgigacha ──
    assert ov["review"], "past baholangan javob topilmadi"
    row = ov["review"][0]
    assert row["rating"] == -1
    assert row["question"].startswith("Q")
    assert len(row["question"]) == stats.REVIEW_QUESTION_CHARS
    assert row["message_id"] and row["conversation_id"]
    assert row["auto_score"] == 2.0


async def test_previous_period_is_empty_when_there_is_no_history(pg_engine: AsyncEngine) -> None:
    """Tarix yo'q foydalanuvchida `prev` nol bo'ladi (`None` emas) —
    frontend `deltaPct()` shunga tayanadi va "taqqoslash uchun ma'lumot
    yo'q" matnini ko'rsatadi."""
    async with _session_factory(pg_engine)() as db:
        user = User(tg_user_id=_uniq(), locale="uz")
        db.add(user)
        await db.commit()
        user_id = user.id
    async with _session_factory(pg_engine)() as db:
        ov = await stats.overview(db, user_id, days=7)
    assert ov["totals"]["questions"] == 0
    assert ov["prev"]["questions"] == 0
    assert ov["prev"]["cost_usd"] == 0.0
    assert ov["prev"]["p50_ms"] is None
    assert ov["prev"]["satisfaction"] is None
    assert ov["series"] and len(ov["series"]) == 7
    assert ov["top_chats"] == [] and ov["review"] == []
