"""F3/H1, H3, M7 — haqiqiy Postgres ustidagi regressiyalar.

`tests/test_db_integration.py` bilan bir xil fixture'lar (`pg_engine`,
`pg_url`) va yordamchilar; alohida fayl faqat parallel agentlar bitta
faylni tahrirlamasin deb.

Nima qo'riqlanadi:

  * **`message_embeddings` / `retrieval_node_embeddings` PK = `(id, model)`**
    (migratsiya 0016). Bir ustunli kalitda `docs/llm.md` dagi "yangi model
    qatorlari eskilari bilan yonma-yon" va'dasi bajarilmas edi: yozuvchilar
    "joriy model bo'yicha yetishmayapti" deb tanlab, o'sha `message_id`
    ustida `UniqueViolation` bilan qulardi, qidiruv esa yangi modelni
    filtrlab bo'sh qaytarardi.
  * **HNSW indekslari** PK almashuvidan omon qolgani — ular vektor ustunida,
    kalitga bog'liq emas, lekin buni taxmin qilmay DB'dan so'raymiz.
  * **Downgrade dedup qiladi** (har id uchun eng yangi qator) — aks holda
    bir ustunli PK'ni tiklab bo'lmaydi.
  * **Byudjet qopqog'i batch submit yo'lida ham** (H3).
  * **Uzilgan hisob qatori** `list_messages` da ko'rinmaydi, lekin
    `limits.daily_usage` uni **baribir** sanaydi (M7).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

# pytest `tests/` ni sys.path ga qo'shadi — yordamchilar bitta nusxada qoladi.
from test_db_integration import (
    _EMB_MODEL,
    _clear_inflight_batches,
    _FakeBatchProvider,
    _fixture_chat,
    _insert_message,
    _synced_chat,
)

from app.config import get_settings

pytestmark = pytest.mark.usefixtures("pg_engine")

ROOT = Path(__file__).resolve().parents[1]

_PK_COLS_SQL = (
    "SELECT a.attname FROM pg_index i "
    "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
    "WHERE i.indrelid = CAST(:tbl AS regclass) AND i.indisprimary ORDER BY a.attname"
)


def _dims() -> int:
    return get_settings().embed_dim


def _vec(seed: float) -> list[float]:
    """ORM yo'li uchun vektor (pgvector `list[float]` kutadi)."""
    return [seed] + [0.0] * (_dims() - 1)


def _vec_txt(seed: float) -> str:
    """Xom SQL (`text()`) yo'li uchun `'[...]'` literal."""
    return str(_vec(seed))


async def _pk_columns(engine: AsyncEngine, table: str) -> list[str]:
    async with engine.begin() as conn:
        rows = (await conn.execute(text(_PK_COLS_SQL), {"tbl": table})).all()
    return [r[0] for r in rows]


# ─── H1: ikki model yonma-yon ────────────────────────────────────────────────


async def test_embedding_primary_keys_include_the_model(pg_engine: AsyncEngine) -> None:
    assert await _pk_columns(pg_engine, "message_embeddings") == ["message_id", "model"]
    assert await _pk_columns(pg_engine, "retrieval_node_embeddings") == ["model", "node_id"]


async def test_hnsw_indexes_survived_the_pk_change(pg_engine: AsyncEngine) -> None:
    """Vektor indeksi kalitga bog'liq emas — lekin buni taxmin qilmaymiz."""
    async with pg_engine.begin() as conn:
        found = {
            r[0]
            for r in (
                await conn.execute(
                    text(
                        "SELECT indexname FROM pg_indexes WHERE indexname IN "
                        "('ix_embeddings_hnsw', 'ix_node_emb_hnsw')"
                    )
                )
            ).all()
        }
    assert found == {"ix_embeddings_hnsw", "ix_node_emb_hnsw"}


async def test_two_models_live_side_by_side_and_the_filter_sees_one(
    pg_engine: AsyncEngine,
) -> None:
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from app.db.models import MessageEmbedding

    _account_id, chat_id = await _fixture_chat(pg_engine)
    msg_id = await _insert_message(pg_engine, chat_id, 9_001, "ikki model uchun yetarlicha matn")

    old, new = "gemini-embedding-001", "gemini-embedding-2"
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as session:
        for model, seed in ((old, 0.1), (new, 0.9)):
            stmt = pg_insert(MessageEmbedding).values(
                [{"message_id": msg_id, "model": model, "vector": _vec(seed)}]
            )
            # Aynan yozuvchilar ishlatadigan konflikt kaliti.
            await session.execute(
                stmt.on_conflict_do_nothing(index_elements=["message_id", "model"])
            )
        await session.commit()

    async with pg_engine.begin() as conn:
        models = sorted(
            r[0]
            for r in (
                await conn.execute(
                    text("SELECT model FROM message_embeddings WHERE message_id = :m"),
                    {"m": msg_id},
                )
            ).all()
        )
        # `search._embed_model` uslubidagi filtr — **faqat** joriy model.
        filtered = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM message_embeddings WHERE message_id = :m AND model = :mod"
                ),
                {"m": msg_id, "mod": new},
            )
        ).scalar_one()
    assert models == sorted([old, new])
    assert filtered == 1


async def test_reembed_writer_adds_a_model_without_touching_the_old_row(
    pg_engine: AsyncEngine,
) -> None:
    """`reembed_all` / `embed_messages` yozuvchisi — qo'shadi, almashtirmaydi."""
    from app.worker import tasks as T

    _account_id, chat_id = await _fixture_chat(pg_engine)
    msg_id = await _insert_message(pg_engine, chat_id, 9_002, "qayta embed qilinadigan matn")

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    for model, seed in (("gemini-embedding-001", 0.25), ("gemini-embedding-2", 0.75)):
        async with maker() as session:
            await T._upsert_message_vectors(
                session, [(msg_id, [seed] + [0.0] * (_dims() - 1))], model
            )
            await session.commit()

    # Ikkinchi marta: konflikt — eski qiymat **saqlanadi**, xato yo'q.
    async with maker() as session:
        await T._upsert_message_vectors(
            session, [(msg_id, [0.42] + [0.0] * (_dims() - 1))], "gemini-embedding-001"
        )
        await session.commit()

    async with pg_engine.begin() as conn:
        rows = dict(
            (
                await conn.execute(
                    text(
                        "SELECT model, (vector::real[])[1] FROM message_embeddings "
                        "WHERE message_id = :m"
                    ),
                    {"m": msg_id},
                )
            ).all()
        )
    assert set(rows) == {"gemini-embedding-001", "gemini-embedding-2"}
    assert float(rows["gemini-embedding-001"]) == pytest.approx(0.25)
    assert float(rows["gemini-embedding-2"]) == pytest.approx(0.75)


async def test_migration_0016_round_trips_and_downgrade_dedupes(
    pg_engine: AsyncEngine, pg_url: str
) -> None:
    """0016 → 0015 → head: downgrade har id uchun **eng yangi** qatorni qoldiradi."""
    env = {**os.environ, "DATABASE_URL": pg_url, "PYTHONPATH": str(ROOT / "src")}

    def alembic(*argv: str) -> None:
        subprocess.run(  # argv qat'iy, shell yo'q  # noqa: S603
            [sys.executable, "-m", "alembic", *argv],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
        )

    _account_id, chat_id = await _fixture_chat(pg_engine)
    msg_id = await _insert_message(pg_engine, chat_id, 9_003, "migratsiya round-trip matni")
    async with pg_engine.begin() as conn:
        for model, seed, days in (("old-model", 0.1, 2), ("new-model", 0.9, 0)):
            await conn.execute(
                text(
                    "INSERT INTO message_embeddings (message_id, model, vector, created_at) "
                    "VALUES (:m, :mod, :v, now() - make_interval(days => :d))"
                ),
                {"m": msg_id, "mod": model, "v": _vec_txt(seed), "d": days},
            )

    try:
        alembic("downgrade", "0015")
        assert await _pk_columns(pg_engine, "message_embeddings") == ["message_id"]
        async with pg_engine.begin() as conn:
            left = [
                r[0]
                for r in (
                    await conn.execute(
                        text("SELECT model FROM message_embeddings WHERE message_id = :m"),
                        {"m": msg_id},
                    )
                ).all()
            ]
        assert left == ["new-model"]  # eng yangisi qoldi, eskisi tushdi
    finally:
        alembic("upgrade", "head")

    assert await _pk_columns(pg_engine, "message_embeddings") == ["message_id", "model"]
    async with pg_engine.begin() as conn:
        empty = (
            await conn.execute(text("SELECT count(*) FROM message_embeddings WHERE model = ''"))
        ).scalar_one()
        nullable = (
            await conn.execute(
                text(
                    "SELECT is_nullable FROM information_schema.columns "
                    "WHERE table_name = 'message_embeddings' AND column_name = 'model'"
                )
            )
        ).scalar_one()
    assert empty == 0 and nullable == "NO"


# ─── H3: byudjet qopqog'i batch submit yo'lida ───────────────────────────────


async def test_submit_messages_skips_budget_blocked_accounts(pg_engine: AsyncEngine) -> None:
    """Ilgari `embed_messages` `blocked` ni hisoblab, batch yo'liga uzatmasdi.

    `EMBED_MODE=batch` da qopqoq umuman ishlamasdi — batch esa **oldindan**
    to'lanadi, ya'ni tejashning yagona joyi aynan shu tanlov.
    """
    from app.services import embed_batches as eb

    await _clear_inflight_batches(pg_engine)
    account_id, chat_id = await _synced_chat(pg_engine)
    await _insert_message(pg_engine, chat_id, 9_101, "byudjet qopqog'i uchun yetarli uzun matn")

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as session:
        rep = await eb.submit_messages(
            session, account_id=account_id, skip_accounts={account_id}, limit=10
        )
        await session.commit()
    assert rep.submitted == 0 and rep.job == "" and not rep.error


async def test_submit_messages_looks_for_the_current_model(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch tanlovi `embed_messages` bilan **bir xil** predikatga tayanadi.

    Eski modelda vektori bor xabar joriy model uchun hali yetishmayapti;
    ilgari batch yo'li "umuman vektori yo'q" deb qarardi va uni ko'rmasdi —
    `EMBED_MODE` ni almashtirish indekslashni jimgina to'xtatardi.
    """
    from app.services import embed_batches as eb

    await _clear_inflight_batches(pg_engine)
    account_id, chat_id = await _synced_chat(pg_engine)
    msg_id = await _insert_message(
        pg_engine, chat_id, 9_102, "joriy model uchun yetishmayotgan matn"
    )
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO message_embeddings (message_id, model, vector) VALUES (:m, :mod, :v)"
            ),
            {"m": msg_id, "mod": "some-old-model", "v": _vec_txt(0.1)},
        )

    provider = _FakeBatchProvider(states=[], vectors=[])
    monkeypatch.setattr(eb, "_provider_for_batch", lambda: (provider, _EMB_MODEL))
    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as session:
        rep = await eb.submit_messages(session, account_id=account_id, limit=10)
        await session.commit()

    assert rep.submitted == 1
    assert provider.submitted and len(provider.submitted[0]) == 1


# ─── M7: uzilgan hisob qatori — serverda filtrlanadi, byudjetda qoladi ──────


async def test_list_messages_hides_aborted_rows_but_budget_still_counts_them(
    pg_engine: AsyncEngine,
) -> None:
    from app.services import chat_service as cs
    from app.services import limits as LIM

    account_id, _chat_id = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        user_id = (
            await conn.execute(
                text("SELECT user_id FROM accounts WHERE id = :a"), {"a": account_id}
            )
        ).scalar_one()
        conv_id = (
            await conn.execute(
                text(
                    "INSERT INTO conversations (user_id, account_id, title) "
                    "VALUES (:u, :a, 'm7') RETURNING id"
                ),
                {"u": user_id, "a": account_id},
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO conversation_messages "
                "(conversation_id, role, content, tokens_in, tokens_out, context) VALUES "
                "(:c, 'user', 'savol', 0, 0, NULL), "
                "(:c, 'assistant', 'javob', 10, 5, CAST(:ok AS jsonb)), "
                "(:c, 'assistant', '', 100, 7, CAST(:aborted AS jsonb))"
            ),
            {
                "c": conv_id,
                "ok": '{"mode": "deep"}',
                "aborted": '{"mode": "deep", "aborted": true}',
            },
        )

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as session:
        rows = await cs.list_messages(session, int(conv_id))
        usage = await LIM.daily_usage(session, int(user_id))

    assert [r.content for r in rows] == ["savol", "javob"]  # uzilgan qator yo'q
    assert all(not (r.context or {}).get("aborted") for r in rows)
    # Byudjet `conversation_messages` ni to'g'ridan-to'g'ri yig'adi.
    assert usage.tokens >= 122  # 10+5 va 100+7 — ikkalasi ham hisobda
