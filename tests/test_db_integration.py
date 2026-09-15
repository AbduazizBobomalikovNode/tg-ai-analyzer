"""Haqiqiy Postgres ustidagi testlar (pgvector/pgvector:pg16).

`TEST_DATABASE_URL` berilmasa hammasi skip bo'ladi — `pg_engine` fixture'i
(`conftest.py`) migratsiyalarni bir marta yugurtiradi.

Nima qo'riqlanadi:

  * **append-only trigger** (`agent_actions`) — audit jurnalini o'chirib yoki
    qayta yozib bo'lmasligi. Bu 6-bosqich invarianti, CI'da yumshatilmaydi.
  * **`uz_fold` SQL ↔ Python pariteti** — indeks (generated column) SQL
    versiyasi bilan quriladi, so'rov esa Python versiyasi bilan. Ikkalasi
    ajralib ketsa qidiruv **jimgina** ishlamay qo'yadi.
  * **generated column'lar** haqiqatan to'ladi va `tsv_uz` prefiks so'rovi
    kirill/apostrof farqidan qat'i nazar topadi.
  * **trigram** yo'li (`strict_word_similarity` + `gin_trgm_ops`) imlo
    farqiga chidaydi.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from test_textnorm import FOLD_CORPUS  # pytest `tests/` ni sys.path ga qo'shadi

from app.services.textnorm import uz_fold, uz_norm

pytestmark = pytest.mark.usefixtures("pg_engine")


async def _fixture_chat(engine: AsyncEngine) -> tuple[int, int]:
    """Bo'sh user/account/chat yaratadi. Qaytaradi: `(account_id, chat_id)`."""
    async with engine.begin() as conn:
        uid = (
            await conn.execute(
                text(
                    "INSERT INTO users (tg_user_id, locale) VALUES "
                    "(floor(random() * 1e12)::bigint, 'uz') RETURNING id"
                )
            )
        ).scalar_one()
        aid = (
            await conn.execute(
                text(
                    "INSERT INTO accounts (user_id, label, status) "
                    "VALUES (:uid, 'test', 'active') RETURNING id"
                ),
                {"uid": uid},
            )
        ).scalar_one()
        cid = (
            await conn.execute(
                text(
                    "INSERT INTO chats (account_id, tg_peer_id, type, title, write_mode, "
                    "sync_state, synced_total) VALUES "
                    "(:aid, floor(random() * 1e12)::bigint, 'supergroup', 'Test', "
                    "'read_only', 'idle', 0) RETURNING id"
                ),
                {"aid": aid},
            )
        ).scalar_one()
    return int(aid), int(cid)


async def _insert_message(engine: AsyncEngine, chat_id: int, tg_msg_id: int, body: str) -> int:
    async with engine.begin() as conn:
        return int(
            (
                await conn.execute(
                    text(
                        "INSERT INTO messages (chat_id, tg_msg_id, published_at, text) "
                        "VALUES (:c, :m, :d, :t) RETURNING id"
                    ),
                    {"c": chat_id, "m": tg_msg_id, "d": datetime.now(UTC), "t": body},
                )
            ).scalar_one()
        )


# ─── (a) append-only trigger ─────────────────────────────────────────────────


async def _agent_action(engine: AsyncEngine) -> tuple[int, int]:
    account_id, chat_id = await _fixture_chat(engine)
    async with engine.begin() as conn:
        uid = (
            await conn.execute(
                text("SELECT user_id FROM accounts WHERE id = :a"), {"a": account_id}
            )
        ).scalar_one()
        run_id = (
            await conn.execute(
                text(
                    "INSERT INTO agent_runs (user_id, account_id, chat_id, prompt, model) "
                    "VALUES (:u, :a, :c, 'test', 'test-model') RETURNING id"
                ),
                {"u": uid, "a": account_id, "c": chat_id},
            )
        ).scalar_one()
        action_id = (
            await conn.execute(
                text(
                    "INSERT INTO agent_actions (run_id, tool, args, status) "
                    "VALUES (:r, 'send_message', '{\"chat\": 1}'::jsonb, 'proposed') "
                    "RETURNING id"
                ),
                {"r": run_id},
            )
        ).scalar_one()
    return int(run_id), int(action_id)


async def test_agent_actions_delete_is_rejected(pg_engine: AsyncEngine) -> None:
    _, action_id = await _agent_action(pg_engine)
    with pytest.raises(Exception, match="append-only"):
        async with pg_engine.begin() as conn:
            await conn.execute(text("DELETE FROM agent_actions WHERE id = :i"), {"i": action_id})


@pytest.mark.parametrize(
    ("column", "value"),
    [("tool", "'delete_message'"), ("args", "'{\"evil\": true}'::jsonb")],
)
async def test_agent_actions_audit_fields_are_frozen(
    pg_engine: AsyncEngine, column: str, value: str
) -> None:
    _, action_id = await _agent_action(pg_engine)
    with pytest.raises(Exception, match="append-only"):
        async with pg_engine.begin() as conn:
            await conn.execute(
                text(f"UPDATE agent_actions SET {column} = {value} WHERE id = :i"),  # noqa: S608
                {"i": action_id},
            )


async def test_agent_actions_status_update_is_allowed(pg_engine: AsyncEngine) -> None:
    _, action_id = await _agent_action(pg_engine)
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE agent_actions SET status = 'executed', confirmed_at = now(), "
                "result_msg_id = 42 WHERE id = :i"
            ),
            {"i": action_id},
        )
        row = (
            await conn.execute(
                text("SELECT status, result_msg_id FROM agent_actions WHERE id = :i"),
                {"i": action_id},
            )
        ).one()
    assert row.status == "executed"
    assert row.result_msg_id == 42


# ─── (b) uz_fold: SQL ↔ Python pariteti ──────────────────────────────────────


async def test_uz_fold_sql_python_parity(pg_engine: AsyncEngine) -> None:
    assert len(FOLD_CORPUS) >= 40, "korpus kamida 40 satr bo'lishi kerak"
    async with pg_engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT s, uz_fold(s) AS folded FROM unnest(CAST(:corpus AS text[])) AS s"),
                {"corpus": list(FOLD_CORPUS)},
            )
        ).all()
    assert len(rows) == len(FOLD_CORPUS)
    mismatches = [
        (row.s, row.folded, uz_fold(row.s)) for row in rows if row.folded != uz_fold(row.s)
    ]
    assert not mismatches, f"SQL/Python drift: {mismatches}"


async def test_uz_norm_sql_matches_python(pg_engine: AsyncEngine) -> None:
    async with pg_engine.connect() as conn:
        rows = (
            await conn.execute(
                text("SELECT s, uz_norm(s) AS n FROM unnest(CAST(:corpus AS text[])) AS s"),
                {"corpus": list(FOLD_CORPUS)},
            )
        ).all()
    assert [row.n for row in rows] == [uz_norm(row.s) for row in rows]


async def test_uz_fold_is_immutable_and_strict(pg_engine: AsyncEngine) -> None:
    async with pg_engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT provolatile, proisstrict, proparallel FROM pg_proc "
                    "WHERE proname = 'uz_fold'"
                )
            )
        ).one()
    # pg_proc."char" ustunlari asyncpg'da bayt bo'lib qaytadi
    assert row.provolatile == b"i"  # IMMUTABLE — generated column uchun shart
    assert row.proisstrict is True
    assert row.proparallel == b"s"


# ─── (c) generated column'lar ────────────────────────────────────────────────


async def test_generated_columns_are_populated(pg_engine: AsyncEngine) -> None:
    _, chat_id = await _fixture_chat(pg_engine)
    mid = await _insert_message(pg_engine, chat_id, 1001, "Oʻzbekiston Respublikasi byudjeti 2026")
    async with pg_engine.connect() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT text_fold, tsv_uz::text AS tsv_uz, tsv_ru::text AS tsv_ru, "
                    "tsv_uz @@ to_tsquery('simple', 'ozbek:*') AS hit "
                    "FROM messages WHERE id = :i"
                ),
                {"i": mid},
            )
        ).one()
    assert row.text_fold == "ozbekiston respublikasi byudjeti 2026"
    assert "ozbekiston" in row.tsv_uz
    assert row.tsv_ru is not None
    assert row.hit is True


async def test_tsv_uz_matches_across_scripts(pg_engine: AsyncEngine) -> None:
    _, chat_id = await _fixture_chat(pg_engine)
    await _insert_message(pg_engine, chat_id, 2001, "Oʻzbekiston bo'yicha hisobot")
    await _insert_message(pg_engine, chat_id, 2002, "Ўзбекистон бўйича ҳисобот")
    await _insert_message(pg_engine, chat_id, 2003, "Kecha havo issiq edi")
    async with pg_engine.connect() as conn:
        ids = (
            (
                await conn.execute(
                    text(
                        "SELECT tg_msg_id FROM messages WHERE chat_id = :c "
                        "AND tsv_uz @@ to_tsquery('simple', 'ozbek:*') ORDER BY tg_msg_id"
                    ),
                    {"c": chat_id},
                )
            )
            .scalars()
            .all()
        )
    assert list(ids) == [2001, 2002]


async def test_generated_column_follows_edit(pg_engine: AsyncEngine) -> None:
    _, chat_id = await _fixture_chat(pg_engine)
    mid = await _insert_message(pg_engine, chat_id, 3001, "eski matn")
    async with pg_engine.begin() as conn:
        await conn.execute(
            text("UPDATE messages SET text = :t WHERE id = :i"),
            {"t": "Тўлов қилинди", "i": mid},
        )
        folded = (
            await conn.execute(text("SELECT text_fold FROM messages WHERE id = :i"), {"i": mid})
        ).scalar_one()
    assert folded == "tolov qilindi"


# ─── (d) trigram ─────────────────────────────────────────────────────────────


async def test_trigram_strict_word_similarity_matches(pg_engine: AsyncEngine) -> None:
    """`strict_word_similarity` so'z chegarasida o'lchaydi — `similarity` esa
    butun xabar uzunligiga nisbatan (uzun xabarda har doim past ball).

    Uchala variant ham fold'dan keyin bitta so'zga tushadi, shuning uchun
    kirill va apostrof farqi trigram yo'liga ham ta'sir qilmaydi.
    """
    _, chat_id = await _fixture_chat(pg_engine)
    await _insert_message(pg_engine, chat_id, 4001, "Oʻzbek tili darslari boshlandi")
    await _insert_message(pg_engine, chat_id, 4002, "Bugun futbol koramiz")
    await _insert_message(pg_engine, chat_id, 4003, "Ўзбек тили дарслари")
    async with pg_engine.connect() as conn:
        ids = (
            (
                await conn.execute(
                    text(
                        "SELECT tg_msg_id FROM messages WHERE chat_id = :c "
                        "AND strict_word_similarity('ozbek', text_fold) >= 0.55 "
                        "ORDER BY tg_msg_id"
                    ),
                    {"c": chat_id},
                )
            )
            .scalars()
            .all()
        )
    assert list(ids) == [4001, 4003]


async def test_trigram_tolerates_typos(pg_engine: AsyncEngine) -> None:
    """Imlo xatosi: `tolv` → `tolovlar` (bu yo'lning asosiy foydasi)."""
    _, chat_id = await _fixture_chat(pg_engine)
    await _insert_message(pg_engine, chat_id, 5001, "To'lovlar kechikdi, uzr")
    async with pg_engine.connect() as conn:
        score = (
            await conn.execute(
                text(
                    "SELECT strict_word_similarity('tolov', text_fold) FROM messages "
                    "WHERE chat_id = :c AND tg_msg_id = 5001"
                ),
                {"c": chat_id},
            )
        ).scalar_one()
    assert score >= 0.4


async def test_trigram_index_exists(pg_engine: AsyncEngine) -> None:
    async with pg_engine.connect() as conn:
        names = (
            (
                await conn.execute(
                    text("SELECT indexname FROM pg_indexes WHERE tablename = 'messages'")
                )
            )
            .scalars()
            .all()
        )
    for expected in ("ix_messages_tsv_uz", "ix_messages_tsv_ru", "ix_messages_fold_trgm"):
        assert expected in names
    # 0008 eski ifodali indekslarni olib tashlaydi
    assert "ix_messages_fts" not in names
    assert "ix_messages_trgm" not in names
