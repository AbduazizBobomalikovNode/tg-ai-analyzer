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

from contextlib import asynccontextmanager
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
    for expected in ("ix_messages_tsv_uz", "ix_messages_tsv_ru", "ix_messages_chat_trgm"):
        assert expected in names
    # 0008 eski ifodali indekslarni olib tashlaydi
    assert "ix_messages_fts" not in names
    assert "ix_messages_trgm" not in names
    # 0015: yolg'iz `text_fold` ustidagi trigram indeksi `(chat_id, text_fold)`
    # bilan almashtirildi — u ham kichikroq, ham "bitta chat ichida trigram"
    # so'roviga xizmat qiladi (o'lchov: docs/architecture-v2.md §11.1).
    assert "ix_messages_fold_trgm" not in names
    for expected in ("ix_messages_chat_msgid_alive", "ix_messages_chat_created_id"):
        assert expected in names


# ─── (e) 0009: retrieval_nodes + halfvec (phase2/A) ──────────────────────────


async def test_retrieval_nodes_tables_and_indexes_exist(pg_engine: AsyncEngine) -> None:
    """0009 qo'llanadi: jadvallar, GIN tsv va HNSW `halfvec_cosine_ops` indeksi.

    HNSW `halfvec` faqat pgvector >= 0.7 da bor (bu yerdagi image — 0.8.x).
    Indeks yo'qolsa vektor qidiruv **jimgina** ketma-ket skanga tushadi.
    """
    async with pg_engine.connect() as conn:
        node_idx = (
            (
                await conn.execute(
                    text("SELECT indexname FROM pg_indexes WHERE tablename = 'retrieval_nodes'")
                )
            )
            .scalars()
            .all()
        )
        emb_idx = (
            (
                await conn.execute(
                    text(
                        "SELECT indexdef FROM pg_indexes "
                        "WHERE tablename = 'retrieval_node_embeddings'"
                    )
                )
            )
            .scalars()
            .all()
        )
        coltype = (
            await conn.execute(
                text(
                    "SELECT format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
                    "WHERE a.attrelid = 'retrieval_node_embeddings'::regclass "
                    "AND a.attname = 'vector'"
                )
            )
        ).scalar_one()
    for expected in ("ix_nodes_tsv", "ix_nodes_chat_level_time", "ix_nodes_account_level_time"):
        assert expected in node_idx
    assert coltype == "halfvec(768)"
    assert any("hnsw" in d and "halfvec_cosine_ops" in d for d in emb_idx)


async def test_node_tsv_is_generated_from_prefix_and_text(pg_engine: AsyncEngine) -> None:
    """`tsv` — `ctx_prefix || ' ' || text` ustida, `uz_fold` bilan.

    Ya'ni kirill/apostrof farqi ham, prefiksdagi chat nomi ham leksik kanalga
    tushadi; qidiruv tomoni `search.fold_terms` bilan bir xil fazoda.
    """
    account_id, chat_id = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO retrieval_nodes (account_id, chat_id, level, starts_at, ends_at, "
                "msg_lo, msg_hi, msg_count, speakers, text, ctx_prefix, content_hash, simhash) "
                "VALUES (:a, :c, 1, now(), now(), 1, 9, 9, '{11,22}'::bigint[], "
                ":t, :p, 'deadbeef', :s)"
            ),
            {
                "a": account_id,
                "c": chat_id,
                "t": "Ўзбек тилида тўлов ҳақида",
                "p": "[chat: Ish guruhi (supergroup) | 2026-03-14 | speakers: user:11]",
                "s": -9223372036854775808,  # eng kichik bigint — belgili saqlash
            },
        )
    async with pg_engine.connect() as conn:
        found = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM retrieval_nodes WHERE chat_id = :c "
                    "AND tsv @@ to_tsquery('simple', :q)"
                ),
                {"c": chat_id, "q": f"{uz_fold('тўлов')}:*"},
            )
        ).scalar_one()
        by_prefix = (
            await conn.execute(
                text(
                    "SELECT count(*) FROM retrieval_nodes WHERE chat_id = :c "
                    "AND tsv @@ to_tsquery('simple', :q)"
                ),
                {"c": chat_id, "q": f"{uz_fold('guruhi')}:*"},
            )
        ).scalar_one()
    assert found == 1  # kirill matn lotin so'rovi bilan topildi
    assert by_prefix == 1  # prefiks ham indeksda


async def test_node_unique_key_and_halfvec_roundtrip(pg_engine: AsyncEngine) -> None:
    """`(chat_id, level, starts_at)` unikal; `halfvec` yozilib, cosine bilan o'qiladi."""
    account_id, chat_id = await _fixture_chat(pg_engine)
    when = datetime(2026, 3, 9, tzinfo=UTC)
    insert = text(
        "INSERT INTO retrieval_nodes (account_id, chat_id, level, starts_at, ends_at, "
        "msg_lo, msg_hi, msg_count, text, ctx_prefix, content_hash, simhash) "
        "VALUES (:a, :c, 1, :d, :d, 1, 2, 2, 'matn', 'pref', 'h1', 0) RETURNING id"
    )
    async with pg_engine.begin() as conn:
        node_id = (
            await conn.execute(insert, {"a": account_id, "c": chat_id, "d": when})
        ).scalar_one()
    with pytest.raises(Exception):  # noqa: B017 — asyncpg UniqueViolation
        async with pg_engine.begin() as conn:
            await conn.execute(insert, {"a": account_id, "c": chat_id, "d": when})

    vec = "[" + ",".join(["0.1"] * 768) + "]"
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO retrieval_node_embeddings (node_id, model, vector) "
                "VALUES (:n, 'gemini-embedding-001', CAST(:v AS halfvec))"
            ),
            {"n": node_id, "v": vec},
        )
    async with pg_engine.connect() as conn:
        dist = (
            await conn.execute(
                text(
                    "SELECT vector <=> CAST(:v AS halfvec) FROM retrieval_node_embeddings "
                    "WHERE node_id = :n"
                ),
                {"n": node_id, "v": vec},
            )
        ).scalar_one()
    assert abs(float(dist)) < 1e-6  # o'ziga cosine masofasi nol


async def test_node_rows_die_with_the_chat(pg_engine: AsyncEngine) -> None:
    """`ON DELETE CASCADE` — chat o'chsa tugun ham, vektori ham qolmaydi."""
    account_id, chat_id = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        node_id = (
            await conn.execute(
                text(
                    "INSERT INTO retrieval_nodes (account_id, chat_id, level, starts_at, "
                    "ends_at, msg_lo, msg_hi, msg_count, text, ctx_prefix, content_hash, "
                    "simhash) VALUES (:a, :c, 2, now(), now(), 1, 2, 2, 't', 'p', 'h', 0) "
                    "RETURNING id"
                ),
                {"a": account_id, "c": chat_id},
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO retrieval_node_embeddings (node_id, model, vector) "
                "VALUES (:n, 'm', CAST(:v AS halfvec))"
            ),
            {"n": node_id, "v": "[" + ",".join(["0"] * 768) + "]"},
        )
        await conn.execute(text("DELETE FROM chats WHERE id = :c"), {"c": chat_id})
    async with pg_engine.connect() as conn:
        left = (
            await conn.execute(
                text(
                    "SELECT (SELECT count(*) FROM retrieval_nodes WHERE id = :n) + "
                    "(SELECT count(*) FROM retrieval_node_embeddings WHERE node_id = :n)"
                ),
                {"n": node_id},
            )
        ).scalar_one()
    assert left == 0


# ─── migratsiya 0011: alertlar ──────────────────────────────────────────────


async def test_alert_rules_halfvec_column_and_dedup(pg_engine: AsyncEngine) -> None:
    """`alert_rules.embedding` — `halfvec(768)`; `uq_alert_event` takrorni to'sadi."""
    account_id, chat_id = await _fixture_chat(pg_engine)
    vec = "[" + ",".join(["0.01"] * 768) + "]"

    async with pg_engine.begin() as conn:
        rule_id = (
            await conn.execute(
                text(
                    "INSERT INTO alert_rules (account_id, chat_id, kind, label, pattern, "
                    "embedding, threshold, enabled, cooldown_minutes) "
                    "VALUES (:a, :c, 'semantic', 'test', 'ish haqida shikoyat', "
                    "CAST(:v AS halfvec), 0.82, true, 30) RETURNING id"
                ),
                {"a": account_id, "c": chat_id, "v": vec},
            )
        ).scalar_one()
        msg_id = (
            await conn.execute(
                text(
                    "INSERT INTO messages (chat_id, tg_msg_id, sender_id, published_at, text) "
                    "VALUES (:c, 1, 42, now(), 'to''lov qilindi') RETURNING id"
                ),
                {"c": chat_id},
            )
        ).scalar_one()

    # ustun turi va `<=>` operatori ishlaydi (halfvec_cosine_ops)
    async with pg_engine.connect() as conn:
        udt = (
            await conn.execute(
                text(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_name = 'alert_rules' AND column_name = 'embedding'"
                )
            )
        ).scalar_one()
        assert udt == "halfvec"
        dist = (
            await conn.execute(
                text("SELECT embedding <=> CAST(:v AS halfvec) FROM alert_rules WHERE id = :r"),
                {"r": rule_id, "v": vec},
            )
        ).scalar_one()
        assert abs(float(dist)) < 1e-6

    # `ON CONFLICT DO NOTHING` — ikkinchi hodisa yozilmaydi
    async with pg_engine.begin() as conn:
        for _ in range(2):
            await conn.execute(
                text(
                    "INSERT INTO alert_events (rule_id, message_id, chat_id, tg_msg_id, score) "
                    "VALUES (:r, :m, :c, 1, 0.9) ON CONFLICT (rule_id, message_id) DO NOTHING"
                ),
                {"r": rule_id, "m": msg_id, "c": chat_id},
            )
    async with pg_engine.connect() as conn:
        n = (
            await conn.execute(
                text("SELECT count(*) FROM alert_events WHERE rule_id = :r"), {"r": rule_id}
            )
        ).scalar_one()
    assert n == 1


async def test_alert_rows_die_with_the_chat(pg_engine: AsyncEngine) -> None:
    """Chat o'chsa qoida ham, hodisalari ham qolmaydi (`ON DELETE CASCADE`)."""
    account_id, chat_id = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        rule_id = (
            await conn.execute(
                text(
                    "INSERT INTO alert_rules (account_id, chat_id, kind, pattern) "
                    "VALUES (:a, :c, 'keyword', 'tolov') RETURNING id"
                ),
                {"a": account_id, "c": chat_id},
            )
        ).scalar_one()
        msg_id = (
            await conn.execute(
                text(
                    "INSERT INTO messages (chat_id, tg_msg_id, published_at, text) "
                    "VALUES (:c, 7, now(), 'x') RETURNING id"
                ),
                {"c": chat_id},
            )
        ).scalar_one()
        await conn.execute(
            text(
                "INSERT INTO alert_events (rule_id, message_id, chat_id, tg_msg_id) "
                "VALUES (:r, :m, :c, 7)"
            ),
            {"r": rule_id, "m": msg_id, "c": chat_id},
        )
        await conn.execute(text("DELETE FROM chats WHERE id = :c"), {"c": chat_id})
    async with pg_engine.connect() as conn:
        left = (
            await conn.execute(
                text(
                    "SELECT (SELECT count(*) FROM alert_rules WHERE id = :r) + "
                    "(SELECT count(*) FROM alert_events WHERE rule_id = :r)"
                ),
                {"r": rule_id},
            )
        ).scalar_one()
    assert left == 0


# ─── (f) phase3/B: migratsiya 0010 — briefing ustunlari va jadvali ───────────


async def test_account_briefing_columns_have_defaults(pg_engine: AsyncEngine) -> None:
    """Yangi akkaunt avtomatik `Asia/Tashkent`, o'chiq briefing, soat 8 bilan keladi."""
    account_id, _ = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT tz, quiet_from, quiet_to, briefing_enabled, briefing_hour, "
                    "briefing_chats FROM accounts WHERE id = :a"
                ),
                {"a": account_id},
            )
        ).one()
    assert row.tz == "Asia/Tashkent"
    assert row.quiet_from is None and row.quiet_to is None
    assert row.briefing_enabled is False
    assert row.briefing_hour == 8
    assert row.briefing_chats is None


async def test_quiet_hours_store_local_time_of_day(pg_engine: AsyncEngine) -> None:
    """`quiet_from`/`quiet_to` — timezone'siz `time`, yarim tun orqali o'ralishi mumkin."""
    account_id, _ = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE accounts SET tz = 'Europe/Moscow', quiet_from = '22:00', "
                "quiet_to = '07:30', briefing_enabled = true, briefing_hour = 9, "
                "briefing_chats = CAST(:chats AS jsonb) WHERE id = :a"
            ),
            {"a": account_id, "chats": "[1, 2, 3]"},
        )
        row = (
            await conn.execute(
                text(
                    "SELECT tz, quiet_from, quiet_to, briefing_hour, briefing_chats "
                    "FROM accounts WHERE id = :a"
                ),
                {"a": account_id},
            )
        ).one()
    assert row.tz == "Europe/Moscow"
    assert (row.quiet_from.hour, row.quiet_from.minute) == (22, 0)
    assert (row.quiet_to.hour, row.quiet_to.minute) == (7, 30)
    assert row.quiet_from > row.quiet_to  # o'ralgan oyna
    assert row.briefing_hour == 9 and row.briefing_chats == [1, 2, 3]


async def test_briefing_is_unique_per_account_and_day(pg_engine: AsyncEngine) -> None:
    """`uq_briefing_account_day` — cron idempotentligi shu kalitga tayanadi."""
    from sqlalchemy.exc import IntegrityError

    account_id, _ = await _fixture_chat(pg_engine)
    ins = text(
        "INSERT INTO briefings (account_id, day, text, structured, tokens_in, tokens_out) "
        "VALUES (:a, DATE '2026-09-14', 'tldr', CAST(:s AS jsonb), 10, 5)"
    )
    async with pg_engine.begin() as conn:
        await conn.execute(ins, {"a": account_id, "s": '{"tldr": "x"}'})
    with pytest.raises(IntegrityError):
        async with pg_engine.begin() as conn:
            await conn.execute(ins, {"a": account_id, "s": "{}"})

    async with pg_engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT structured, cost_usd, delivered_at, delivery_error, created_at "
                    "FROM briefings WHERE account_id = :a"
                ),
                {"a": account_id},
            )
        ).one()
    assert row.structured == {"tldr": "x"}
    assert row.cost_usd is None and row.delivered_at is None and row.delivery_error is None
    assert row.created_at is not None


async def test_briefing_rows_die_with_the_account(pg_engine: AsyncEngine) -> None:
    account_id, _ = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO briefings (account_id, day, text) VALUES (:a, DATE '2026-09-13', 'x')"
            ),
            {"a": account_id},
        )
        await conn.execute(text("DELETE FROM accounts WHERE id = :a"), {"a": account_id})
        left = (
            await conn.execute(
                text("SELECT count(*) FROM briefings WHERE account_id = :a"), {"a": account_id}
            )
        ).scalar_one()
    assert left == 0


# ─── phase4/A: `node_entities` (migratsiya 0012) ─────────────────────────────


async def test_node_entities_table_exists_with_expected_shape(pg_engine: AsyncEngine) -> None:
    """Migratsiya 0012 qo'llanadi; ustunlar, indekslar va `CHECK` joyida."""
    async with pg_engine.begin() as conn:
        cols = {
            r.column_name: r.is_nullable
            for r in (
                await conn.execute(
                    text(
                        "SELECT column_name, is_nullable FROM information_schema.columns "
                        "WHERE table_name = 'node_entities'"
                    )
                )
            ).all()
        }
        indexes = {
            r.indexname
            for r in (
                await conn.execute(
                    text("SELECT indexname FROM pg_indexes WHERE tablename = 'node_entities'")
                )
            ).all()
        }
    assert cols["message_id"] == "YES" and cols["node_id"] == "YES"  # ikkalasi nullable
    assert cols["account_id"] == "NO" and cols["kind"] == "NO" and cols["value"] == "NO"
    assert "seen_at" in cols and "value_raw" in cols
    assert {
        "ix_node_entities_account_kind_value",
        "ix_node_entities_chat_seen",
        "ix_node_entities_node",
    } <= indexes


async def test_node_entity_is_unique_per_message_kind_value(pg_engine: AsyncEngine) -> None:
    """`uq_node_entity_value` — qayta sync `ON CONFLICT DO NOTHING` ga tayanadi."""
    from sqlalchemy.exc import IntegrityError

    account_id, chat_id = await _fixture_chat(pg_engine)
    mid = await _insert_message(pg_engine, chat_id, 1, "@nurbek keldi")
    ins = text(
        "INSERT INTO node_entities (account_id, chat_id, message_id, kind, value, value_raw) "
        "VALUES (:a, :c, :m, 'username', '@nurbek', '@Nurbek')"
    )
    async with pg_engine.begin() as conn:
        await conn.execute(ins, {"a": account_id, "c": chat_id, "m": mid})
    with pytest.raises(IntegrityError):
        async with pg_engine.begin() as conn:
            await conn.execute(ins, {"a": account_id, "c": chat_id, "m": mid})
    # boshqa `kind` — konflikt emas
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO node_entities (account_id, chat_id, message_id, kind, value) "
                "VALUES (:a, :c, :m, 'hashtag', '@nurbek')"
            ),
            {"a": account_id, "c": chat_id, "m": mid},
        )


async def test_phone_like_kind_is_rejected_by_the_database(pg_engine: AsyncEngine) -> None:
    """Telefon raqami indeksga tushmasligi kod xushmuomalaligiga tayanmaydi."""
    from sqlalchemy.exc import IntegrityError

    account_id, chat_id = await _fixture_chat(pg_engine)
    mid = await _insert_message(pg_engine, chat_id, 2, "tel +998901234567")
    with pytest.raises(IntegrityError):
        async with pg_engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO node_entities (account_id, chat_id, message_id, kind, value) "
                    "VALUES (:a, :c, :m, 'phone_like', '+998901234567')"
                ),
                {"a": account_id, "c": chat_id, "m": mid},
            )


async def test_node_entities_die_with_the_message(pg_engine: AsyncEngine) -> None:
    account_id, chat_id = await _fixture_chat(pg_engine)
    mid = await _insert_message(pg_engine, chat_id, 3, "#deploy")
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO node_entities (account_id, chat_id, message_id, kind, value) "
                "VALUES (:a, :c, :m, 'hashtag', '#deploy')"
            ),
            {"a": account_id, "c": chat_id, "m": mid},
        )
        await conn.execute(text("DELETE FROM messages WHERE id = :m"), {"m": mid})
        left = (
            await conn.execute(
                text("SELECT count(*) FROM node_entities WHERE message_id = :m"), {"m": mid}
            )
        ).scalar_one()
    assert left == 0


async def test_episode_link_fills_node_id_for_entities_in_range(pg_engine: AsyncEngine) -> None:
    """`episodes.link_entities` — bitta `UPDATE`, oyna `msg_lo..msg_hi` bo'yicha."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.services.episodes import link_entities

    account_id, chat_id = await _fixture_chat(pg_engine)
    inside = await _insert_message(pg_engine, chat_id, 10, "@nurbek")
    outside = await _insert_message(pg_engine, chat_id, 99, "@nurbek ham")
    async with pg_engine.begin() as conn:
        node_id = (
            await conn.execute(
                text(
                    "INSERT INTO retrieval_nodes (account_id, chat_id, level, starts_at, ends_at, "
                    "msg_lo, msg_hi, msg_count, content_hash) VALUES "
                    "(:a, :c, 1, now(), now(), 5, 20, 2, 'x') RETURNING id"
                ),
                {"a": account_id, "c": chat_id},
            )
        ).scalar_one()
        for mid in (inside, outside):
            await conn.execute(
                text(
                    "INSERT INTO node_entities (account_id, chat_id, message_id, kind, value) "
                    "VALUES (:a, :c, :m, 'username', '@nurbek')"
                ),
                {"a": account_id, "c": chat_id, "m": mid},
            )

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    async with maker() as session:
        changed = await link_entities(session, chat_id)
        await session.commit()
    assert changed == 1

    async with pg_engine.begin() as conn:
        rows = dict(
            (
                await conn.execute(
                    text("SELECT message_id, node_id FROM node_entities WHERE chat_id = :c"),
                    {"c": chat_id},
                )
            ).all()
        )
    assert rows[inside] == node_id and rows[outside] is None


# ─── (h) phase4/C: `embed_batches` (migratsiya 0013) ─────────────────────────
#
# Nima qo'riqlanadi: tartiblangan `item_ids` ↔ vektorlar mosligi. Bu yagona
# bog'lovchi — Gemini javoblarni kirish tartibida qaytaradi, boshqa kalit yo'q.
# Uzunlik mos kelmasa **hech narsa** yozilmasligi shart (0-bosqich qoidasi):
# yarim yozish vektorni boshqa xabarga tushirib, indeksni jimgina buzardi.


class _FakeBatchProvider:
    """`GeminiProvider` o'rnini bosadi — hech qanday tarmoq chaqiruvi yo'q."""

    name = "gemini"

    def __init__(self, *, states: list[str], vectors: list[list[float]] | None = None) -> None:
        self.states = states  # har `poll` chaqiruvida keyingisi
        self.vectors = vectors or []
        self.submitted: list[list[str]] = []

    async def embed_batch_submit(self, texts, **kw):  # type: ignore[no-untyped-def]
        from uuid import uuid4

        self.submitted.append(list(texts))
        # Baza testlar orasida tozalanmaydi — nom global unikal bo'lishi shart.
        return f"batches/{uuid4().hex}"

    async def embed_batch_poll(self, job: str):  # type: ignore[no-untyped-def]
        from app.llm.base import BatchState, BatchStatus

        raw = self.states.pop(0) if self.states else "JOB_STATE_SUCCEEDED"
        mapping = {
            "JOB_STATE_PENDING": BatchState.PENDING,
            "JOB_STATE_RUNNING": BatchState.RUNNING,
            "JOB_STATE_SUCCEEDED": BatchState.SUCCEEDED,
            "JOB_STATE_EXPIRED": BatchState.EXPIRED,
            "JOB_STATE_FAILED": BatchState.FAILED,
        }
        return BatchStatus(job=job, state=mapping[raw], raw_state=raw, model=_EMB_MODEL)

    async def embed_batch_fetch(self, job: str, *, model: str = ""):  # type: ignore[no-untyped-def]
        from app.llm.base import EmbedResult, Usage

        return EmbedResult(
            vectors=self.vectors,
            model=_EMB_MODEL,
            provider="gemini",
            usage=Usage(tokens_in=100),
        )


_EMB_MODEL = "gemini-embedding-001"


def _vec(seed: float, dims: int) -> list[float]:
    return [seed] + [0.0] * (dims - 1)


async def _clear_inflight_batches(pg_engine: AsyncEngine) -> None:
    """`poll_batches` **global** ishlaydi (har qanday uchayotgan job).

    Bu ishlab chiqarishda to'g'ri, testda esa oldingi testning qatori
    hisoblarni buzadi — shuning uchun poll testlari o'z jobidan boshqasini
    qoldirmaydi.
    """
    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE embed_batches SET status = 'succeeded' "
                "WHERE status IN ('pending', 'running')"
            )
        )


def _maker(pg_engine: AsyncEngine):  # type: ignore[no-untyped-def]
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(pg_engine, expire_on_commit=False)


async def _synced_chat(pg_engine: AsyncEngine) -> tuple[int, int]:
    """`(account_id, chat_id)`. Testlar **akkaunt bo'yicha** ajratiladi.

    Baza testlar orasida tozalanmaydi, `submit_messages` esa vektori yo'q
    hamma xabarni ko'radi — akkaunt filtri bo'lmasa oldingi testlarning
    xabarlari ham jobga tushardi.
    """
    account_id, chat_id = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        await conn.execute(text("UPDATE chats SET synced_total = 10 WHERE id = :c"), {"c": chat_id})
    return account_id, chat_id


async def test_embed_batches_table_and_gin_index_exist(pg_engine: AsyncEngine) -> None:
    async with pg_engine.begin() as conn:
        cols = dict(
            (
                await conn.execute(
                    text(
                        "SELECT column_name, data_type FROM information_schema.columns "
                        "WHERE table_name = 'embed_batches'"
                    )
                )
            ).all()
        )
        indexes = {
            r[0]
            for r in (
                await conn.execute(
                    text("SELECT indexname FROM pg_indexes WHERE tablename = 'embed_batches'")
                )
            ).all()
        }
    assert cols["item_ids"] == "jsonb"
    assert cols["account_id"] == "integer"
    assert {"ix_embed_batches_items", "ix_embed_batches_status"} <= indexes


async def test_embed_batches_kind_is_constrained(pg_engine: AsyncEngine) -> None:
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        async with pg_engine.begin() as conn:
            await conn.execute(
                text("INSERT INTO embed_batches (job_name, kind) VALUES ('b/x', 'digests')")
            )


async def test_submit_records_ordered_ids_and_excludes_inflight(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.services import embed_batches as eb

    account_id, chat_id = await _synced_chat(pg_engine)
    ids = [
        await _insert_message(pg_engine, chat_id, 9000 + i, f"yetarlicha uzun xabar matni {i}")
        for i in range(3)
    ]

    provider = _FakeBatchProvider(states=[])
    monkeypatch.setattr(eb, "_provider_for_batch", lambda: (provider, _EMB_MODEL))

    async with _maker(pg_engine)() as session:
        rep = await eb.submit_messages(session, limit=50, account_id=account_id)
        await session.commit()
    assert rep.submitted == 3

    async with pg_engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT item_ids, status, kind, attempts FROM embed_batches WHERE job_name = :j"
                ),
                {"j": rep.job},
            )
        ).one()
    # `Message.id DESC` tartibida yuborilgan — aynan shu tartib saqlanishi shart.
    assert row[0] == sorted(ids, reverse=True)
    assert (row[1], row[2], row[3]) == ("pending", "messages", 1)

    # Ikkinchi submit: hamma element uchayotgan jobda → hech narsa yuborilmaydi.
    async with _maker(pg_engine)() as session:
        again = await eb.submit_messages(session, limit=50, account_id=account_id)
    assert again.submitted == 0
    assert len(provider.submitted) == 1


async def test_poll_writes_vectors_strictly_aligned(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    from app.services import embed_batches as eb

    account_id, chat_id = await _synced_chat(pg_engine)
    ids = [
        await _insert_message(pg_engine, chat_id, 9100 + i, f"yana bir uzunroq xabar matni {i}")
        for i in range(3)
    ]
    ordered = sorted(ids, reverse=True)
    dims = settings.embed_dims
    vectors = [_vec(1.0, dims), _vec(2.0, dims), _vec(3.0, dims)]

    provider = _FakeBatchProvider(
        states=["JOB_STATE_RUNNING", "JOB_STATE_SUCCEEDED"], vectors=vectors
    )
    monkeypatch.setattr(eb, "_provider_for_batch", lambda: (provider, _EMB_MODEL))
    await _clear_inflight_batches(pg_engine)

    async with _maker(pg_engine)() as session:
        rep = await eb.submit_messages(session, limit=50, account_id=account_id)
        await session.commit()

    # 1-poll: hali ishlayapti — vektor yozilmaydi, holat `running` bo'ladi.
    async with _maker(pg_engine)() as session:
        first = await eb.poll_batches(session)
        await session.commit()
    assert (first.still_running, first.written) == (1, 0)
    async with pg_engine.begin() as conn:
        assert (
            await conn.execute(
                text("SELECT status FROM embed_batches WHERE job_name = :j"), {"j": rep.job}
            )
        ).scalar_one() == "running"

    # 2-poll: tugadi — vektorlar `item_ids` tartibida yoziladi.
    async with _maker(pg_engine)() as session:
        second = await eb.poll_batches(session)
        await session.commit()
    assert (second.succeeded, second.written) == (1, 3)

    async with pg_engine.begin() as conn:
        stored = dict(
            (
                await conn.execute(
                    text(
                        "SELECT message_id, (vector::real[])[1] FROM message_embeddings "
                        "WHERE message_id = ANY(:ids)"
                    ),
                    {"ids": ordered},
                )
            ).all()
        )
        batch = (
            await conn.execute(
                text("SELECT status, model, cost_usd FROM embed_batches WHERE job_name = :j"),
                {"j": rep.job},
            )
        ).one()
    # vectors[i] ↔ item_ids[i] — moslik aynan tartib bo'yicha. Har vektorning
    # birinchi komponenti farqli (1/2/3), shuning uchun almashib ketish
    # darhol ko'rinadi. (Normallashtirish adapter ichida, soxta provider
    # xom qiymat qaytaradi — bu test aynan moslikni o'lchaydi.)
    assert [stored[mid] for mid in ordered] == pytest.approx([1.0, 2.0, 3.0])
    assert batch[0] == "succeeded"
    assert batch[1] == _EMB_MODEL
    assert batch[2] > 0  # batch narxi yozildi


async def test_poll_mismatch_writes_nothing(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch, settings
) -> None:
    from app.services import embed_batches as eb

    account_id, chat_id = await _synced_chat(pg_engine)
    ids = [
        await _insert_message(pg_engine, chat_id, 9200 + i, f"uchinchi to'plam xabari {i}")
        for i in range(3)
    ]

    # 3 element yuborildi, 2 vektor qaytdi.
    provider = _FakeBatchProvider(
        states=["JOB_STATE_SUCCEEDED"],
        vectors=[_vec(1.0, settings.embed_dims), _vec(2.0, settings.embed_dims)],
    )
    monkeypatch.setattr(eb, "_provider_for_batch", lambda: (provider, _EMB_MODEL))
    await _clear_inflight_batches(pg_engine)

    async with _maker(pg_engine)() as session:
        rep = await eb.submit_messages(session, limit=50, account_id=account_id)
        await session.commit()
    async with _maker(pg_engine)() as session:
        out = await eb.poll_batches(session)
        await session.commit()

    assert (out.written, out.failed) == (0, 1)
    async with pg_engine.begin() as conn:
        written = (
            await conn.execute(
                text("SELECT count(*) FROM message_embeddings WHERE message_id = ANY(:ids)"),
                {"ids": ids},
            )
        ).scalar_one()
        row = (
            await conn.execute(
                text("SELECT status, attempts, error FROM embed_batches WHERE job_name = :j"),
                {"j": rep.job},
            )
        ).one()
    assert written == 0  # yarim yozish yo'q
    assert row[0] == "failed" and row[1] == 2
    assert "misaligned" in row[2]


async def test_expired_job_increments_attempts_then_goes_permanent(
    pg_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cheksiz qayta yuborish = cheksiz to'lov. `max_attempts` dan keyin to'xtaydi."""
    from app.services import embed_batches as eb

    account_id, chat_id = await _synced_chat(pg_engine)
    await _insert_message(pg_engine, chat_id, 9300, "eskirgan job uchun xabar matni")

    provider = _FakeBatchProvider(states=["JOB_STATE_EXPIRED"])
    monkeypatch.setattr(eb, "_provider_for_batch", lambda: (provider, _EMB_MODEL))
    await _clear_inflight_batches(pg_engine)

    async with _maker(pg_engine)() as session:
        rep = await eb.submit_messages(session, limit=50, account_id=account_id)
        await session.commit()

    async with _maker(pg_engine)() as session:
        out = await eb.poll_batches(session)
        await session.commit()
    assert out.failed == 1 and out.permanent == 0
    async with pg_engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT status, attempts FROM embed_batches WHERE job_name = :j"),
                {"j": rep.job},
            )
        ).one()
    assert tuple(row) == ("expired", 2)

    # `expired` — uchayotgan emas: element keyingi yugurishda qayta olinadi.
    provider.states = ["JOB_STATE_EXPIRED"]
    async with _maker(pg_engine)() as session:
        again = await eb.submit_messages(session, limit=50, account_id=account_id)
        await session.commit()
    assert again.submitted == 1

    # Ikkinchi urinishdan keyin `max_attempts` tugaydi → `failed_permanent`.
    async with pg_engine.begin() as conn:
        await conn.execute(
            text("UPDATE embed_batches SET attempts = 2 WHERE job_name = :j"), {"j": again.job}
        )
    async with _maker(pg_engine)() as session:
        out2 = await eb.poll_batches(session)
        await session.commit()
    assert out2.permanent == 1
    async with pg_engine.begin() as conn:
        assert (
            await conn.execute(
                text("SELECT status FROM embed_batches WHERE job_name = :j"), {"j": again.job}
            )
        ).scalar_one() == "failed_permanent"


# ─── F1/H2: uzilgan deep yugurishning hisob qatori ───────────────────────────


async def test_aborted_deep_run_is_counted_by_the_daily_budget(pg_engine: AsyncEngine) -> None:
    """Uzilgan yugurish `conversation_messages` ga tushadi va byudjetni ko'taradi.

    Aynan shu qator bo'lmasa, sintez oldidan uzilgan (yoki `LLMError` bergan)
    deep so'rov bepul bo'lib qolardi: `limits.daily_usage` faqat assistant
    qatorlarini yig'adi.
    """
    from app.services import deep_search as DS
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
                    "VALUES (:u, :a, 'deep') RETURNING id"
                ),
                {"u": user_id, "a": account_id},
            )
        ).scalar_one()

    maker = _maker(pg_engine)

    @asynccontextmanager
    async def factory():  # type: ignore[no-untyped-def]
        """`db.base.session_scope` kabi — chiqishda **commit** qiladi."""
        async with maker() as session:
            yield session
            await session.commit()

    async with maker() as session:
        before = await LIM.daily_usage(session, int(user_id))

    meter = DS._Meter.__new__(DS._Meter)
    meter.inner = None  # type: ignore[assignment]
    meter.tokens_in, meter.tokens_out, meter.cost_usd, meter.calls = 1_234, 56, 0.004, 3

    await DS._persist_usage(
        factory,
        conversation_id=int(conv_id),
        user_id=int(user_id),
        account_id=int(account_id),
        question="to'lov haqida",
        meter=meter,
        status="cancelled",
        latency_ms=900,
    )

    async with maker() as session:
        after = await LIM.daily_usage(session, int(user_id))
    assert after.tokens - before.tokens == 1_290
    assert after.cost_usd > before.cost_usd

    async with pg_engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT role, content, context, task FROM conversation_messages "
                    "WHERE conversation_id = :c ORDER BY id DESC LIMIT 1"
                ),
                {"c": conv_id},
            )
        ).one()
        assert row[0] == "assistant" and row[1] == "" and row[3] == "deep"
        assert row[2]["aborted"] is True and row[2]["status"] == "cancelled"
        run = (
            await conn.execute(
                text(
                    "SELECT tokens_in, tokens_out, error FROM agent_runs "
                    "WHERE user_id = :u ORDER BY id DESC LIMIT 1"
                ),
                {"u": user_id},
            )
        ).one()
        assert tuple(run) == (1_234, 56, "deep:cancelled")


# ─── ui/R1a (migratsiya 0017): chat profili ustunlari ────────────────────────


async def test_chat_profile_columns_exist_with_defaults(pg_engine: AsyncEngine) -> None:
    """0017 qo'llanadi: yangi ustunlar bor, bool/int sukutlari to'ldiriladi."""
    account_id, chat_id = await _fixture_chat(pg_engine)
    assert account_id
    async with pg_engine.begin() as conn:
        row = (
            await conn.execute(
                text(
                    "SELECT about, is_verified, is_scam, is_fake, unread_count, "
                    "last_message_text, avatar_photo_id, avatar_mime, avatar_updated_at, "
                    "avatar_error FROM chats WHERE id = :c"
                ),
                {"c": chat_id},
            )
        ).one()
    assert row[0] is None
    assert (row[1], row[2], row[3]) == (False, False, False)
    assert row[4] == 0
    assert all(v is None for v in row[5:])

    async with pg_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE chats SET about = :a, is_verified = true, unread_count = 5, "
                "last_message_text = :t, avatar_photo_id = :p, avatar_mime = 'image/jpeg' "
                "WHERE id = :c"
            ),
            {"a": "Kanal haqida", "t": "salom", "p": 9_223_372_036_854_775_806, "c": chat_id},
        )
        got = (
            await conn.execute(
                text("SELECT avatar_photo_id, unread_count FROM chats WHERE id = :c"),
                {"c": chat_id},
            )
        ).one()
    assert got == (9_223_372_036_854_775_806, 5)  # BigInteger — photo_id sig'adi


async def test_chat_search_sql_uses_uz_fold(pg_engine: AsyncEngine) -> None:
    """`chatdata._search_sql` haqiqiy bazada ishlaydi va fold fazosida topadi."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from app.db.models import Chat
    from app.web.routers import chatdata as CD

    account_id, chat_id = await _fixture_chat(pg_engine)
    async with pg_engine.begin() as conn:
        await conn.execute(
            text("UPDATE chats SET title = :t, username = 'tashnews', about = :a WHERE id = :c"),
            {"t": "Oʻzbekiston yangiliklari", "a": "Ўзбекистон haqida", "c": chat_id},
        )

    maker = async_sessionmaker(pg_engine, expire_on_commit=False)
    conds = [Chat.account_id == account_id]
    async with maker() as session:
        for query in ("O'zbekiston", "ozbekiston", "Ўзбекистон", "OʻZBEK"):
            rows = await CD._search_sql(session, conds, uz_fold(query), 10)
            assert [c.id for c in rows] == [chat_id], query
        # prefiks (3) — so'z ichidagi moslikdan (1) yuqori turadi
        assert await CD._search_sql(session, conds, uz_fold("yangiliklari"), 10)
        # about bo'yicha ham topiladi, mos kelmaydigan so'rov — bo'sh
        assert not await CD._search_sql(session, conds, "qwertyxyz", 10)
        # LIKE maxsus belgilari literal sifatida qaraladi (SQL injection emas)
        assert not await CD._search_sql(session, conds, "%", 10)
