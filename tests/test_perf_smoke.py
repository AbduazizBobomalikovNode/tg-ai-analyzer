"""Miqyos regressiya qopqog'i — kichik korpus, lekin **haqiqiy** so'rov shakli.

`evals/perf/bench.py` to'liq o'lchov (300k xabar, bir necha daqiqa); bu yerda
esa 2 ming xabarlik korpus va ikki savol:

  1. o'lchanadigan har bir chaqiruv 2 soniyadan tez tugaydimi;
  2. katta leksik kanal (trigram) **indeksdan** o'qiydimi.

Ikkinchisi muhimroq: vaqt mashinaga qarab tebranadi, plan esa yo'q.
`search._run_trgm` ilgari `strict_word_similarity(...) > 0.55` ni yolg'iz
`WHERE` ga qo'yardi — bu funksiya chaqirig'i, Postgres uni indekslay olmaydi
va chatdagi **har** qatorda hisoblardi. Endi so'rov ikki qavatli: ichki qavat
`<<%` operatori bilan (GIN, `ix_messages_chat_trgm`), tashqi qavat aniq
ostona bilan. Agar kimdir operatorni olib tashlasa yoki 0015-indeksni
o'chirsa, `EXPLAIN` da indeks nomi yo'qoladi va shu test yiqiladi.

DB'siz — skip (`TEST_DATABASE_URL`).
"""

from __future__ import annotations

import random
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

pytestmark = [pytest.mark.asyncio]

SMOKE_MESSAGES = 2_000
#: Har chaqiruv uchun qattiq shift. Bu "tez" emas, "miqyos buzilmadi" chegarasi.
MAX_SECONDS = 2.0
#: Trigram kanali shu indeksdan o'qishi kerak (migratsiya 0015).
TRGM_INDEX = "ix_messages_chat_trgm"


def _unit(seed: int, dim: int = 768) -> str:
    rng = random.Random(seed)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return "[" + ",".join(f"{x / norm:.5f}" for x in v) + "]"


@pytest.fixture(scope="module")
def sessionmaker_(pg_engine: AsyncEngine) -> Any:
    return async_sessionmaker(pg_engine, expire_on_commit=False)


@pytest.fixture(scope="module")
async def corpus(sessionmaker_: Any) -> dict[str, int]:
    """2k xabarlik korpus — bitta chat, yarmida embedding."""
    from app.llm import embed_model_name

    words = ["to'lov", "muddat", "оплата", "договор", "payment", "deadline", "deploy", "server"]
    rng = random.Random(42)
    model = embed_model_name()
    async with sessionmaker_() as s:
        await s.execute(
            text(
                "TRUNCATE users, accounts, chats, messages, message_embeddings, "
                "retrieval_nodes, retrieval_node_embeddings, node_entities "
                "RESTART IDENTITY CASCADE"
            )
        )
        uid = (
            await s.execute(
                text(
                    "INSERT INTO users (tg_user_id, username, locale) "
                    "VALUES (99001, 'perf', 'uz') RETURNING id"
                )
            )
        ).scalar_one()
        aid = (
            await s.execute(
                text(
                    "INSERT INTO accounts (user_id, tg_account_id, label, status) "
                    "VALUES (:u, 99001, 'perf', 'active') RETURNING id"
                ),
                {"u": uid},
            )
        ).scalar_one()
        cid = (
            await s.execute(
                text(
                    "INSERT INTO chats (account_id, tg_peer_id, type, title, username, "
                    "write_mode, sync_state, synced_total) "
                    "VALUES (:a, 555001, 'supergroup', 'Perf', 'perf', 'read_only', "
                    "'done', :n) RETURNING id"
                ),
                {"a": aid, "n": SMOKE_MESSAGES},
            )
        ).scalar_one()
        base = datetime.now(UTC) - timedelta(days=90)
        rows = [
            {
                "c": cid,
                "t": i + 1,
                "s": 900_000 + (i % 7),
                "p": base + timedelta(minutes=13 * i),
                "x": " ".join(rng.choice(words) for _ in range(rng.randint(4, 20))),
            }
            for i in range(SMOKE_MESSAGES)
        ]
        await s.execute(
            text(
                "INSERT INTO messages (chat_id, tg_msg_id, sender_id, published_at, text, "
                'lang, "out", created_at) '
                "VALUES (:c, :t, :s, :p, :x, 'uz', false, :p)"
            ),
            rows,
        )
        ids = [
            int(r[0])
            for r in (
                await s.execute(
                    text("SELECT id FROM messages WHERE chat_id = :c ORDER BY id"), {"c": cid}
                )
            ).all()
        ]
        await s.execute(
            text(
                "INSERT INTO message_embeddings (message_id, model, vector) "
                "VALUES (:m, :mo, CAST(:v AS vector))"
            ),
            [{"m": mid, "mo": model, "v": _unit(mid)} for mid in ids[::2]],
        )
        await s.commit()
        await s.execute(text("ANALYZE messages"))
        await s.commit()
    return {"account_id": int(aid), "chat_id": int(cid), "user_id": int(uid)}


class _FakeLLM:
    async def embed(self, texts: list[str], **_kw: Any) -> Any:
        from types import SimpleNamespace

        from app.llm import embed_model_name

        vecs = [[float(x) for x in _unit(7)[1:-1].split(",")] for _ in texts]
        return SimpleNamespace(
            vectors=vecs,
            model=embed_model_name(),
            provider="fake",
            usage=SimpleNamespace(tokens_in=0, tokens_out=0),
        )


async def _timed(label: str, coro: Any) -> Any:
    t0 = time.perf_counter()
    result = await coro
    elapsed = time.perf_counter() - t0
    assert elapsed < MAX_SECONDS, f"{label}: {elapsed:.2f}s > {MAX_SECONDS}s"
    return result


async def test_search_paths_stay_under_budget(sessionmaker_: Any, corpus: dict[str, int]) -> None:
    """Har o'lchanadigan chaqiruv 2 s dan tez (2k xabarlik korpusda)."""
    from app.services import search as S
    from app.services.digests import day_fingerprint
    from app.services.episodes import build_episodes

    llm = _FakeLLM()
    chat_id, account_id = corpus["chat_id"], corpus["account_id"]
    async with sessionmaker_() as s:
        await _timed("hybrid_search", S.hybrid_search(s, chat_id, "to'lov muddati", llm=llm))
        await _timed(
            "hybrid_search_multi",
            S.hybrid_search_multi(s, account_id, "to'lov muddati", llm=llm),
        )
        await _timed("entity_search", S.entity_search(s, account_id, ["@user1", "#deploy"]))
        await _timed("node_search", S.node_search(s, account_id, "to'lov", llm=llm))
        await _timed(
            "day_fingerprint",
            day_fingerprint(
                s, chat_id, datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            ),
        )
        report = await _timed("build_episodes", build_episodes(s, chat_id, full=True))
        assert report.created > 0
        await s.rollback()


async def test_trigram_channel_uses_the_gin_index(
    sessionmaker_: Any, corpus: dict[str, int]
) -> None:
    """Trigram kanali `ix_messages_chat_trgm` dan o'qiydi (indekssiz — O(chat))."""
    from sqlalchemy.dialects import postgresql

    from app.services import search as S

    col = S.Message.text_fold
    stmt = S._trgm_stmt(
        col,
        fn=S.func.strict_word_similarity,
        op=S.STRICT_TRGM_OP,
        threshold=S.STRICT_TRGM_THRESHOLD,
        terms=["tolov", "mudat"],
        conds=[S.Message.chat_id == corpus["chat_id"], S.Message.deleted_at.is_(None)],
        limit=40,
    )
    # `paramstyle="named"` — sukut `pyformat` `%` ni ikkilantiradi va `<<%`
    # operatori `<<%%` bo'lib buziladi.
    sql = str(
        stmt.compile(
            dialect=postgresql.dialect(paramstyle="named"),
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "<<%" in sql, "indekslanadigan operator yo'qolgan — kanal yana O(chat) bo'ladi"
    # Nomzodlar qopqog'i ham shart: usiz ball chatdagi **har** qatorda hisoblanadi.
    assert f"LIMIT {S.TRGM_CANDIDATE_CAP}" in sql

    async with sessionmaker_() as s:
        # Planner qaysi yo'lni tanlashi jadval hajmiga bog'liq (2k qatorda
        # `chat_id` indeksi ham arzon) — shuning uchun test **tanlovni** emas,
        # indeksning aynan shu predikatga xizmat qilishini tekshiradi:
        # tanlanadigan (kamdan-kam uchraydigan) so'zda boshqa yo'l yo'q.
        # 2 ming qatorda ketma-ket skan har doim arzon — uni o'chirib,
        # "indeks yo'li bormi" degan savolni ajratamiz.
        await s.execute(text("SET LOCAL enable_seqscan = off"))
        rows = (
            await s.execute(
                # `chat_id` ataylab yo'q: u bo'lsa planner har doim
                # `ix_messages_chat_id` ni tanlaydi va bu test hech narsani
                # tekshirmagan bo'lardi. Savol — `<<% text_fold` predikatiga
                # **indeks yo'li** bormi (0015 dan keyin faqat bitta bor).
                text("EXPLAIN SELECT id FROM messages WHERE 'zzqwxyv' <<% text_fold"),
            )
        ).all()
        plan = "\n".join(str(r[0]) for r in rows)
        await s.rollback()
    assert TRGM_INDEX in plan, f"{TRGM_INDEX} plandan yo'qolgan:\n{plan}"
