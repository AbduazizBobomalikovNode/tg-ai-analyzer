"""Qidiruv yo'llarining SQL miqyosi benchmarki (`make perf-bench`).

`evals/perf/seed.py` qurgan korpus ustida **haqiqiy** servis funksiyalari
chaqiriladi (`search`, `episodes`, `nodes`, `alerts`, `digests`, `analytics`,
`cluster`). LLM chaqirilmaydi: embedding o'rniga determinstik tasodifiy
birlik vektor qaytaruvchi soxta fasad, kengaytirish uchun esa tayyor
`Expansion` obyekti beriladi.

Har o'lchov **kamida 5 marta** yuguradi va p50/p95 chiqaradi — LLM'siz ham
Postgres tebranadi (kesh isishi, autovacuum), bitta yugurish hech narsani
isbotlamaydi (CLAUDE.md, "Narx qopqoqlari" bandidagi qoida).

Chiqish: konsolga jadval + `--json <fayl>` bo'lsa mashina o'qiydigan nusxa.

Ishlatish::

    PYTHONPATH=src python -m evals.perf.bench --dsn postgresql+asyncpg://... \\
        --runs 5 --json bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

EMBED_DIM = 768
CLUSTER_N = 400
#: Epizod qurishning **inkremental** yo'lini o'lchash uchun qo'shiladigan
#: xabarlar soni (real sync bir siklda shu tartibda xabar keltiradi).
APPEND_MESSAGES = 500

QUESTIONS = [
    "to'lov muddati qachon kelishildi",
    "оплата договора сроки",
    "deploy release version",
    "yig'ilish qarori hisobot",
    "server xatolik tuzatish",
    "клиент скидка доставка",
]


@dataclass(slots=True)
class Timing:
    name: str
    samples: list[float] = field(default_factory=list)
    note: str = ""

    @property
    def p50(self) -> float:
        return statistics.median(self.samples) if self.samples else float("nan")

    @property
    def p95(self) -> float:
        if not self.samples:
            return float("nan")
        ordered = sorted(self.samples)
        idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return ordered[max(0, idx)]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runs": len(self.samples),
            "p50_ms": round(self.p50, 1),
            "p95_ms": round(self.p95, 1),
            "note": self.note,
        }


class FakeLLM:
    """Soxta LLM fasadi — embedding determinstik, `chat` umuman yo'q."""

    def __init__(self, seed: int = 7) -> None:
        self._rng = random.Random(seed)
        self.calls = 0

    async def embed(self, texts: Sequence[str], **_kw: Any) -> Any:
        self.calls += 1
        vectors = [unit_vector(t) for t in texts]
        return SimpleNamespace(
            vectors=vectors,
            model=_model_name(),
            provider="fake",
            usage=SimpleNamespace(tokens_in=0, tokens_out=0),
        )

    async def chat(self, *_a: Any, **_k: Any) -> Any:  # pragma: no cover — chaqirilmasin
        raise AssertionError("bench LLM chaqirmaydi")


def unit_vector(seed_text: str, dim: int = EMBED_DIM) -> list[float]:
    """Matndan determinstik tasodifiy birlik vektor (real embedding emas)."""
    rng = random.Random(hash(seed_text) & 0xFFFF_FFFF)
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _model_name() -> str:
    from app.llm import embed_model_name

    return embed_model_name()


def vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


# ─── o'lchov mexanikasi ──────────────────────────────────────────────────────


class RecordingSession:
    """`AsyncSession` ustidagi yupqa qobiq — bajarilgan statementlarni yozadi."""

    def __init__(self, inner: AsyncSession) -> None:
        self._inner = inner
        self.statements: list[Any] = []

    async def execute(self, stmt: Any, *a: Any, **kw: Any) -> Any:
        self.statements.append(stmt)
        return await self._inner.execute(stmt, *a, **kw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


async def measure(
    timings: list[Timing],
    name: str,
    factory: Callable[[], Awaitable[Any]],
    *,
    runs: int,
    note_from: Callable[[Any], str] | None = None,
) -> Any:
    t = Timing(name)
    result: Any = None
    for _ in range(runs):
        t0 = time.perf_counter()
        result = await factory()
        t.samples.append((time.perf_counter() - t0) * 1000.0)
    if note_from is not None:
        t.note = note_from(result)
    timings.append(t)
    print(f"  {name:<52} p50={t.p50:9.1f}ms  p95={t.p95:9.1f}ms  {t.note}")
    return result


# ─── korpus haqida ma'lumot ──────────────────────────────────────────────────


async def corpus_info(session: AsyncSession) -> dict[str, Any]:
    from app.db.models import Chat, Message

    total = (await session.execute(text("SELECT count(*) FROM messages"))).scalar_one()
    embs = (await session.execute(text("SELECT count(*) FROM message_embeddings"))).scalar_one()
    ents = (await session.execute(text("SELECT count(*) FROM node_entities"))).scalar_one()
    rows = (
        await session.execute(
            select(Chat.id, Chat.title, Chat.type, Chat.synced_total)
            .order_by(Chat.synced_total.desc())
            .limit(60)
        )
    ).all()
    account_id = (await session.execute(text("SELECT min(id) FROM accounts"))).scalar_one()
    msg_max = (await session.execute(text("SELECT max(id) FROM messages"))).scalar_one()
    return {
        "messages": int(total),
        "embeddings": int(embs),
        "entities": int(ents),
        "account_id": int(account_id),
        "max_message_id": int(msg_max or 0),
        "chats": [
            {"id": int(r[0]), "title": r[1], "type": r[2], "n": int(r[3] or 0)} for r in rows
        ],
        "_unused": Message.__tablename__,
    }


# ─── bo'limlar ───────────────────────────────────────────────────────────────


def _expansion() -> Any:
    """LLM'siz "kengaytirilgan" so'rov — real `query_expand` chiqishi shaklida."""
    from app.services.query_expand import Expansion

    return Expansion(
        uz_lat=["tolov", "muddat", "shartnoma"],
        uz_cyr=["тўлов"],
        ru=["оплата", "платеж", "договор"],
        en=["payment", "deadline", "contract"],
        synonyms=["hisob", "schet"],
    )


async def bench_hybrid(
    sm: async_sessionmaker[AsyncSession],
    timings: list[Timing],
    info: dict[str, Any],
    *,
    runs: int,
) -> None:
    from app.services import search as S

    llm: Any = FakeLLM()
    big = info["chats"][0]
    small = next(c for c in info["chats"] if c["n"] < 1_000)
    print(
        f"\n── hybrid_search (big={big['title']}:{big['n']}, small={small['title']}:{small['n']})"
    )
    exp = _expansion()
    for label, chat, expansion in (
        (f"hybrid_search big ({big['n']} msgs)", big, None),
        ("hybrid_search big + expansion", big, exp),
        (f"hybrid_search small ({small['n']} msgs)", small, None),
        ("hybrid_search small + expansion", small, exp),
    ):

        async def run(chat: dict[str, Any] = chat, expansion: Any = expansion) -> Any:
            async with sm() as s:
                return await S.hybrid_search(
                    s, chat["id"], QUESTIONS[0], limit=20, expansion=expansion, llm=llm
                )

        await measure(timings, label, run, runs=runs, note_from=lambda r: f"hits={len(r)}")


async def bench_multi(
    sm: async_sessionmaker[AsyncSession],
    timings: list[Timing],
    info: dict[str, Any],
    *,
    runs: int,
) -> None:
    from app.services import search as S

    llm: Any = FakeLLM()
    acc = info["account_id"]
    print("\n── hybrid_search_multi (akkaunt bo'ylab)")

    async def one() -> Any:
        async with sm() as s:
            return await S.hybrid_search_multi(s, acc, QUESTIONS[0], limit=40, llm=llm)

    await measure(
        timings,
        "hybrid_search_multi (1 so'rov)",
        one,
        runs=runs,
        note_from=lambda r: f"hits={len(r)}",
    )

    async def six() -> Any:
        out = []
        async with sm() as s:
            for q in QUESTIONS:
                out.append(await S.hybrid_search_multi(s, acc, q, limit=40, llm=llm))
        return out

    await measure(
        timings,
        "hybrid_search_multi (6 sub-query ketma-ket)",
        six,
        runs=runs,
        note_from=lambda r: f"hits={sum(len(x) for x in r)}",
    )

    ents = ["@user3", "#deploy", "example.com"]

    async def entity() -> Any:
        async with sm() as s:
            return await S.entity_search(s, acc, ents, limit=80)

    await measure(
        timings,
        "entity_search (3 entity)",
        entity,
        runs=runs,
        note_from=lambda r: f"hits={len(r)}",
    )


async def bench_episodes(
    sm: async_sessionmaker[AsyncSession],
    timings: list[Timing],
    info: dict[str, Any],
    *,
    runs: int,
) -> list[int]:
    """5 katta chatda `build_episodes` (to'liq) + 500 xabar qo'shgandan keyingi inkremental."""
    from app.services.episodes import build_episodes

    big_ids = [c["id"] for c in info["chats"][:5]]
    print(f"\n── build_episodes (5 katta chat: {big_ids})")

    async def full() -> Any:
        total = 0
        async with sm() as s:
            for cid in big_ids:
                rep = await build_episodes(s, cid, full=True)
                total += rep.created + rep.updated + rep.unchanged
            await s.commit()
        return total

    # To'liq qurish qimmat — ikki marta (ikkinchisi "hech narsa o'zgarmagan" yo'li).
    await measure(
        timings,
        "build_episodes full ×5 katta chat",
        full,
        runs=min(runs, 2),
        note_from=lambda r: f"nodes={r}",
    )

    async def incremental() -> Any:
        async with sm() as s:
            appended = await _append_messages(s, big_ids[0], APPEND_MESSAGES)
            await s.commit()
        t0 = time.perf_counter()
        async with sm() as s:
            rep = await build_episodes(s, big_ids[0], since_msg_id=appended)
            await s.commit()
        return (time.perf_counter() - t0) * 1000.0, rep

    t = Timing("build_episodes inkremental (+500 msg)")
    last: Any = None
    for _ in range(runs):
        ms, last = await incremental()
        t.samples.append(ms)
    t.note = f"created={last.created} updated={last.updated} read={last.messages}"
    timings.append(t)
    print(f"  {t.name:<52} p50={t.p50:9.1f}ms  p95={t.p95:9.1f}ms  {t.note}")
    return big_ids


async def _append_messages(session: AsyncSession, chat_id: int, n: int) -> int:
    """Chat oxiriga `n` ta yangi xabar qo'shadi; `from` uchun eng kichik tg_msg_id."""
    row = (
        await session.execute(
            text(
                "SELECT coalesce(max(tg_msg_id), 0), coalesce(max(published_at), now()) "
                "FROM messages WHERE chat_id = :c"
            ),
            {"c": chat_id},
        )
    ).one()
    start_tg = int(row[0]) + 1
    base = row[1]
    rng = random.Random(chat_id * 31 + start_tg)
    rows = []
    for k in range(n):
        rows.append(
            {
                "c": chat_id,
                "t": start_tg + k,
                "s": 1_000_000 + rng.randrange(20),
                "p": base + timedelta(minutes=3 * (k + 1)),
                "x": " ".join(
                    rng.choice(["to'lov", "оплата", "payment", "muddat", "deploy"])
                    for _ in range(rng.randint(4, 20))
                ),
            }
        )
    await session.execute(
        text(
            "INSERT INTO messages (chat_id, tg_msg_id, sender_id, published_at, text, lang, "
            "\"out\", created_at) VALUES (:c, :t, :s, :p, :x, 'uz', false, :p) "
            "ON CONFLICT (chat_id, tg_msg_id) DO NOTHING"
        ),
        rows,
    )
    return start_tg


async def seed_node_embeddings(session: AsyncSession, *, batch: int = 500) -> int:
    """Tugunlarga sintetik `halfvec` vektor beradi (vektor kanali bo'sh qolmasin)."""
    model = _model_name()
    total = 0
    while True:
        ids = [
            int(r[0])
            for r in (
                await session.execute(
                    text(
                        "SELECT n.id FROM retrieval_nodes n "
                        "LEFT JOIN retrieval_node_embeddings e "
                        "  ON e.node_id = n.id AND e.model = :m "
                        "WHERE e.node_id IS NULL LIMIT :k"
                    ),
                    {"m": model, "k": batch},
                )
            ).all()
        ]
        if not ids:
            break
        await session.execute(
            text(
                "INSERT INTO retrieval_node_embeddings (node_id, model, vector) "
                "VALUES (:n, :m, CAST(:v AS halfvec)) ON CONFLICT DO NOTHING"
            ),
            [{"n": i, "m": model, "v": vector_literal(unit_vector(f"node{i}"))} for i in ids],
        )
        await session.commit()
        total += len(ids)
    return total


async def bench_nodes(
    sm: async_sessionmaker[AsyncSession],
    timings: list[Timing],
    info: dict[str, Any],
    big_ids: list[int],
    *,
    runs: int,
) -> None:
    from app.services import search as S

    llm: Any = FakeLLM()
    acc = info["account_id"]
    async with sm() as s:
        n_nodes = (await s.execute(text("SELECT count(*) FROM retrieval_nodes"))).scalar_one()
        t0 = time.perf_counter()
        embedded = await seed_node_embeddings(s)
    print(
        f"\n── node_search (nodes={n_nodes}, embedded={embedded} in "
        f"{time.perf_counter() - t0:.1f}s)"
    )

    async def ns() -> Any:
        async with sm() as s:
            return await S.node_search(s, acc, QUESTIONS[0], limit=20, llm=llm)

    hits = await measure(
        timings,
        "node_search (akkaunt, level 1-3)",
        ns,
        runs=runs,
        note_from=lambda r: f"nodes={len(r)}",
    )

    async def ns_chat() -> Any:
        async with sm() as s:
            return await S.node_search(s, acc, QUESTIONS[0], chat_ids=big_ids, limit=20, llm=llm)

    await measure(
        timings,
        "node_search (5 katta chat)",
        ns_chat,
        runs=runs,
        note_from=lambda r: f"nodes={len(r)}",
    )

    async def n2m() -> Any:
        async with sm() as s:
            return await S.node_hits_to_messages(s, hits)

    await measure(
        timings,
        "node_hits_to_messages (12 tugun)",
        n2m,
        runs=runs,
        note_from=lambda r: f"msgs={len(r)}",
    )


async def bench_alerts(
    sm: async_sessionmaker[AsyncSession],
    timings: list[Timing],
    info: dict[str, Any],
    big_ids: list[int],
    *,
    runs: int,
) -> None:
    from app.db.models import AlertRule
    from app.services import alerts as A

    acc = info["account_id"]
    model = _model_name()
    print("\n── alerts.scan_rule")
    async with sm() as s:
        await s.execute(text("DELETE FROM alert_events"))
        await s.execute(text("DELETE FROM alert_rules"))
        await s.execute(
            text(
                "INSERT INTO alert_rules (id, account_id, chat_id, kind, label, pattern, "
                "threshold, enabled, cooldown_minutes, last_checked_msg_id) VALUES "
                "(1, :a, :c, 'keyword', 'kw-chat', 'to''lov, оплата, payment', 0.8, true, 0, 0),"
                "(2, :a, NULL, 'keyword', 'kw-acc', 'deploy, релиз', 0.8, true, 0, 0),"
                "(3, :a, :c, 'semantic', 'sem-chat', 'to''lov kechikishi', 0.2, true, 0, 0),"
                "(4, :a, NULL, 'semantic', 'sem-acc', 'server xatolik', 0.2, true, 0, 0)"
            ),
            {"a": acc, "c": big_ids[0]},
        )
        await s.execute(
            text("UPDATE alert_rules SET embedding = CAST(:v AS halfvec) WHERE kind = 'semantic'"),
            {"v": vector_literal(unit_vector("alert-rule"))},
        )
        await s.commit()
    _ = model

    for rid, label in (
        (1, "keyword, bitta chat"),
        (2, "keyword, akkaunt"),
        (3, "semantic, bitta chat"),
        (4, "semantic, akkaunt"),
    ):

        async def run(rid: int = rid) -> Any:
            async with sm() as s:
                rule = await s.get(AlertRule, rid)
                assert rule is not None
                rule.last_checked_msg_id = 0  # har yugurish bir xil ishni ko'rsin
                events = await A.scan_rule(s, rule)
                await s.rollback()
                return events

        await measure(
            timings,
            f"alerts.scan_rule ({label})",
            run,
            runs=runs,
            note_from=lambda r: f"events={len(r)}",
        )


async def bench_misc(
    sm: async_sessionmaker[AsyncSession],
    timings: list[Timing],
    info: dict[str, Any],
    big_ids: list[int],
    *,
    runs: int,
) -> None:
    from app.db.models import Chat
    from app.services import analytics as AN
    from app.services import cluster as CL
    from app.services import search as S
    from app.services.digests import day_fingerprint

    print("\n── digests / analytics / cluster")
    chat_id = big_ids[0]
    day = (datetime.now(UTC) - timedelta(days=30)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    async def fp() -> Any:
        async with sm() as s:
            return await day_fingerprint(s, chat_id, day)

    await measure(
        timings, "digests.day_fingerprint (1 kun)", fp, runs=runs, note_from=lambda r: f"n={r[1]}"
    )

    for days in (30, 540):

        async def stats(days: int = days) -> Any:
            async with sm() as s:
                chat = await s.get(Chat, chat_id)
                assert chat is not None
                return await AN.chat_stats(s, chat, days=days)

        await measure(
            timings,
            f"analytics.chat_stats ({days} kun)",
            stats,
            runs=runs,
            note_from=lambda r: f"posts={r.posts}",
        )

    # cluster: 400 hit + 768-d vektorlar (sof Python O(n²))
    async with sm() as s:
        rows = (
            await s.execute(
                text(
                    "SELECT m.id, m.tg_msg_id, m.text, m.published_at FROM messages m "
                    "WHERE m.chat_id = :c ORDER BY m.id LIMIT :k"
                ),
                {"c": chat_id, "k": CLUSTER_N},
            )
        ).all()
        ids = [int(r[0]) for r in rows]
        vectors = await S.fetch_vectors(s, ids)

    def make_hits() -> list[Any]:
        return [
            S.Hit(
                int(r[0]),
                int(r[1]),
                1.0 / (i + 1),
                "vector",
                chat_id=chat_id,
                text=r[2] or "",
                date=r[3],
            )
            for i, r in enumerate(rows)
        ]

    t = Timing(f"cluster.dedup ({CLUSTER_N} hit, 768-d)")
    for _ in range(runs):
        hits = make_hits()
        t0 = time.perf_counter()
        deduped = CL.dedup(hits, vectors)
        t.samples.append((time.perf_counter() - t0) * 1000.0)
    t.note = f"in={len(rows)} out={len(deduped)} vecs={len(vectors)}"
    timings.append(t)
    print(f"  {t.name:<52} p50={t.p50:9.1f}ms  p95={t.p95:9.1f}ms  {t.note}")

    t = Timing(f"cluster.cluster_topics ({CLUSTER_N} hit, 768-d)")
    for _ in range(runs):
        hits = make_hits()
        t0 = time.perf_counter()
        clusters = CL.cluster_topics(hits, vectors)
        t.samples.append((time.perf_counter() - t0) * 1000.0)
    t.note = f"clusters={len(clusters)}"
    timings.append(t)
    print(f"  {t.name:<52} p50={t.p50:9.1f}ms  p95={t.p95:9.1f}ms  {t.note}")


# ─── EXPLAIN + indeks statistikasi ───────────────────────────────────────────

EXPLAIN_HEADER = "EXPLAIN (ANALYZE, BUFFERS)"


def log_render_fallback(exc: BaseException) -> None:
    """`literal_binds` ishlamadi — ikkinchi yo'lga o'tamiz (sabab ko'rinib tursin)."""
    print(f"  (literal_binds: {str(exc)[:80]} → parametrlarni qo'lda qo'yamiz)")


def _render(stmt: Any, dialect: Any) -> str | None:
    """Statementni **bajariladigan** SQL matniga aylantiradi (EXPLAIN uchun).

    `literal_binds` ba'zi turlarda ishlamaydi (`regconfig`, `vector` —
    "No literal value renderer"), shuning uchun ikkinchi yo'l: nomli
    parametrlar bilan kompilyatsiya qilib, qiymatlarni o'zimiz qo'yamiz.
    Faqat o'lchov skripti uchun; ilova kodi bunday qilmaydi.
    """
    try:
        return str(stmt.compile(dialect=dialect, compile_kwargs={"literal_binds": True}))
    except Exception as exc:
        log_render_fallback(exc)
    try:
        compiled = stmt.compile(dialect=dialect)
        sql = str(compiled)
        for name, value in sorted(compiled.params.items(), key=lambda kv: -len(kv[0])):
            if isinstance(value, str):
                literal = "'" + value.replace("'", "''") + "'"
            elif isinstance(value, bool | int | float):
                literal = str(value)
            else:
                return None
            sql = sql.replace(f":{name}", literal)
        return sql
    except Exception as exc:
        print(f"  (EXPLAIN render failed: {exc})")
        return None


async def explain_channels(
    sm: async_sessionmaker[AsyncSession], info: dict[str, Any]
) -> dict[str, str]:
    """To'rt leksik/vektor kanal + tugun tsv so'rovi uchun `EXPLAIN (ANALYZE, BUFFERS)`.

    Statementlar `search.py` ning o'zidan **yozib olinadi** (`RecordingSession`),
    shuning uchun plan aynan ilova yuboradigan so'rovniki. Vektor kanali
    istisno: `vector` bind parametri `literal_binds` bilan render qilinmaydi,
    shuning uchun u qo'lda — `vector_search` bilan bir xil shaklda — yoziladi.
    """
    from sqlalchemy.dialects import postgresql

    from app.services import search as S

    big = info["chats"][0]
    out: dict[str, str] = {}
    llm: Any = FakeLLM()
    async with sm() as inner:
        rec = RecordingSession(inner)
        await S.hybrid_search(rec, big["id"], QUESTIONS[0], limit=20, llm=llm)  # type: ignore[arg-type]
        await S.node_search(rec, info["account_id"], QUESTIONS[0], limit=20, llm=llm)  # type: ignore[arg-type]
        labels = ["tsv_uz", "tsv_ru", "trigram", "vector", "node_tsv"]
        picked: dict[str, Any] = {}
        for stmt in rec.statements:
            sql = str(stmt)
            if "tsv_uz" in sql and "tsv_uz" not in picked:
                picked["tsv_uz"] = stmt
            elif "tsv_ru" in sql and "tsv_ru" not in picked:
                picked["tsv_ru"] = stmt
            elif "word_similarity" in sql and "trigram" not in picked:
                picked["trigram"] = stmt
            elif "retrieval_nodes.tsv" in sql and "node_tsv" not in picked:
                picked["node_tsv"] = stmt
        for label in labels:
            stmt = picked.get(label)
            if stmt is None:
                continue
            dialect: Any = postgresql.dialect(paramstyle="named")  # type: ignore[no-untyped-call]
            sql_text = _render(stmt, dialect)
            if sql_text is None:
                out[label] = "(render failed)"
                continue
            rows = (await inner.execute(text(f"{EXPLAIN_HEADER} {sql_text}"))).all()
            out[label] = "\n".join(str(r[0]) for r in rows)

        # Vektor kanali — `vector_search` bilan bir xil shakl, literal vektor bilan.
        vec_sql = (
            "SELECT messages.id, messages.tg_msg_id, "
            f"(message_embeddings.vector <=> '{vector_literal(unit_vector(QUESTIONS[0]))}') "
            "AS dist FROM messages JOIN message_embeddings "
            "ON message_embeddings.message_id = messages.id "
            f"WHERE messages.chat_id = {big['id']} AND messages.deleted_at IS NULL "
            f"AND message_embeddings.model = '{_model_name()}' ORDER BY dist LIMIT 40"
        )
        rows = (await inner.execute(text(f"{EXPLAIN_HEADER} {vec_sql}"))).all()
        out["vector"] = "\n".join(str(r[0]) for r in rows)
    return out


INDEX_STATS_SQL = """
SELECT indexrelname, idx_scan, idx_tup_read, idx_tup_fetch,
       pg_size_pretty(pg_relation_size(indexrelid)) AS size
  FROM pg_stat_user_indexes
 WHERE relname IN ('messages', 'message_embeddings', 'retrieval_nodes',
                   'retrieval_node_embeddings', 'node_entities')
 ORDER BY idx_scan DESC
"""


async def index_stats(sm: async_sessionmaker[AsyncSession]) -> list[dict[str, Any]]:
    async with sm() as s:
        rows = (await s.execute(text(INDEX_STATS_SQL))).all()
    return [
        {
            "index": r[0],
            "idx_scan": int(r[1] or 0),
            "tup_read": int(r[2] or 0),
            "tup_fetch": int(r[3] or 0),
            "size": r[4],
        }
        for r in rows
    ]


# ─── kirish nuqtasi ──────────────────────────────────────────────────────────


async def run(dsn: str, *, runs: int, json_path: str, sections: set[str]) -> None:
    engine = create_async_engine(dsn, poolclass=NullPool)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    timings: list[Timing] = []
    try:
        async with sm() as s:
            await s.execute(text("SELECT pg_stat_reset()"))
            info = await corpus_info(s)
        print(
            f"korpus: {info['messages']} xabar, {info['embeddings']} embedding, "
            f"{info['entities']} entity, {len(info['chats'])} chat"
        )
        big_ids = [c["id"] for c in info["chats"][:5]]
        if "hybrid" in sections:
            await bench_hybrid(sm, timings, info, runs=runs)
        if "multi" in sections:
            await bench_multi(sm, timings, info, runs=runs)
        if "episodes" in sections:
            big_ids = await bench_episodes(sm, timings, info, runs=runs)
        if "nodes" in sections:
            await bench_nodes(sm, timings, info, big_ids, runs=runs)
        if "alerts" in sections:
            await bench_alerts(sm, timings, info, big_ids, runs=runs)
        if "misc" in sections:
            await bench_misc(sm, timings, info, big_ids, runs=runs)

        plans = await explain_channels(sm, info) if "explain" in sections else {}
        stats = await index_stats(sm)
    finally:
        await engine.dispose()

    print("\n── EXPLAIN (ANALYZE, BUFFERS)")
    for label, plan in plans.items():
        print(f"\n[{label}]\n{plan}")
    print("\n── pg_stat_user_indexes (idx_scan)")
    for row in stats:
        print(
            f"  {row['index']:<40} scans={row['idx_scan']:<8} "
            f"read={row['tup_read']:<10} size={row['size']}"
        )

    payload = {
        "corpus": {k: v for k, v in info.items() if k != "chats"},
        "timings": [t.as_dict() for t in timings],
        "plans": plans,
        "indexes": stats,
    }
    if json_path:
        # Oxirgi qadam, o'lchov tugagan — bloklovchi yozuv shu yerda zararsiz
        # (ASYNC230), lekin `to_thread` bilan qoidani ham buzmaymiz.
        await asyncio.to_thread(
            Path(json_path).write_text,
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON: {json_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="perf benchmark")
    ap.add_argument("--dsn", default=os.environ.get("PERF_DATABASE_URL", ""))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--json", default="")
    ap.add_argument(
        "--sections",
        default="hybrid,multi,episodes,nodes,alerts,misc,explain",
        help="vergul bilan: hybrid,multi,episodes,nodes,alerts,misc,explain",
    )
    args = ap.parse_args()
    dsn = args.dsn or os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise SystemExit("--dsn yoki PERF_DATABASE_URL/DATABASE_URL kerak")
    asyncio.run(
        run(
            dsn,
            runs=max(5, args.runs),
            json_path=args.json,
            sections={x.strip() for x in args.sections.split(",") if x.strip()},
        )
    )


if __name__ == "__main__":
    main()
