"""Sintetik korpus seeder — SQL miqyosini o'lchash uchun (`make perf-seed`).

Nega kerak: 2 ming xabarlik test bazasida har so'rov 1 ms. Muammolar
(indekssiz trigram kanali, HNSW ustidagi filtr, GIN natijasini to'liq
saralash) faqat **yuz minglab** qatorda ko'rinadi. Bu skript shunday
korpusni bir necha daqiqada quradi.

Nima sintetik va nima real:

  * matn — uz/ru/en so'z ro'yxatlaridan (apostrof variantlari va kirill
    ataylab bor: `uz_fold` yo'li aynan shularda sinaladi);
  * embedding — **tasodifiy birlik vektorlar**. Real LLM chaqirilmaydi.
    Ma'no yo'q, lekin o'lcham (768), qatorlar soni va HNSW yuki real;
  * tuzilma — reply zanjirlari, forwardlar, 18 oylik vaqt oynasi kunlik
    ritm bilan, `lang`, entity'lar — hammasi ilova kutgan shaklda.

Xavfsizlik: hech qanday `.env` qiymati o'qilmaydi/chop etilmaydi; DB manzili
`--dsn` yoki `DATABASE_URL` dan olinadi va log'ga chiqmaydi.

Ishlatish::

    PYTHONPATH=src python -m evals.perf.seed --dsn postgresql://... --messages 300000

HNSW indekslari seed **oldidan** o'chiriladi va oxirida qayta quriladi:
qator-qator INSERT'da HNSW yozuvi ~1 ms turadi, ya'ni 180k vektor uchun
soatlab. Ommaviy qurish esa bir necha daqiqa.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import random
import struct
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg

# ── korpus shakli ────────────────────────────────────────────────────────────
N_CHATS = 60
N_SUPERGROUPS = 40
N_CHANNELS = 15
N_PRIVATE = 5
BIG_CHATS = 5  # 30k+ xabarli chatlar
BIG_CHAT_MESSAGES = 32_000
MONTHS = 18
EMBED_FRACTION = 0.60
ENTITY_FRACTION = 0.20
REPLY_FRACTION = 0.30
REPLY_LOOKBACK = 20
FORWARD_FRACTION = 0.05
COPY_BATCH = 5_000
#: Oldindan tayyorlanadigan vektor satrlari (har biri 768-d birlik vektor).
#: Har xabarga yangi vektor yasash 138M float formatlash degani — seed
#: byudjetidan chiqib ketadi; pool esa HNSW grafi uchun yetarlicha xilma-xil.
VECTOR_POOL = 20_000
VECTOR_DIM = 768

UZ_WORDS = [
    "to'lov",
    "muddat",
    "shartnoma",
    "yig'ilish",
    "loyiha",
    "hisobot",
    "buyurtma",
    "mijoz",
    "ta'minot",
    "narx",
    "chegirma",
    "yetkazib",
    "berish",
    "omborxona",
    "bo'lim",
    "rahbar",
    "xodim",
    "ish",
    "haqi",
    "kelishuv",
    "so'rov",
    "javob",
    "muammo",
    "yechim",
    "tekshiruv",
    "sifat",
    "nazorat",
    "reja",
    "byudjet",
    "o'zgarish",
    "qo'shimcha",
    "imkoniyat",
    "e'lon",
    "ta'lim",
    "kurs",
    "dastur",
    "server",
    "baza",
    "xatolik",
    "tuzatish",
    "yangilanish",
    "versiya",
    "sinov",
    "ishga",
    "tushirish",
    "hujjat",
    "imzo",
    "kelishildi",
    "roziman",
    "yaxshi",
    "bugun",
    "ertaga",
    "kecha",
    "hafta",
    "oy",
    "yil",
    "soat",
    "daqiqa",
]
RU_WORDS = [
    "оплата",
    "платеж",
    "договор",
    "встреча",
    "проект",
    "отчет",
    "заказ",
    "клиент",
    "поставка",
    "цена",
    "скидка",
    "доставка",
    "склад",
    "отдел",
    "руководитель",
    "сотрудник",
    "зарплата",
    "соглашение",
    "запрос",
    "ответ",
    "проблема",
    "решение",
    "проверка",
    "качество",
    "контроль",
    "план",
    "бюджет",
    "изменение",
    "дополнение",
    "возможность",
    "объявление",
    "обучение",
    "курс",
    "сервер",
    "база",
    "ошибка",
    "исправление",
    "обновление",
    "версия",
    "тест",
    "запуск",
    "документ",
    "подпись",
    "согласовано",
    "хорошо",
    "сегодня",
    "завтра",
    "вчера",
    "неделя",
    "месяц",
    "год",
    "час",
    "минута",
    "деньги",
]
EN_WORDS = [
    "payment",
    "deadline",
    "contract",
    "meeting",
    "project",
    "report",
    "order",
    "client",
    "supply",
    "price",
    "discount",
    "delivery",
    "warehouse",
    "team",
    "manager",
    "employee",
    "salary",
    "agreement",
    "request",
    "response",
    "problem",
    "solution",
    "review",
    "quality",
    "control",
    "plan",
    "budget",
    "change",
    "addition",
    "option",
    "announcement",
    "training",
    "course",
    "server",
    "database",
    "error",
    "fix",
    "update",
    "version",
    "test",
    "deploy",
    "document",
    "signature",
    "approved",
    "today",
    "tomorrow",
    "yesterday",
    "week",
    "month",
    "year",
    "hour",
    "minute",
    "release",
]
LANGS = (("uz", UZ_WORDS), ("ru", RU_WORDS), ("en", EN_WORDS))
LANG_WEIGHTS = (0.5, 0.35, 0.15)

USERNAMES = [f"@user{i}" for i in range(1, 60)]
HASHTAGS = ["#deploy", "#release", "#bug", "#todo", "#urgent", "#payment", "#meeting"]
DOMAINS = ["example.com", "docs.example.org", "github.com", "telegram.org", "news.uz"]
MEDIA_TYPES = [None, None, None, None, None, None, None, "photo", "video", "document"]

#: Kunlik ritm — soatlar bo'yicha og'irlik (tunda jim, ish vaqtida gavjum).
HOUR_WEIGHTS = [
    1,
    1,
    1,
    1,
    1,
    2,
    4,
    8,
    20,
    35,
    45,
    48,
    40,
    44,
    46,
    42,
    35,
    28,
    22,
    18,
    14,
    10,
    6,
    3,
]


@dataclass(slots=True)
class ChatSpec:
    chat_id: int
    type: str
    title: str
    n_messages: int
    senders: list[int]


def _chat_specs(rng: random.Random, total_messages: int) -> list[ChatSpec]:
    """60 chat: 5 tasi 30k+, qolgani uzun dum (Zipf-ga yaqin)."""
    types = (
        [("supergroup", i) for i in range(N_SUPERGROUPS)]
        + [("channel", i) for i in range(N_CHANNELS)]
        + [("private", i) for i in range(N_PRIVATE)]
    )
    assert len(types) == N_CHATS
    counts = [0] * N_CHATS
    # Katta chatlar — birinchi 4 supergroup + 1 kanal. Kichik korpusda
    # (`test_perf_smoke`) ular ham proporsional qisqaradi, aks holda 2k
    # xabar so'ralganda 170k qator yozilardi.
    big = min(BIG_CHAT_MESSAGES, max(20, total_messages // 8))
    big_idx = [0, 1, 2, 3, N_SUPERGROUPS]
    for i in big_idx:
        counts[i] = big + rng.randint(0, max(1, big // 10))
    rest = max(N_CHATS, total_messages - sum(counts))
    tail = [i for i in range(N_CHATS) if i not in big_idx]
    weights = [1.0 / (k + 1) ** 0.85 for k in range(len(tail))]
    wsum = sum(weights)
    for k, i in enumerate(tail):
        counts[i] = max(1, int(rest * weights[k] / wsum))
    specs: list[ChatSpec] = []
    for i, (ctype, _) in enumerate(types):
        n_senders = 1 if ctype == "private" else (3 if ctype == "channel" else rng.randint(5, 40))
        specs.append(
            ChatSpec(
                chat_id=i + 1,
                type=ctype,
                title=f"{ctype.title()} {i + 1}",
                n_messages=counts[i],
                senders=[1_000_000 + i * 100 + k for k in range(n_senders)],
            )
        )
    return specs


def _make_text(rng: random.Random) -> tuple[str, str]:
    """(matn, lang) — aralash tilli, ba'zan entity bilan."""
    lang, words = rng.choices(LANGS, weights=LANG_WEIGHTS, k=1)[0]
    n = rng.randint(3, 25)
    parts = [rng.choice(words) for _ in range(n)]
    roll = rng.random()
    if roll < 0.12:
        parts.insert(rng.randrange(len(parts)), rng.choice(USERNAMES))
    if roll < 0.08:
        parts.append(rng.choice(HASHTAGS))
    if roll < 0.05:
        parts.append("https://" + rng.choice(DOMAINS) + "/a/b")
    return " ".join(parts), lang


def _make_vector_pool(rng: random.Random, n: int) -> list[bytes]:
    """`n` ta 768-d tasodifiy birlik vektor — pgvector binar wire shaklida."""
    pack = struct.Struct(f">hh{VECTOR_DIM}f").pack
    out: list[bytes] = []
    gauss = rng.gauss
    for _ in range(n):
        v = [gauss(0.0, 1.0) for _ in range(VECTOR_DIM)]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append(pack(VECTOR_DIM, 0, *[x / norm for x in v]))
    return out


def vector_literal(vec: bytes) -> str:
    """Binar vektorni pgvector matn literaliga (`'[a,b,…]'`) qaytaradi."""
    dim = struct.unpack_from(">h", vec, 0)[0]
    vals = struct.unpack_from(f">{dim}f", vec, 4)
    return "[" + ",".join(f"{x:.6f}" for x in vals) + "]"


def _timestamps(rng: random.Random, n: int, start: datetime, end: datetime) -> list[datetime]:
    """Kunlik ritmli, o'sish tartibidagi vaqt belgilari."""
    span_days = max(1, (end - start).days)
    days = sorted(rng.random() ** 0.9 * span_days for _ in range(n))
    hours = rng.choices(range(24), weights=HOUR_WEIGHTS, k=n)
    out: list[datetime] = []
    prev = start
    for d, h in zip(days, hours, strict=True):
        ts = start + timedelta(
            days=int(d), hours=h, minutes=rng.randrange(60), seconds=rng.randrange(60)
        )
        if ts < prev:
            ts = prev + timedelta(seconds=rng.randrange(1, 90))
        out.append(ts)
        prev = ts
    return out


# ─── DB ──────────────────────────────────────────────────────────────────────

HNSW_INDEXES = (
    ("ix_embeddings_hnsw", "message_embeddings", "vector vector_cosine_ops"),
    ("ix_node_emb_hnsw", "retrieval_node_embeddings", "vector halfvec_cosine_ops"),
)


async def _drop_hnsw(conn: asyncpg.Connection) -> None:
    for name, _table, _ops in HNSW_INDEXES:
        await conn.execute(f"DROP INDEX IF EXISTS {name}")


async def _create_hnsw(conn: asyncpg.Connection) -> None:
    for name, table, ops in HNSW_INDEXES:
        await conn.execute(
            f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
            f"USING hnsw ({ops}) WITH (m = 16, ef_construction = 64)"
        )


async def _register_vector_codec(conn: asyncpg.Connection) -> None:
    """`vector` uchun **binar** kodek — COPY boshqasini qabul qilmaydi.

    asyncpg `copy_records_to_table` binar protokolda yuradi va pgvector turi
    uchun enkoder bilmaydi ("no binary format encoder for type vector").
    pgvector wire shakli oddiy: `int16 dim`, `int16 unused`, keyin `float4`
    massiv (tarmoq bayt tartibi). Shu baytlarni to'g'ridan-to'g'ri o'tkazamiz
    — matn shakliga nisbatan ~1.8x kichik va float formatlash umuman yo'q.
    """
    await conn.set_type_codec(
        "vector",
        schema="public",
        encoder=lambda v: v,
        decoder=lambda v: v,
        format="binary",
    )


async def _wipe(conn: asyncpg.Connection) -> None:
    await conn.execute(
        "TRUNCATE users, accounts, chats, messages, message_embeddings, "
        "retrieval_nodes, retrieval_node_embeddings, node_entities, chat_digests, "
        "alert_rules, alert_events RESTART IDENTITY CASCADE"
    )


async def _seed_owner(conn: asyncpg.Connection) -> tuple[int, int]:
    user_id = await conn.fetchval(
        "INSERT INTO users (tg_user_id, username, locale) VALUES ($1, $2, 'uz') RETURNING id",
        777_001,
        "perfowner",
    )
    account_id = await conn.fetchval(
        "INSERT INTO accounts (user_id, tg_account_id, label, status) "
        "VALUES ($1, $2, 'perf', 'active') RETURNING id",
        user_id,
        777_001,
    )
    return int(user_id), int(account_id)


async def _seed_chats(
    conn: asyncpg.Connection, account_id: int, specs: list[ChatSpec], rng: random.Random
) -> None:
    rows = []
    for i, spec in enumerate(specs):
        rows.append(
            (
                account_id,
                5_000_000 + spec.chat_id,
                rng.getrandbits(40),
                spec.type,
                spec.title,
                f"chat{spec.chat_id}",
                False,
                False,
                rng.randint(5, 50_000),
                "read_only",
                "done",
                1 if i % 2 == 0 else None,  # yarmida tarix boshiga yetilgan
                spec.n_messages,
                spec.n_messages,
                f"https://t.me/c/{5_000_000 + spec.chat_id}/",
            )
        )
    await conn.copy_records_to_table(
        "chats",
        records=rows,
        columns=[
            "account_id",
            "tg_peer_id",
            "access_hash",
            "type",
            "title",
            "username",
            "is_writable",
            "is_admin",
            "participants_count",
            "write_mode",
            "sync_state",
            "synced_min_id",
            "synced_max_id",
            "synced_total",
            "link_prefix",
        ],
    )


MSG_COLUMNS = [
    "id",
    "chat_id",
    "tg_msg_id",
    "sender_id",
    "published_at",
    "text",
    "media_type",
    "reply_to_msg_id",
    "fwd_from_peer_id",
    "fwd_from_msg_id",
    "fwd_date",
    "views",
    "forwards",
    "reactions_total",
    "lang",
    "lang_conf",
    "out",
    "created_at",
]


async def seed_messages(
    conn: asyncpg.Connection,
    specs: list[ChatSpec],
    *,
    rng: random.Random,
    vectors: list[bytes],
    model: str,
    account_id: int,
    entity_fraction: float = ENTITY_FRACTION,
) -> dict[str, int]:
    """Xabarlar + embeddinglar + entity'larni COPY bilan yozadi."""
    from app.services.entities import extract_entities

    end = datetime.now(UTC).replace(microsecond=0)
    start = end - timedelta(days=MONTHS * 30)
    next_id = 1
    stats = {"messages": 0, "embeddings": 0, "entities": 0}

    for spec in specs:
        stamps = _timestamps(rng, spec.n_messages, start, end)
        msg_rows: list[tuple[Any, ...]] = []
        emb_rows: list[tuple[Any, ...]] = []
        ent_rows: list[tuple[Any, ...]] = []
        is_channel = spec.type == "channel"
        for k in range(spec.n_messages):
            mid = next_id + k
            tg_id = k + 1
            text, lang = _make_text(rng)
            reply_to = None
            if k > 0 and rng.random() < REPLY_FRACTION:
                reply_to = max(1, tg_id - rng.randint(1, min(REPLY_LOOKBACK, k)))
            fwd_peer = fwd_msg = fwd_at = None
            if rng.random() < FORWARD_FRACTION:
                fwd_peer = 5_000_000 + rng.randint(1, N_CHATS)
                fwd_msg = rng.randint(1, 5_000)
                fwd_at = stamps[k] - timedelta(days=rng.randint(1, 60))
            sender = None if is_channel and rng.random() < 0.8 else rng.choice(spec.senders)
            msg_rows.append(
                (
                    mid,
                    spec.chat_id,
                    tg_id,
                    sender,
                    stamps[k],
                    text,
                    rng.choice(MEDIA_TYPES),
                    reply_to,
                    fwd_peer,
                    fwd_msg,
                    fwd_at,
                    rng.randint(10, 50_000) if is_channel else None,
                    rng.randint(0, 200) if is_channel else None,
                    rng.randint(0, 500),
                    lang,
                    round(rng.uniform(0.6, 0.99), 3),
                    False,
                    stamps[k],
                )
            )
            if rng.random() < EMBED_FRACTION:
                emb_rows.append((mid, model, rng.choice(vectors), stamps[k]))
            if rng.random() < entity_fraction:
                for ent in extract_entities(text):
                    ent_rows.append(
                        (
                            account_id,
                            spec.chat_id,
                            mid,
                            None,
                            ent.kind,
                            ent.value,
                            ent.value_raw,
                            stamps[k],
                        )
                    )
            if len(msg_rows) >= COPY_BATCH:
                await _flush(conn, msg_rows, emb_rows, ent_rows, stats)
                msg_rows, emb_rows, ent_rows = [], [], []
        await _flush(conn, msg_rows, emb_rows, ent_rows, stats)
        next_id += spec.n_messages
    await conn.execute("SELECT setval('messages_id_seq', $1)", next_id)
    return stats


async def _flush(
    conn: asyncpg.Connection,
    msg_rows: list[tuple[Any, ...]],
    emb_rows: list[tuple[Any, ...]],
    ent_rows: list[tuple[Any, ...]],
    stats: dict[str, int],
) -> None:
    if msg_rows:
        await conn.copy_records_to_table("messages", records=msg_rows, columns=MSG_COLUMNS)
        stats["messages"] += len(msg_rows)
    if emb_rows:
        await conn.copy_records_to_table(
            "message_embeddings",
            records=emb_rows,
            columns=["message_id", "model", "vector", "created_at"],
        )
        stats["embeddings"] += len(emb_rows)
    if ent_rows:
        # `uq_node_entity_value` — COPY konfliktni ko'tara olmaydi, shuning uchun
        # takrorlar shu yerda tashlanadi (bitta xabarda bir xil entity ikki marta
        # chiqmaydi, lekin ehtiyot shart kalit bo'yicha yig'amiz).
        seen: set[tuple[int, str, str]] = set()
        uniq = []
        for r in ent_rows:
            key = (int(r[2]), str(r[4]), str(r[5]))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        await conn.copy_records_to_table(
            "node_entities",
            records=uniq,
            columns=[
                "account_id",
                "chat_id",
                "message_id",
                "node_id",
                "kind",
                "value",
                "value_raw",
                "seen_at",
            ],
        )
        stats["entities"] += len(uniq)


def _pg_dsn(url: str) -> str:
    """SQLAlchemy URL → asyncpg DSN (sir chop etilmaydi)."""
    return url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgres+asyncpg://", "postgresql://"
    )


async def run(dsn: str, *, total: int, seed: int, wipe: bool) -> None:
    from app.llm import embed_model_name

    rng = random.Random(seed)
    model = embed_model_name()
    t0 = time.perf_counter()
    conn = await asyncpg.connect(_pg_dsn(dsn))
    try:
        await _register_vector_codec(conn)
        if wipe:
            await _wipe(conn)
        await _drop_hnsw(conn)
        _user_id, account_id = await _seed_owner(conn)
        specs = _chat_specs(rng, total)
        await _seed_chats(conn, account_id, specs, rng)
        print(f"chats: {len(specs)} (max {max(s.n_messages for s in specs)} msgs)")

        t_vec = time.perf_counter()
        vectors = _make_vector_pool(rng, VECTOR_POOL)
        print(f"vector pool: {len(vectors)} × {VECTOR_DIM}d in {time.perf_counter() - t_vec:.1f}s")

        t_msg = time.perf_counter()
        stats = await seed_messages(
            conn, specs, rng=rng, vectors=vectors, model=model, account_id=account_id
        )
        print(
            f"messages: {stats['messages']} embeddings: {stats['embeddings']} "
            f"entities: {stats['entities']} in {time.perf_counter() - t_msg:.1f}s"
        )

        t_idx = time.perf_counter()
        await _create_hnsw(conn)
        print(f"hnsw rebuild: {time.perf_counter() - t_idx:.1f}s")
        t_an = time.perf_counter()
        await conn.execute("VACUUM ANALYZE")
        print(f"vacuum analyze: {time.perf_counter() - t_an:.1f}s")
    finally:
        await conn.close()
    print(f"total seed time: {time.perf_counter() - t0:.1f}s (account_id={1})")


def main() -> None:
    ap = argparse.ArgumentParser(description="perf korpusini seed qiladi")
    ap.add_argument("--dsn", default=os.environ.get("PERF_DATABASE_URL", ""))
    ap.add_argument("--messages", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--no-wipe", action="store_true")
    args = ap.parse_args()
    dsn = args.dsn or os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise SystemExit("--dsn yoki PERF_DATABASE_URL/DATABASE_URL kerak")
    asyncio.run(run(dsn, total=args.messages, seed=args.seed, wipe=not args.no_wipe))


if __name__ == "__main__":
    main()
