"""Relevantlik filtrining chegarasini **o'lchov bilan** tanlash (scale search, (d) bosqich).

`scale_search.FILTER_KEEP_MIN = 2` — taxmin emas, balki shu skript bilan
o'lchanadigan qaror. Skript ikki rejimda ishlaydi va ular orasida **odam**
turadi:

1. ``sample`` — beshta savol uchun korpusdan nomzod tortadi (bucket rejasi +
   `pull_candidates`, ya'ni jonli filtr ko'radigan aynan o'sha hovuz) va
   belgilash uchun CSV yozadi. LLM **chaqirilmaydi**: embedding uchun
   `--fake-embed` bilan determinstik vektor ishlatish mumkin, aks holda
   haqiqiy embed (bu yagona LLM chaqirig'i, `Task.EMBED`).

2. ``recall`` — odam to'ldirgan CSV'ni o'qiydi va **har chegara uchun**
   recall/precision beradi: "chegarani 3 ga ko'tarsak nechta haqiqiy javob
   yo'qoladi". `FILTER_KEEP_MIN` ni o'zgartirishdan oldin shu jadval kerak.

CSV ustunlari::

    q_id,question,chat_id,tg_msg_id,rank,fused_score,model_score,label,text

`label` — **odam** qo'yadigan yagona ustun: `1` = savolga javob beradi yoki
javobga olib boradi, `0` = bermaydi. `model_score` ``sample`` da bo'sh
qoladi; uni ``score`` rejimi (ixtiyoriy, LLM chaqiradi) to'ldiradi.

Xavfsizlik: `.env` o'qilmaydi, DB manzili `--dsn`/`DATABASE_URL` dan keladi
va hech qayerga chop etilmaydi. CSV'da xabar matni bo'ladi — u **lokal
fayl**, hech qayerga yuborilmaydi; `evals/runs/` kabi commit qilinmaydi.

Ishlatish::

    PYTHONPATH=src python -m evals.calibrate_filter sample \\
        --dsn postgresql+asyncpg://... --account 1 --n 300 --out cal.csv
    # cal.csv ni qo'lda belgilang (label ustuni)
    PYTHONPATH=src python -m evals.calibrate_filter recall --csv cal.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

#: Kalibrovka savollari — ataylab har xil turdagi (aniq fakt, davr, odam,
#: ko'p tilli, keng mavzu). Bitta turdagi beshta savol chegarani yolg'on
#: ishonch bilan tanlaydi.
QUESTIONS: list[str] = [
    "to'lov muddati qachon kelishilgan edi",
    "какие сроки запуска обсуждали",
    "deploy release qachon bo'lgan",
    "Nurbek shartnoma haqida nima dedi",
    "server xatoliklari haqida nimalar yozilgan",
]

FIELDS = (
    "q_id",
    "question",
    "chat_id",
    "tg_msg_id",
    "rank",
    "fused_score",
    "model_score",
    "label",
    "text",
)
TEXT_CHARS = 200
EMBED_DIM = 768


# ─── LLM'siz embedding (sample rejimi uchun) ─────────────────────────────────


class FakeEmbed:
    """Determinstik birlik vektor — `evals/perf/bench.FakeLLM` bilan bir mantiq."""

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = dim

    async def embed(self, texts: list[str], **_kw: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            vectors=[_unit(t, self.dim) for t in texts],
            model="fake-embed",
            provider="fake",
            usage=SimpleNamespace(tokens_in=0, tokens_out=0),
        )

    async def chat(self, *_a: Any, **_k: Any) -> Any:  # pragma: no cover
        raise AssertionError("calibrate_filter sample rejimida chat chaqirmaydi")


def _unit(seed_text: str, dim: int) -> list[float]:
    rng = random.Random(hash(seed_text) & 0xFFFF_FFFF)
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ─── sample ──────────────────────────────────────────────────────────────────


async def sample_rows(
    sm: async_sessionmaker[AsyncSession],
    *,
    account_id: int,
    n: int,
    fake_embed: bool,
    buckets: int,
) -> list[dict[str, Any]]:
    """Har savol uchun jonli hovuzdan `n` ta nomzod (rank tartibida)."""
    from sqlalchemy import select

    from app.db.models import Chat
    from app.llm import LLM
    from app.services import deep_search as DS
    from app.services import scale_search as SC

    llm: Any = FakeEmbed() if fake_embed else LLM()
    meter = DS._Meter(llm)

    def factory() -> Any:
        return sm()

    async with sm() as session:
        titles = {
            int(cid): title
            for cid, title in (
                await session.execute(
                    select(Chat.id, Chat.title).where(Chat.account_id == account_id)
                )
            ).all()
        }

    out: list[dict[str, Any]] = []
    for q_id, question in enumerate(QUESTIONS, 1):
        plan = DS.DeepPlan(subqueries=[question], standalone_question=question)
        vectors = await SC.query_vectors(meter, question=question, subqueries=plan.subqueries)
        async with sm() as session:
            planned = await SC.plan_buckets(
                session, account_id, plan=plan, vectors=vectors, chat_titles=titles, limit=buckets
            )
        spec = SC.build_pull_spec(
            account_id,
            question=question,
            subqueries=plan.subqueries,
            qvec=vectors[0] if vectors else None,
        )
        pool, _bucket_of, scanned = await SC.pull_candidates(
            factory, buckets=planned, spec=spec, budget_s=120.0
        )
        async with sm() as session:
            pool = await DS._hydrate(session, pool[:n], account_id=account_id)
        pool = SC.dedup_fast(pool)
        print(f"  q{q_id}: {len(planned)} bucket ({scanned} skanlandi) → {len(pool)} nomzod")
        for rank, hit in enumerate(pool[:n]):
            out.append(
                {
                    "q_id": q_id,
                    "question": question,
                    "chat_id": hit.chat_id,
                    "tg_msg_id": hit.tg_msg_id,
                    "rank": rank,
                    "fused_score": round(hit.score, 6),
                    "model_score": "",
                    "label": "",
                    "text": (hit.text or "").replace("\n", " ")[:TEXT_CHARS],
                }
            )
    return out


async def score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ixtiyoriy: `model_score` ni jonli filtr bilan to'ldiradi (LLM chaqiradi)."""
    from app.llm import LLM, Msg, Task
    from app.services import deep_search as DS
    from app.services import scale_search as SC
    from app.services import search as S
    from app.services.prompts_deep import SCALE_FILTER_PROMPT

    meter = DS._Meter(LLM())
    by_q: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_q[int(row["q_id"])].append(row)

    for q_id, group in sorted(by_q.items()):
        question = group[0]["question"]
        hits = [
            S.Hit(
                message_id=i,
                tg_msg_id=int(r["tg_msg_id"]),
                score=float(r["fused_score"] or 0),
                source="cal",
                chat_id=int(r["chat_id"]),
                text=str(r["text"]),
            )
            for i, r in enumerate(group)
        ]
        for batch in SC.batches(hits):
            prompt = SC.build_filter_envelope(question, batch)
            res = await meter.chat(
                Task.ROUTE,
                [Msg.system(SCALE_FILTER_PROMPT), Msg.user(prompt)],
                json_mode=True,
                max_tokens=SC.FILTER_MAX_TOKENS,
            )
            scores = SC.parse_keep(res.text, allowed=[i for i, _ in batch])
            for i, _hit in batch:
                group[i]["model_score"] = scores.get(i, 0)
        print(f"  q{q_id}: baholandi ({len(group)} nomzod)")
    print(f"jami xarajat: ~${meter.cost_usd:.4f}")
    return rows


# ─── recall ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ThresholdStat:
    threshold: int
    kept: int
    true_kept: int
    recall: float
    precision: float

    def line(self) -> str:
        return (
            f"  >= {self.threshold}: saqlangan {self.kept:5d} · "
            f"topilgan {self.true_kept:5d} · recall {self.recall:6.1%} · "
            f"precision {self.precision:6.1%}"
        )


def recall_table(rows: list[dict[str, Any]]) -> list[ThresholdStat]:
    """Har chegara uchun recall/precision — belgilangan CSV ustidan."""
    labelled = [r for r in rows if str(r.get("label", "")).strip() in ("0", "1")]
    scored = [r for r in labelled if str(r.get("model_score", "")).strip() != ""]
    if not scored:
        raise SystemExit("`model_score` bo'sh — avval `score` rejimini yugurting")
    positives = sum(1 for r in scored if str(r["label"]).strip() == "1")
    out: list[ThresholdStat] = []
    for threshold in (1, 2, 3):
        kept = [r for r in scored if int(r["model_score"]) >= threshold]
        true_kept = sum(1 for r in kept if str(r["label"]).strip() == "1")
        out.append(
            ThresholdStat(
                threshold=threshold,
                kept=len(kept),
                true_kept=true_kept,
                recall=(true_kept / positives) if positives else 0.0,
                precision=(true_kept / len(kept)) if kept else 0.0,
            )
        )
    return out


# ─── CSV ─────────────────────────────────────────────────────────────────────


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ─── kirish nuqtasi ──────────────────────────────────────────────────────────


async def _sample(args: argparse.Namespace) -> None:
    engine = create_async_engine(args.dsn, poolclass=NullPool)
    sm = async_sessionmaker(engine, expire_on_commit=False)
    try:
        rows = await sample_rows(
            sm,
            account_id=args.account,
            n=args.n,
            fake_embed=args.fake_embed,
            buckets=args.buckets,
        )
    finally:
        await engine.dispose()
    write_csv(args.out, rows)
    print(f"\n{len(rows)} qator → {args.out}")
    print("Endi `label` ustunini to'ldiring: 1 = kerakli, 0 = kerak emas.")


async def _score(args: argparse.Namespace) -> None:
    rows = await score_rows(read_csv(args.csv))
    write_csv(args.csv, rows)
    print(f"{args.csv} yangilandi (`model_score`).")


def _recall(args: argparse.Namespace) -> None:
    stats = recall_table(read_csv(args.csv))
    print("\nchegara bo'yicha (FILTER_KEEP_MIN nomzodlari):")
    for stat in stats:
        print(stat.line())
    print(
        "\nTanlash mezoni: recall yuqori bo'lgan eng **qattiq** chegara. "
        "Recall 5% dan ko'p tushsa chegarani ko'tarmang — javob yo'qoladi, "
        "token esa filtr bosqichida emas, sintezda tejalmaydi."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="scale search filtr chegarasi kalibrovkasi")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="korpusdan nomzod tortadi va belgilash CSV'sini yozadi")
    s.add_argument("--dsn", default=os.environ.get("DATABASE_URL", ""))
    s.add_argument("--account", type=int, required=True)
    s.add_argument("--n", type=int, default=300, help="har savol uchun nomzod soni")
    s.add_argument("--buckets", type=int, default=200)
    s.add_argument("--out", default="calibrate_filter.csv")
    s.add_argument("--fake-embed", action="store_true", help="LLM'siz (determinstik vektor)")

    c = sub.add_parser("score", help="`model_score` ni jonli filtr bilan to'ldiradi (LLM)")
    c.add_argument("--csv", required=True)

    r = sub.add_parser("recall", help="belgilangan CSV → chegara bo'yicha recall/precision")
    r.add_argument("--csv", required=True)

    args = ap.parse_args()
    if args.cmd == "sample":
        if not args.dsn:
            raise SystemExit("--dsn yoki DATABASE_URL kerak")
        asyncio.run(_sample(args))
    elif args.cmd == "score":
        asyncio.run(_score(args))
    else:
        _recall(args)


if __name__ == "__main__":
    main()
