"""Golden set ustida retrieval/deep-search sifatini o'lchaydi.

CLI::

    python -m evals.run_eval --golden evals/golden.jsonl \
        --config baseline expand expand+rerank deep \
        --runs 3 --out evals/runs/<ts>.json

``--config`` bir yoki bir nechta qiymat qabul qiladi (``baseline``,
``expand``, ``expand+rerank``, ``deep``); berilmasa hammasi yuguriladi.

Har konfiguratsiya uchun, har savol uchun:

  baseline        `search.hybrid_search(expand=False)`
  expand          `search.hybrid_search(expand=True)` (`services/query_expand`)
  expand+rerank   yuqoridagi + `services.rerank.rerank` (listwise LLM)
  deep            `services.deep_search.run_deep_search` — **agar mavjud
                  bo'lsa** (boshqa agent yozadi); modul yo'q bo'lsa
                  konfiguratsiya izoh bilan o'tkazib yuboriladi, xato emas.

Metrikalar (`evals/metrics.py`): Recall@20, nDCG@10, MRR@10 — retrieval
konfiguratsiyalari uchun; `deep` uchun sitata aniqligi (`cited ∩ gold`) va
qamrov da'vosi aqli (kodda tekshiriladi, LLM-judge emas). Kesimlar: `lang_q`,
`script_src`, `chat_id`.

``CLAUDE.md`` "Narx qopqoqlari" qoidasi: kamida 3 yugurish, median + tarqalish;
ikki konfiguratsiya median oraliqlari kesishsa — "inconclusive".
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import session_scope
from app.db.models import Chat
from app.llm import EMBED_DOCUMENT, LLM, ChatResult, EmbedResult, EmbedTaskType, Msg, Task, ToolSpec
from app.llm.pricing import estimate_cost
from app.logging import get_logger
from app.services import rerank as RR
from app.services import search as S
from app.services.textnorm import uz_fold
from evals.build_golden import read_jsonl
from evals.metrics import intervals_overlap, median_spread, mrr_at_k, ndcg_at_k, recall_at_k

log = get_logger(__name__)

CONFIGS: tuple[str, ...] = ("baseline", "expand", "expand+rerank", "deep")
_RETRIEVAL_FLAGS: dict[str, tuple[bool, bool]] = {
    # config -> (expand, rerank)
    "baseline": (False, False),
    "expand": (True, False),
    "expand+rerank": (True, True),
}
METRICS_RETRIEVAL: tuple[str, ...] = ("recall@20", "ndcg@10", "mrr@10")
METRICS_DEEP: tuple[str, ...] = ("citation_precision", "coverage_sane")
RETRIEVAL_LIMIT = 50

# Digest v2 (9-bo'lim) da qo'llanadigan qoida deep search javobiga ham
# taalluqli: "hamma xabarlar korildi" da'vosi hech qachon chiqmasligi kerak.
# Apostrof shakllari uz_fold() bilan bir xil fazoga tushiriladi (services/textnorm) —
# bu yerda ambiguous unicode belgilarni saqlamaslik uchun tekshiruv fold qilingan matnda.
_FORBIDDEN_COVERAGE_CLAIM = re.compile(
    r"hamma xabarlar korildi"
    r"|barcha xabarlar (korib chiqildi|toliq tahlil qilindi)"
    r"|all messages (were )?(seen|reviewed|scanned)",
    re.IGNORECASE,
)
_CITED_ID_RE = re.compile(r"#(\d+)")


# ─── deep_search — ixtiyoriy import ───────────────────────────────────────────

DeepSearchFn = Callable[..., Awaitable[Any]]


def _import_run_deep_search() -> DeepSearchFn | None:
    """``services.deep_search`` hali yozilmagan bo'lishi mumkin — u holda ``None``.

    Alohida funksiya sifatida: testda ``evals.run_eval._import_run_deep_search``
    monkeypatch qilinadi, real importga tegilmaydi.
    """
    try:
        from app.services.deep_search import run_deep_search
    except ImportError:
        return None
    return run_deep_search


def _extract_answer_text(result: Any) -> str:
    for attr in ("answer", "text"):
        val = getattr(result, attr, None)
        if isinstance(val, str):
            return val
    return str(result)


def _extract_cited_ids(answer_text: str) -> set[int]:
    return {int(m) for m in _CITED_ID_RE.findall(answer_text)}


def coverage_claim_is_sane(answer_text: str) -> bool:
    """``True`` — javobda taqiqlangan "hamma xabar ko'rildi" da'vosi yo'q.

    Tekshiruv ``uz_fold`` qilingan matnda — apostrof/registr farqi natijaga
    ta'sir qilmaydi (``services/textnorm``, invariant bilan bir xil fazo).
    """
    return not _FORBIDDEN_COVERAGE_CLAIM.search(uz_fold(answer_text or ""))


# ─── token/narx kuzatuvi ───────────────────────────────────────────────────────


@dataclass(slots=True)
class UsageTracker:
    records: list[tuple[str, str, int, int]] = field(default_factory=list)

    def add(self, provider: str, model: str, tokens_in: int, tokens_out: int) -> None:
        self.records.append((provider, model, tokens_in, tokens_out))

    def totals(self) -> dict[str, Any]:
        tokens_in = sum(r[2] for r in self.records)
        tokens_out = sum(r[3] for r in self.records)
        cost = 0.0
        priced = 0
        for provider, model, ti, to in self.records:
            c = estimate_cost(provider, model, ti, to)
            if c is not None:
                cost += c
                priced += 1
        return {
            "calls": len(self.records),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost, 4),
            "priced_calls": priced,
        }


class TrackingLLM(LLM):
    """`LLM` fasadi — har chaqiruvni `UsageTracker` ga yozadi, boshqa hech narsa."""

    def __init__(self, tracker: UsageTracker) -> None:
        self._tracker = tracker

    async def chat(
        self,
        task: Task,
        messages: list[Msg],
        *,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        res = await super().chat(
            task,
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        self._tracker.add(res.provider, res.model, res.usage.tokens_in, res.usage.tokens_out)
        return res

    async def embed(
        self, texts: list[str], *, dim: int | None = None, task_type: EmbedTaskType = EMBED_DOCUMENT
    ) -> EmbedResult:
        res = await super().embed(texts, dim=dim, task_type=task_type)
        self._tracker.add(res.provider, res.model, res.usage.tokens_in, res.usage.tokens_out)
        return res


# ─── bitta config, bitta yugurish ──────────────────────────────────────────────


@dataclass(slots=True)
class ConfigRunOutput:
    config: str
    skipped: bool
    note: str
    rows: list[dict[str, Any]]


async def _load_chats(session: AsyncSession, chat_ids: set[int]) -> dict[int, Chat]:
    if not chat_ids:
        return {}
    rows = (await session.execute(select(Chat).where(Chat.id.in_(chat_ids)))).scalars().all()
    return {c.id: c for c in rows}


async def score_retrieval_query(
    session: AsyncSession,
    item: dict[str, Any],
    *,
    expand: bool,
    rerank_on: bool,
    llm: LLM,
    chats: dict[int, Chat],
) -> dict[str, Any]:
    chat_id = int(item["chat_id"])
    hits = await S.hybrid_search(
        session, chat_id, item["question"], limit=RETRIEVAL_LIMIT, expand=expand, llm=llm
    )
    if rerank_on and len(hits) > 1:
        chat = chats.get(chat_id)
        if chat is not None:
            hits = await S.hydrate_hits(session, chat, hits)
        hits = await RR.rerank(
            item["question"], hits, llm=llm, provider=RR.PROVIDER_LLM, top_n=RETRIEVAL_LIMIT
        )
    retrieved = [h.tg_msg_id for h in hits]
    gold = item["gold_msg_ids"]
    return {
        "recall@20": recall_at_k(retrieved, gold, 20),
        "ndcg@10": ndcg_at_k(retrieved, gold, 10),
        "mrr@10": mrr_at_k(retrieved, gold, 10),
        "lang_q": item.get("lang_q", "other"),
        "script_src": item.get("script_src", "other"),
        "chat_id": chat_id,
    }


async def score_deep_query(
    session: AsyncSession, item: dict[str, Any], *, account_id: int, run_deep_search: DeepSearchFn
) -> dict[str, Any] | None:
    try:
        result = await run_deep_search(session, account_id=account_id, question=item["question"])
    except Exception as exc:  # B'ning imzosi hali barqaror emas — o'tkazib yuboramiz
        log.warning("run_eval.deep_search_failed", error=str(exc)[:200])
        return None
    answer_text = _extract_answer_text(result)
    cited = _extract_cited_ids(answer_text)
    gold = set(int(x) for x in item["gold_msg_ids"])
    precision = (len(cited & gold) / len(cited)) if cited else 0.0
    return {
        "citation_precision": precision,
        "coverage_sane": 1.0 if coverage_claim_is_sane(answer_text) else 0.0,
        "lang_q": item.get("lang_q", "other"),
        "script_src": item.get("script_src", "other"),
        "chat_id": int(item["chat_id"]),
    }


async def run_config(
    session: AsyncSession,
    golden: Sequence[dict[str, Any]],
    config: str,
    *,
    llm: LLM,
    account_id: int | None,
    chats: dict[int, Chat],
) -> ConfigRunOutput:
    if config == "deep":
        run_deep_search = _import_run_deep_search()
        if run_deep_search is None:
            return ConfigRunOutput(
                config=config,
                skipped=True,
                note="services.deep_search hali yo'q (boshqa agent yozmoqda) — o'tkazib yuborildi",
                rows=[],
            )
        if account_id is None:
            return ConfigRunOutput(
                config=config,
                skipped=True,
                note="--account-id berilmagan — deep_search akkaunt darajasida ishlaydi",
                rows=[],
            )
        rows = []
        for item in golden:
            row = await score_deep_query(
                session, item, account_id=account_id, run_deep_search=run_deep_search
            )
            if row is not None:
                rows.append(row)
        return ConfigRunOutput(config=config, skipped=False, note="", rows=rows)

    if config not in _RETRIEVAL_FLAGS:
        return ConfigRunOutput(
            config=config, skipped=True, note=f"noma'lum konfiguratsiya: {config!r}", rows=[]
        )

    expand, rerank_on = _RETRIEVAL_FLAGS[config]
    rows = [
        await score_retrieval_query(
            session, item, expand=expand, rerank_on=rerank_on, llm=llm, chats=chats
        )
        for item in golden
    ]
    return ConfigRunOutput(config=config, skipped=False, note="", rows=rows)


# ─── agregatsiya (sof funksiyalar — DB/LLM yo'q) ──────────────────────────────


def aggregate_run(rows: Sequence[dict[str, Any]], metric_names: Sequence[str]) -> dict[str, float]:
    """Bir yugurish ichidagi savol-satrlari -> har metrik uchun o'rtacha."""
    out: dict[str, float] = {}
    for name in metric_names:
        vals = [r[name] for r in rows if name in r]
        out[name] = (sum(vals) / len(vals)) if vals else 0.0
    return out


def slice_rows(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get(key, "other"))].append(r)
    return dict(groups)


def combine_runs(
    run_aggregates: Sequence[dict[str, float]],
) -> dict[str, tuple[float, float, float]]:
    """Har metrik uchun ``(median, min, max)`` — ``>=3`` yugurish bo'yicha."""
    if not run_aggregates:
        return {}
    metric_names = run_aggregates[0].keys()
    return {name: median_spread([ra[name] for ra in run_aggregates]) for name in metric_names}


def compare_configs(
    a: dict[str, tuple[float, float, float]], b: dict[str, tuple[float, float, float]]
) -> dict[str, str]:
    """Har metrik uchun ``"a_better" | "b_better" | "inconclusive"``.

    Oraliqlar (``min``, ``max``) kesishsa — bitta yugurish to'plami boshqasidan
    **aniq** ustun emas, natija "inconclusive".
    """
    out: dict[str, str] = {}
    for name in a:
        if name not in b:
            continue
        a_med, a_lo, a_hi = a[name]
        b_med, b_lo, b_hi = b[name]
        if intervals_overlap((a_lo, a_hi), (b_lo, b_hi)):
            out[name] = "inconclusive"
        else:
            out[name] = "a_better" if a_med > b_med else "b_better"
    return out


# ─── to'liq quvur ──────────────────────────────────────────────────────────────


async def run_all(
    session: AsyncSession,
    golden: Sequence[dict[str, Any]],
    configs: Sequence[str],
    *,
    runs: int,
    account_id: int | None,
) -> dict[str, Any]:
    chat_ids = {int(item["chat_id"]) for item in golden}
    chats = await _load_chats(session, chat_ids)

    report: dict[str, Any] = {"golden_count": len(golden), "runs": runs, "configs": {}}
    for config in configs:
        tracker = UsageTracker()
        llm = TrackingLLM(tracker)
        metric_names = METRICS_DEEP if config == "deep" else METRICS_RETRIEVAL

        per_run: list[dict[str, float]] = []
        by_slice: dict[str, dict[str, list[dict[str, float]]]] = {
            "lang_q": defaultdict(list),
            "script_src": defaultdict(list),
            "chat_id": defaultdict(list),
        }
        skipped, note = False, ""
        for _ in range(runs):
            out = await run_config(
                session, golden, config, llm=llm, account_id=account_id, chats=chats
            )
            if out.skipped:
                skipped, note = True, out.note
                break
            per_run.append(aggregate_run(out.rows, metric_names))
            for slice_key, groups_out in by_slice.items():
                for slice_val, rows in slice_rows(out.rows, slice_key).items():
                    groups_out[slice_val].append(aggregate_run(rows, metric_names))

        if skipped:
            report["configs"][config] = {"skipped": True, "note": note}
            continue

        report["configs"][config] = {
            "skipped": False,
            "overall": combine_runs(per_run),
            "slices": {
                slice_key: {
                    slice_val: combine_runs(runs_) for slice_val, runs_ in groups_out.items()
                }
                for slice_key, groups_out in by_slice.items()
            },
            "usage": tracker.totals(),
        }

    report["comparisons"] = _build_comparisons(report["configs"])
    return report


def _build_comparisons(configs: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Ketma-ket konfiguratsiyalarni taqqoslaydi: baseline vs expand, expand vs
    expand+rerank — deep alohida (boshqa metrikalar), taqqoslanmaydi."""
    order = [
        c
        for c in ("baseline", "expand", "expand+rerank")
        if c in configs and not configs[c].get("skipped")
    ]
    out: dict[str, dict[str, str]] = {}
    for a_name, b_name in pairwise(order):
        out[f"{a_name}_vs_{b_name}"] = compare_configs(
            configs[a_name]["overall"], configs[b_name]["overall"]
        )
    return out


# ─── konsol jadvali ────────────────────────────────────────────────────────────


def render_table(report: dict[str, Any]) -> str:
    lines: list[str] = [f"golden={report['golden_count']} runs={report['runs']}", ""]
    for config, data in report["configs"].items():
        if data.get("skipped"):
            lines.append(f"[{config}] SKIPPED — {data.get('note', '')}")
            continue
        lines.append(f"[{config}]")
        for metric, (median, lo, hi) in data["overall"].items():
            lines.append(f"  {metric:<20} median={median:.4f}  spread=[{lo:.4f}, {hi:.4f}]")
        usage = data.get("usage", {})
        if usage:
            lines.append(
                f"  usage: calls={usage['calls']} tokens_in={usage['tokens_in']} "
                f"tokens_out={usage['tokens_out']} cost_usd~{usage['cost_usd']}"
            )
    if report.get("comparisons"):
        lines.append("")
        lines.append("comparisons:")
        for pair, metrics in report["comparisons"].items():
            for metric, verdict in metrics.items():
                lines.append(f"  {pair}  {metric:<16} {verdict}")
    return "\n".join(lines)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m evals.run_eval",
        description="Golden set ustida retrieval/deep-search sifatini o'lchaydi.",
    )
    p.add_argument("--golden", type=Path, default=Path("evals/golden.jsonl"))
    p.add_argument("--config", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    p.add_argument("--runs", type=int, default=3, help="CLAUDE.md qoidasi: kamida 3.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--account-id", type=int, default=None, help="Faqat --config deep uchun kerak.")
    return p


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    golden = read_jsonl(args.golden)
    if not golden:
        raise SystemExit(f"golden set bo'sh yoki topilmadi: {args.golden}")
    async with session_scope() as session:
        return await run_all(
            session, golden, args.config, runs=args.runs, account_id=args.account_id
        )


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    report = asyncio.run(_run(args))

    out_path = args.out or Path(f"evals/runs/{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(render_table(report))
    print(f"\nfull report: {out_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "CONFIGS",
    "METRICS_DEEP",
    "METRICS_RETRIEVAL",
    "ConfigRunOutput",
    "TrackingLLM",
    "UsageTracker",
    "aggregate_run",
    "combine_runs",
    "compare_configs",
    "coverage_claim_is_sane",
    "main",
    "render_table",
    "run_all",
    "run_config",
    "score_deep_query",
    "score_retrieval_query",
    "slice_rows",
]
