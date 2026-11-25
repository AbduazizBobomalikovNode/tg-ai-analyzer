"""Nomzodlarni qayta tartiblash — listwise LLM (`Task.ROUTE`).

Nega kerak: RRF ballari **o'lchov birligi yo'q** ballarni birlashtiradi, ya'ni
u "qaysi xabar savolga javob beradi" degan savolga emas, "qaysi xabar ko'p
kanalda chiqdi" degan savolga javob beradi. Kalit so'zni eslatgan xabar bilan
javobni o'zida saqlagan xabar bir xil ballga tushishi mumkin. Rerank shu
farqni ajratadi va byudjetga birinchi bo'lib **javobi bor** xabarlar kiradi.

Dizayn qarorlari:

* **Dedup rerank'dan oldin majburiy** (chaqiruvchi mas'ul) — aks holda model
  o'nta bir xil forwardni tartiblab, byudjetni bitta matn bilan to'ldiradi.
* Model nomzod **matnini qaytarmaydi** — faqat indekslarni. Chiqish shu bilan
  ~40 token bo'lib qoladi va prompt injection natijani boshqara olmaydi:
  eng yomoni tartib buziladi, matn esa koddan keladi.
* Hamma nomzod **bitta** `<untrusted_data kind="candidates">` konvertida
  (invariant 3), har biri `neutralize()` dan o'tadi.
* Har qanday nosozlik — model xatosi, buzuq JSON, o'ylab topilgan indeks —
  **kirish tartibi** qaytadi. Rerank sifat yaxshilaydi, u ishlamasa qidiruv
  ishlamay qolmaydi.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from app.config import get_settings
from app.llm import LLM, LLMError, Msg, Task
from app.logging import get_logger
from app.services import untrusted as U
from app.services.prompts_retrieval import RERANK_PROMPT

if TYPE_CHECKING:
    from app.services.search import Hit

log = get_logger(__name__)

MAX_CANDIDATES = 60  # narx qopqog'i: 60 x 300 belgi ≈ 5k token kirish
CANDIDATE_CHARS = 300
PROVIDER_LLM = "llm"
PROVIDER_NONE = "none"

_JSON_RE = re.compile(r"\{.*\}", re.S)


def candidate_line(hit: Hit, *, max_chars: int = CANDIDATE_CHARS) -> str:
    """`chat | sana | jo'natuvchi` konteksti + qisqartirilgan matn."""
    from app.services.search import compact_text

    when = hit.date.strftime("%Y-%m-%d") if hit.date else "?"
    head = " | ".join(x for x in (hit.chat_title or "", when, hit.sender or "") if x)
    body = compact_text(hit.text or "", max_chars=max_chars)
    return f"[{head}] {body}".strip()


def build_envelope(query: str, candidates: list[Hit]) -> str:
    """Bitta konvert, nomzodlar faqat indeks bilan tanaladi."""
    lines = [f"{i}. {U.neutralize(candidate_line(h))}" for i, h in enumerate(candidates)]
    body = "\n".join(lines)
    return (
        f"<question>{U.neutralize(query)}</question>\n"
        f'<untrusted_data kind="candidates" count="{len(candidates)}">\n'
        f"{body}\n</untrusted_data>"
    )


def parse_ranking(raw: str, *, n: int) -> list[int]:
    """`{"ranking": [...]}` → tekshirilgan indekslar. Buzuq bo'lsa bo'sh ro'yxat."""
    return parse_ranking_opt(raw, n=n) or []


def parse_ranking_opt(raw: str, *, n: int) -> list[int] | None:
    """`parse_ranking`, lekin **buzuq** javob (`None`) bilan "hech biri mos emas"
    (`[]`) farqlanadi — bo'sh-natija strategiyasi aynan shu farqqa tayanadi."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return None
    try:
        data: Any = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    raw_list = data.get("ranking")
    if not isinstance(raw_list, list):
        return None
    out: list[int] = []
    for item in raw_list:
        if isinstance(item, bool) or not isinstance(item, int | str):
            continue
        try:
            idx = int(item)
        except (ValueError, TypeError):
            continue
        if 0 <= idx < n and idx not in out:
            out.append(idx)
    return out


async def rerank(
    query: str,
    candidates: list[Hit],
    *,
    llm: LLM | None = None,
    top_n: int | None = None,
    provider: str | None = None,
) -> list[Hit]:
    """Nomzodlarni savolga mosligi bo'yicha tartiblaydi (eng ko'pi `top_n`)."""
    ranked, _relevant = await rerank_scored(
        query, candidates, llm=llm, top_n=top_n, provider=provider
    )
    return ranked


async def rerank_scored(
    query: str,
    candidates: list[Hit],
    *,
    llm: LLM | None = None,
    top_n: int | None = None,
    provider: str | None = None,
) -> tuple[list[Hit], int | None]:
    """`rerank` + model **mos** deb sanagan nomzodlar soni.

    `None` — baho yo'q (rerank o'chiq, bitta nomzod, model xatosi, buzuq JSON):
    chaqiruvchi hamma nomzodni mos deb hisoblaydi. `0` — model javob berdi va
    birortasini ham mos demadi (`{"ranking": []}`): tartib baribir kirish
    tartibi, lekin bo'sh-natija strategiyasi "hech narsa topilmadi" deya oladi.
    """
    s = get_settings()
    provider = (provider or s.rerank_provider or PROVIDER_NONE).lower()
    top_n = top_n or s.rerank_top_n
    if not candidates:
        return [], 0
    pool = candidates[:MAX_CANDIDATES]
    if provider != PROVIDER_LLM or len(pool) < 2:
        return pool[:top_n], None

    msgs = [Msg.system(RERANK_PROMPT), Msg.user(build_envelope(query, pool))]
    try:
        res = await (llm or LLM()).chat(Task.ROUTE, msgs, json_mode=True, max_tokens=300)
    except LLMError as exc:
        log.warning("rerank.failed", error=str(exc)[:200])
        return pool[:top_n], None

    order = parse_ranking_opt(res.text, n=len(pool))
    if not order:
        log.info("rerank.no_ranking", candidates=len(pool), valid=order is not None)
        return pool[:top_n], (0 if order is not None else None)

    ranked = [pool[i] for i in order]
    # Model tashlab ketganlari yo'qolmaydi — ular kirish tartibida orqaga tushadi.
    seen = set(order)
    ranked.extend(h for i, h in enumerate(pool) if i not in seen)
    log.debug("rerank.done", candidates=len(pool), ranked=len(order))
    return ranked[:top_n], min(len(order), top_n)


__all__ = [
    "CANDIDATE_CHARS",
    "MAX_CANDIDATES",
    "PROVIDER_LLM",
    "PROVIDER_NONE",
    "build_envelope",
    "candidate_line",
    "parse_ranking",
    "parse_ranking_opt",
    "rerank",
    "rerank_scored",
]
