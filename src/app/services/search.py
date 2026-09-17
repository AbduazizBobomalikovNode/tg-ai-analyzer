"""Qidiruv va token-tejamkor kontekst tanlash (3-bosqich, rejaning 7-bo'limi).

Nega: "oxirgi N xabarni promptga tashlash" katta chatlarda ham qimmat, ham
sifatsiz. Bu modul savolga **mos** xabarlarni topadi va byudjetga sig'diradi:

  fts_uz_search   `messages.tsv_uz` (fold qilingan matn, `simple` config, `:*` prefiks)
  fts_ru_search   `messages.tsv_ru` (`russian` config — stemming ishlaydi)
  trigram_search  `strict_word_similarity(term, text_fold)` — imlo xatolari
  vector_search   pgvector cosine (`RETRIEVAL_QUERY` embedding)
  hybrid_search   to'rttasi **og'irlikli** RRF bilan; ixtiyoriy ko'p tilli
                  so'rov kengaytirish (`services/query_expand`)
  select_context  strategiya: recent | search | window | auto → `ContextBundle`
                  (byudjet belgida, xabar matni qisqartirilib, URL/whitespace siqilib)
  compact_window  juda katta oynani map-reduce bilan siqadi (arzon `Task.ROUTE`
                  model bo'laklarni xulosalaydi → yakuniy javob xulosalar ustida)

Hech narsa promptga to'g'ridan-to'g'ri qo'shilmaydi — chat_service konvertlaydi.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import String, and_, func, literal, or_, select
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Chat, ChatType, Message, MessageEmbedding
from app.llm import EMBED_QUERY, LLM, LLMError, Msg, Task, embed_model_name
from app.logging import get_logger
from app.mtproto.pool import MessageInfo
from app.services import untrusted as U
from app.services.prompts import MAP_DIGEST_PROMPT
from app.services.query_expand import Expansion
from app.services.stopwords import STOP_ALL, is_stopword
from app.services.textnorm import uz_fold

log = get_logger(__name__)

# ── byudjet ──────────────────────────────────────────────────────────────────
CHARS_PER_TOKEN = 3.5  # o'zbek/rus/lotin aralash matn uchun ehtiyotkor baho
DEFAULT_CONTEXT_TOKENS = 6_000  # ~21k belgi — SEARCH vazifasi uchun
DEEP_CONTEXT_TOKENS = 14_000
MAX_MSG_CHARS = 700  # bitta xabar shundan uzun bo'lsa qisqartiriladi
NEIGHBORS = 1  # topilgan xabar atrofidan ±N (dialog konteksti)
RECENT_TAIL = 15  # search rejimida ham oxirgi N xabar (joriy mavzu)

# ── map-reduce ───────────────────────────────────────────────────────────────
MAP_CHUNK_CHARS = 14_000
MAP_MAX_CHUNKS = 12  # undan ko'p bo'lsa oynani qisqartiramiz (eng yangi bo'laklar)

# Stop-so'zlar `services/stopwords.py` da (uz + ru + en) — ilgari shu faylda
# qo'lda yozilgan ~90 so'zlik ro'yxat bor edi, endi yagona manba. Ro'yxat
# **fold fazosida** saqlanadi, shuning uchun tekshiruv `is_stopword` orqali.
_STOP = STOP_ALL

# ── RRF og'irliklari (arxitektura v2, 6-bo'lim) ──────────────────────────────
# Leksik kanallar teng; trigram shovqinliroq (imlo yaqinligi ≠ ma'no yaqinligi),
# vektor esa boshqa tildagi/boshqacha aytilgan javobni topadigan yagona kanal.
W_TSV_UZ = 1.0
W_TSV_RU = 1.0
W_TRIGRAM = 0.6
W_VECTOR = 1.2
DEFAULT_RRF_WEIGHTS: tuple[float, float, float, float] = (W_TSV_UZ, W_TSV_RU, W_TRIGRAM, W_VECTOR)

# ── trigram ostonalar ────────────────────────────────────────────────────────
# `strict_word_similarity` so'z chegarasida o'lchaydi — uzun xabar jazolanmaydi.
# Eski `similarity(text, term)` butun matn uzunligiga nisbatan o'lchagani uchun
# uzun xabarda hech qachon oshmasdi; u yo'l olib tashlandi.
STRICT_TRGM_THRESHOLD = 0.55
TRGM_THRESHOLD = 0.7  # `strict_word_similarity` yo'q/bo'sh bo'lsa — zaxira yo'l

# `strict_word_similarity(t, col) > 0.55` **funksiya chaqirig'i** — Postgres uni
# indekslay olmaydi va har qatorda hisoblaydi (107 belgili matnda ≈30 µs, ya'ni
# 35k xabarli chatda 4 ta so'z uchun ≈4.4 s; o'lchov: architecture-v2 §11.1).
# pg_trgm ayni shartning **operator** shaklini beradi va u GIN bilan bajariladi:
#   `t <<% col`  ≡  strict_word_similarity(t, col) > pg_trgm.strict_word_similarity_threshold (0.5)
#   `t <% col`   ≡  word_similarity(t, col)        > pg_trgm.word_similarity_threshold (0.6)
# Ikkala sukut ostona ham bizning ostonadan **past**, ya'ni operator — aniq
# shartning nadto'plami: u indeks orqali nomzodlarni beradi, aniq `>` esa
# ularni qirqadi. Natija bir xil, ish esa indeksda qoladi.
STRICT_TRGM_OP = "<<%"
TRGM_OP = "<%"
#: Ball hisoblanadigan nomzodlar qopqog'i. Operator bilan ham gapiradigan
#: chatda nomzod ko'p bo'lishi mumkin; ball esa har nomzodda yana bir marta
#: `strict_word_similarity` degani. Qopqoqsiz kanal chat hajmi bilan chiziqli
#: o'sadi. Nomzodlarning **hammasi** allaqachon 0.5 ostonadan yuqori, shuning
#: uchun kesilgan dum — "eng yomon" emas, "yana bir xil darajadagi" qatorlar.
TRGM_CANDIDATE_CAP = 2_000

#: HNSW indeksi `chat_id` filtri haqida hech narsa bilmaydi: sukut bo'yicha u
#: `ef_search` tagacha nomzod qaytaradi va ular **keyin** filtrlanadi. O'lchovda
#: (300k xabar, 35k'lik chat) bu `LIMIT 40` ga 33 qator qaytardi — kichikroq
#: chatda esa deyarli hech narsa qolmasdi. `iterative_scan` (pgvector ≥0.8)
#: indeksni yetarli qator topilguncha davom ettiradi.
HNSW_EF_SEARCH = 100
HNSW_ITERATIVE_SCAN = "relaxed_order"
_HNSW_TUNE_SQL = sa_text(
    f"SELECT set_config('hnsw.ef_search', '{HNSW_EF_SEARCH}', true), "
    f"set_config('hnsw.iterative_scan', '{HNSW_ITERATIVE_SCAN}', true)"
)

_URL_RE = re.compile(r"https?://\S+")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_TSQ_SAFE_RE = re.compile(r"[^\w]+", re.UNICODE)
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


@dataclass(slots=True)
class Hit:
    """Bitta topilma. Matn/metadata `hydrate_hits` bilan to'ldiriladi (rerank uchun)."""

    message_id: int
    tg_msg_id: int
    score: float
    source: str  # tsv_uz | tsv_ru | trigram | vector | fts | both
    # rerank/cluster uchun — SQL kanallari to'ldirmaydi, `hydrate_hits` to'ldiradi
    chat_id: int = 0
    chat_title: str = ""
    sender: str = ""
    date: datetime | None = None
    text: str = ""
    # dedup vakili ostiga yig'ilgan nusxalar: (chat_id, tg_msg_id, date)
    also_seen_in: list[tuple[int, int, datetime | None]] = field(default_factory=list)
    # ierarxiya darajasi: 0 — xom xabar (sukut), 1 epizod, 2 kun, 3 hafta.
    # Tugun kanalidan kelgan hit'da bu manba tugunning darajasi bo'ladi.
    level: int = 0


@dataclass(slots=True)
class ContextBundle:
    title: str
    messages: list[MessageInfo]
    strategy: str  # recent | search | window | window_compacted
    source: str = "db"
    est_tokens: int = 0
    truncated: bool = False
    hits: int = 0
    map_summaries: list[str] = field(default_factory=list)
    map_tokens_in: int = 0
    map_tokens_out: int = 0
    map_model: str = ""


# ─── matn siqish ─────────────────────────────────────────────────────────────


def compact_text(text_: str, *, max_chars: int = MAX_MSG_CHARS) -> str:
    """URL'larni qisqartiradi, ortiqcha bo'shliqni yig'adi, uzun matnni kesadi."""
    t = _URL_RE.sub(lambda m: _short_url(m.group(0)), text_)
    t = _WS_RE.sub(" ", t)
    t = _NL_RE.sub("\n\n", t).strip()
    if len(t) > max_chars:
        t = t[: max_chars - 1].rstrip() + "…"
    return t


def _short_url(url: str) -> str:
    m = re.match(r"https?://([^/\s]+)", url)
    return f"<{m.group(1)}>" if m else "<url>"


def est_tokens(text_: str) -> int:
    return int(len(text_) / CHARS_PER_TOKEN) + 1


def keywords(question: str, *, max_terms: int = 8) -> list[str]:
    """Savoldan qidiruv so'zlari (stop-so'zlarsiz, 3+ belgi, takrorsiz).

    Stop-so'z tekshiruvi **ikki shaklda**: yozilganidek va `uz_fold` dan keyin \u2014
    "bo'lib" / "bolib" / "\u0431\u045e\u043b\u0438\u0431" bitta stop-so'z.
    """
    words = re.findall(r"[\w'\u2019\u02bb-]{3,}", question.lower())
    out: list[str] = []
    for raw in words:
        w = raw.strip("'\u2019\u02bb-")
        if len(w) < 3 or w.isdigit() or w in out:
            continue
        if is_stopword(w):
            continue
        out.append(w)
    return out[:max_terms]


def fold_terms(terms: Sequence[str], *, max_terms: int = 12) -> list[str]:
    """Qidiruv variantlarini indeks bilan **bir xil** fold'ga keltiradi.

    `messages.text_fold` / `tsv_uz` generated ustunlari `uz_fold` ustida
    qurilgan \u2014 so'rov tomonida ayni funksiya qo'llanmasa, apostrof yoki kirill
    yozuvi tufayli hech narsa topilmaydi.
    """
    out: list[str] = []
    for t in terms:
        for part in uz_fold(t).split():
            if len(part) >= 2 and part not in out:
                out.append(part)
    return out[:max_terms]


def _tsq_safe(term: str) -> str:
    """`to_tsquery` uchun xavfsiz token \u2014 operator belgilari qolmaydi."""
    return _TSQ_SAFE_RE.sub(" ", term).split(" ")[0].strip() if term else ""


def _tsquery(terms: Sequence[str], *, prefix: bool, op: str) -> str:
    parts = [f"{t}:*" if prefix else t for t in (_tsq_safe(x) for x in terms) if t]
    return op.join(dict.fromkeys(parts))


def _embed_model(col: Any) -> list[Any]:
    """Joriy embedding modeli bo'yicha filtr (phase4/B integratsiyasi).

    `message_embeddings` / `retrieval_node_embeddings` bir vaqtning o'zida bir
    necha modelning vektorini saqlashi mumkin (re-embed hech narsa o'chirmaydi).
    Filtr bo'lmasa cosine masofa **turli fazolardagi** vektorlar orasida
    hisoblanardi — ya'ni model almashgan kuni reyting jimgina chalkashardi.
    """
    want = embed_model_name()
    return [col == want] if want else []


def _alive() -> list[Any]:
    """O'chirilgan xabarlar hech bir kanaldan chiqmaydi (`deleted_at` \u2014 0008)."""
    col = getattr(Message, "deleted_at", None)
    return [col.is_(None)] if col is not None else []


# ─── leksik kanallar (tsv_uz / tsv_ru / trigram) ─────────────────────────────


async def _run_tsv(
    session: AsyncSession,
    col: Any,
    *,
    config: str,
    tsq: str,
    chat_id: int,
    limit: int,
    source: str,
) -> list[Hit]:
    q = func.to_tsquery(config, tsq)
    score = func.ts_rank_cd(col, q).label("score")
    stmt = (
        select(Message.id, Message.tg_msg_id, score)
        .where(Message.chat_id == chat_id, *_alive(), col.op("@@")(q))
        .order_by(score.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [Hit(int(r[0]), int(r[1]), float(r[2] or 0), source) for r in rows]


async def fts_uz_search(
    session: AsyncSession, chat_id: int, terms: Sequence[str], *, limit: int = 20
) -> list[Hit]:
    """`tsv_uz` (simple config, fold qilingan matn) + prefiks mos.

    Avval AND (barcha so'z bor xabarlar \u2014 aniqroq), hech narsa topilmasa OR.
    """
    col = getattr(Message, "tsv_uz", None)
    if col is None or not terms:
        return []
    folded = fold_terms(terms)
    if not folded:
        return []
    and_q = _tsquery(folded, prefix=True, op=" & ")
    if not and_q:
        return []
    hits = await _run_tsv(
        session, col, config="simple", tsq=and_q, chat_id=chat_id, limit=limit, source="tsv_uz"
    )
    if hits or len(folded) < 2:
        return hits
    or_q = _tsquery(folded, prefix=True, op=" | ")
    return await _run_tsv(
        session, col, config="simple", tsq=or_q, chat_id=chat_id, limit=limit, source="tsv_uz"
    )


async def fts_ru_search(
    session: AsyncSession, chat_id: int, terms: Sequence[str], *, limit: int = 20
) -> list[Hit]:
    """`tsv_ru` (`russian` config \u2014 stemming ishlaydi, shuning uchun prefiks yo'q)."""
    col = getattr(Message, "tsv_ru", None)
    if col is None or not terms:
        return []
    clean = [t for t in (_tsq_safe(x.lower()) for x in terms) if len(t) >= 2]
    if not clean:
        return []
    and_q = _tsquery(clean, prefix=False, op=" & ")
    if not and_q:
        return []
    hits = await _run_tsv(
        session, col, config="russian", tsq=and_q, chat_id=chat_id, limit=limit, source="tsv_ru"
    )
    if hits or len(clean) < 2:
        return hits
    or_q = _tsquery(clean, prefix=False, op=" | ")
    return await _run_tsv(
        session, col, config="russian", tsq=or_q, chat_id=chat_id, limit=limit, source="tsv_ru"
    )


async def trigram_search(
    session: AsyncSession, chat_id: int, terms: Sequence[str], *, limit: int = 20
) -> list[Hit]:
    """Imlo xatosi/qo'shimcha farqi \u2014 so'z chegarasida o'lchangan trigram."""
    col = getattr(Message, "text_fold", None)
    if col is None or not terms:
        return []
    folded = fold_terms(terms)[:4]  # har term alohida indeks skani \u2014 qopqoq
    if not folded:
        return []
    hits = await _run_trgm(
        session,
        col,
        fn=func.strict_word_similarity,
        op=STRICT_TRGM_OP,
        threshold=STRICT_TRGM_THRESHOLD,
        terms=folded,
        conds=[Message.chat_id == chat_id, *_alive()],
        limit=limit,
    )
    if hits:
        return hits
    # Postgres/pg_trgm versiyasi `strict_word_similarity` bermasa yoki juda
    # qattiq kelsa \u2014 kengroq `word_similarity` (yuqoriroq ostona bilan).
    return await _run_trgm(
        session,
        col,
        fn=func.word_similarity,
        op=TRGM_OP,
        threshold=TRGM_THRESHOLD,
        terms=folded,
        conds=[Message.chat_id == chat_id, *_alive()],
        limit=limit,
    )


def _trgm_candidates(op: str, col: Any, terms: Sequence[str]) -> Any:
    """`t <<% col OR \u2026` \u2014 pg_trgm operatorlari (GIN bilan bajariladi)."""
    preds = [literal(t, String).bool_op(op)(col) for t in terms]
    return preds[0] if len(preds) == 1 else or_(*preds)


def _trgm_stmt(
    col: Any,
    *,
    fn: Any,
    op: str,
    threshold: float,
    terms: Sequence[str],
    conds: Sequence[Any],
    limit: int,
    with_chat: bool = False,
    join: Any = None,
) -> Any:
    """Ikki qavatli trigram so'rovi: indeksli nomzodlar \u2192 ball \u2192 tartib.

    Ichki qavat faqat **operator** bilan ishlaydi (GIN), `TRGM_CANDIDATE_CAP`
    bilan chegaralangan. Tashqi qavat aynan shu nomzodlarda
    `strict_word_similarity` ni hisoblaydi va aniq ostona bilan qirqadi.
    Ilgari ikkalasi bitta qavatda edi \u2014 funksiya chat/akkauntdagi **har**
    qatorda hisoblanardi.
    """
    cols = [Message.id.label("id"), Message.tg_msg_id.label("tg_msg_id")]
    if with_chat:
        cols.append(Message.chat_id.label("chat_id"))
    inner = select(*cols, col.label("fold")).where(*conds, _trgm_candidates(op, col, terms))
    if join is not None:
        inner = inner.join(join[0], join[1])
    sub = inner.limit(TRGM_CANDIDATE_CAP).subquery("trgm_cand")
    sims = [fn(t, sub.c.fold) for t in terms]
    expr = sims[0] if len(sims) == 1 else func.greatest(*sims)
    out = [sub.c.id, sub.c.tg_msg_id]
    if with_chat:
        out.append(sub.c.chat_id)
    return (
        select(*out, expr.label("score")).where(expr > threshold).order_by(expr.desc()).limit(limit)
    )


async def _run_trgm(
    session: AsyncSession,
    col: Any,
    *,
    fn: Any,
    op: str,
    threshold: float,
    terms: Sequence[str],
    conds: Sequence[Any],
    limit: int,
) -> list[Hit]:
    stmt = _trgm_stmt(col, fn=fn, op=op, threshold=threshold, terms=terms, conds=conds, limit=limit)
    try:
        rows = (await session.execute(stmt)).all()
    except Exception as exc:  # funksiya/operator yo'q bo'lsa \u2014 kanal jimgina bo'sh
        log.warning("search.trigram_failed", error=str(exc)[:200])
        return []
    return [Hit(int(r[0]), int(r[1]), float(r[2] or 0), "trigram") for r in rows]


async def fts_search(
    session: AsyncSession, chat_id: int, question: str, *, limit: int = 20
) -> list[Hit]:
    """Mos keluvchi leksik yo'l (uz tsv + trigram) \u2014 eski imzo saqlanadi."""
    terms = keywords(question)
    if not terms:
        return []
    uz = await fts_uz_search(session, chat_id, terms, limit=limit)
    tri = await trigram_search(session, chat_id, terms, limit=limit)
    fused = rrf(uz, tri, weights=(W_TSV_UZ, W_TRIGRAM))
    return fused[:limit]


# ─── qidiruv ─────────────────────────────────────────────────────────────────


async def vector_search(
    session: AsyncSession, chat_id: int, question: str, *, limit: int = 20, llm: LLM | None = None
) -> list[Hit]:
    """Embedding bor bo'lsa — cosine yaqinlik. Yo'q bo'lsa bo'sh (jimgina emas — log)."""
    has_any = (
        await session.execute(
            select(MessageEmbedding.message_id)
            .join(Message, Message.id == MessageEmbedding.message_id)
            .where(Message.chat_id == chat_id, *_embed_model(MessageEmbedding.model))
            .limit(1)
        )
    ).first()
    if not has_any:
        return []
    try:
        # Indeks DOCUMENT sifatida embed qilingan — so'rov QUERY bo'lishi kerak,
        # aks holda asimmetrik retrieval bepul aniqlik yo'qotadi (v2, 5.3).
        emb = await (llm or LLM()).embed([question], task_type=EMBED_QUERY)
    except LLMError as exc:
        log.warning("search.embed_failed", error=str(exc)[:200])
        return []
    qvec = emb.vectors[0]
    await tune_hnsw(session)
    dist = MessageEmbedding.vector.cosine_distance(qvec).label("dist")
    stmt = (
        select(Message.id, Message.tg_msg_id, dist)
        .join(MessageEmbedding, MessageEmbedding.message_id == Message.id)
        .where(Message.chat_id == chat_id, *_alive(), *_embed_model(MessageEmbedding.model))
        .order_by(dist)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [Hit(int(r[0]), int(r[1]), 1.0 - float(r[2] or 0), "vector") for r in rows]


def _as_lists(ranked: tuple[Any, ...]) -> list[Sequence[Hit]]:
    """`rrf(a, b)` ham, `rrf([a, b])` ham qabul qilinadi (eski chaqiruvlar buzilmaydi)."""
    if len(ranked) == 1 and ranked[0] and not isinstance(ranked[0][0], Hit):
        return list(ranked[0])
    return list(ranked)


def rrf(
    *ranked: Sequence[Hit] | Sequence[Sequence[Hit]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    """Og'irlikli reciprocal rank fusion — `w / (k + rank)`.

    Ballar o'lchovi har xil (ts_rank, cosine, trigram) bo'lgani uchun ular
    qo'shilmaydi — faqat **o'rin** qo'shiladi. Og'irlik kanalga ishonch
    darajasini beradi; berilmasa hamma kanal 1.0.
    """
    lists = _as_lists(ranked)
    acc: dict[int, Hit] = {}
    for idx, hits in enumerate(lists):
        w = weights[idx] if weights is not None and idx < len(weights) else 1.0
        for i, h in enumerate(hits):
            score = w / (k + i + 1)
            cur = acc.get(h.message_id)
            if cur is None:
                acc[h.message_id] = Hit(
                    h.message_id,
                    h.tg_msg_id,
                    score,
                    h.source,
                    chat_id=h.chat_id,
                    chat_title=h.chat_title,
                    sender=h.sender,
                    date=h.date,
                    text=h.text,
                    level=h.level,
                )
            else:
                cur.score += score
                cur.source = "both" if cur.source != h.source else cur.source
    return sorted(acc.values(), key=lambda h: h.score, reverse=True)


async def hybrid_search(
    session: AsyncSession,
    chat_id: int,
    question: str,
    *,
    limit: int = 20,
    expansion: Expansion | None = None,
    expand: bool = False,
    llm: LLM | None = None,
    weights: Sequence[float] | None = None,
) -> list[Hit]:
    """To'rt kanal (tsv_uz, tsv_ru, trigram, vektor) → og'irlikli RRF.

    `expand=True` bo'lsa (yoki tayyor `expansion` berilsa) so'rov ko'p tilli
    variantlarga yoyiladi. Default `False` — mavjud chaqiruvchilar (`tools.py`)
    xatti-harakati o'zgarmaydi.

    Kanallar **ketma-ket** yuradi: bitta `AsyncSession` ustida parallel so'rov
    xavfsiz emas (`asyncio.gather` bu yerda InterfaceError beradi).
    """
    base = keywords(question)
    if expansion is None and expand and get_settings().search_expand_enabled:
        from app.services.query_expand import expand as expand_query

        expansion = await expand_query(question, llm=llm)

    uz_terms = list(base)
    ru_terms = [t for t in base if _CYRILLIC_RE.search(t)]
    if expansion is not None:
        uz_terms += [t for t in expansion.latin_terms() if t not in uz_terms]
        ru_terms += [t for t in expansion.cyrillic_terms() if t not in ru_terms]

    if not uz_terms and not ru_terms:
        return []

    wide = limit * 2
    uz = await fts_uz_search(session, chat_id, uz_terms, limit=wide)
    ru = await fts_ru_search(session, chat_id, ru_terms, limit=wide)
    tri = await trigram_search(session, chat_id, uz_terms or ru_terms, limit=wide)
    vec = await vector_search(session, chat_id, question, limit=wide, llm=llm)

    fused = rrf(uz, ru, tri, vec, weights=weights or DEFAULT_RRF_WEIGHTS)
    log.debug(
        "search.hybrid",
        chat_id=chat_id,
        tsv_uz=len(uz),
        tsv_ru=len(ru),
        trigram=len(tri),
        vector=len(vec),
        fused=len(fused),
        expanded=bool(expansion and not expansion.is_empty),
    )
    return fused[:limit]


# ─── akkaunt bo'ylab qidiruv + tugun kanali (2-bosqich) ─────────────────────
#
# Muhim: bitta `AsyncSession` **parallel** ishlatilmaydi. Kanallar shu yerda
# ham ketma-ket yuradi — `asyncio.gather` asyncpg ulanishida `InterfaceError`
# beradi. Parallellik kerak bo'lsa har kanalga o'z sessiyasi ochilishi kerak.


@dataclass(slots=True)
class NodeHit:
    """`retrieval_nodes` dan topilma — sitata uchun `msg_lo..msg_hi` oralig'i bilan.

    Tugun matni javobda to'g'ridan-to'g'ri sitata qilinmaydi: chaqiruvchi
    oraliqdan **level 0** xabar id'larini oladi va sitata o'shalarga beriladi
    (halollik: "shu tugunda gapirilgan" emas, "shu xabarda yozilgan").
    """

    node_id: int
    chat_id: int
    level: int
    score: float
    source: str  # node_tsv | node_vector | both
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    msg_lo: int = 0
    msg_hi: int = 0
    msg_count: int = 0
    ctx_prefix: str = ""
    text: str = ""


# Tugun kanali RRF'da beshinchi kirish (v2, 6-bo'lim: tugunlar 0.8).
W_NODES = 0.8
DEFAULT_MULTI_RRF_WEIGHTS: tuple[float, float, float, float, float] = (
    W_TSV_UZ,
    W_TSV_RU,
    W_TRIGRAM,
    W_VECTOR,
    W_NODES,
)
#: Tugun kanalidan xabarga qaytarishda: nechta tugun va har biridan nechta xabar.
NODE_CITE_NODES = 12
NODE_CITE_PER_NODE = 2

# Entity kanali — RRF'da **oltinchi** kirish (phase4/A). Og'irligi leksik
# kanallardan ham yuqori (1.1): `@nurbek` tengligi "nurbek" so'zining matnda
# uchrashidan kuchliroq signal — u yerda taxmin yo'q, aniq identifikator bor.
# Kanal faqat planner entity bergan bo'lsa qo'shiladi, shuning uchun besh
# kanalli sukut og'irliklari (`DEFAULT_MULTI_RRF_WEIGHTS`) o'zgarmaydi.
W_ENTITY = 1.1
ENTITY_MULTI_RRF_WEIGHTS: tuple[float, float, float, float, float, float] = (
    *DEFAULT_MULTI_RRF_WEIGHTS,
    W_ENTITY,
)


def _scope(
    account_id: int,
    chat_ids: Sequence[int] | None,
    since: datetime | None,
    until: datetime | None,
) -> list[Any]:
    """Akkaunt + chat + sana oynasi shartlari (xabar kanallari uchun)."""
    conds: list[Any] = [Chat.account_id == account_id]
    if chat_ids:
        conds.append(Message.chat_id.in_(list(chat_ids)))
    if since is not None:
        conds.append(Message.published_at >= since)
    if until is not None:
        conds.append(Message.published_at < until)
    return conds


async def _tsv_multi(
    session: AsyncSession,
    col: Any,
    *,
    config: str,
    tsq: str,
    conds: Sequence[Any],
    limit: int,
    source: str,
) -> list[Hit]:
    q = func.to_tsquery(config, tsq)
    score = func.ts_rank_cd(col, q).label("score")
    stmt = (
        select(Message.id, Message.tg_msg_id, Message.chat_id, score)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*conds, *_alive(), col.op("@@")(q))
        .order_by(score.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [Hit(int(r[0]), int(r[1]), float(r[3] or 0), source, chat_id=int(r[2])) for r in rows]


async def _trigram_multi(
    session: AsyncSession, terms: Sequence[str], *, conds: Sequence[Any], limit: int
) -> list[Hit]:
    col = getattr(Message, "text_fold", None)
    folded = fold_terms(terms)[:4]
    if col is None or not folded:
        return []
    stmt = _trgm_stmt(
        col,
        fn=func.strict_word_similarity,
        op=STRICT_TRGM_OP,
        threshold=STRICT_TRGM_THRESHOLD,
        terms=folded,
        conds=[*conds, *_alive()],
        limit=limit,
        with_chat=True,
        join=(Chat, Chat.id == Message.chat_id),
    )
    try:
        rows = (await session.execute(stmt)).all()
    except Exception as exc:  # funksiya yo'q bo'lsa — kanal jimgina bo'sh
        log.warning("search.trigram_multi_failed", error=str(exc)[:200])
        return []
    return [Hit(int(r[0]), int(r[1]), float(r[3] or 0), "trigram", chat_id=int(r[2])) for r in rows]


async def _vector_multi(
    session: AsyncSession, qvec: Sequence[float], *, conds: Sequence[Any], limit: int
) -> list[Hit]:
    await tune_hnsw(session)
    dist = MessageEmbedding.vector.cosine_distance(list(qvec)).label("dist")
    stmt = (
        select(Message.id, Message.tg_msg_id, Message.chat_id, dist)
        .join(MessageEmbedding, MessageEmbedding.message_id == Message.id)
        .join(Chat, Chat.id == Message.chat_id)
        .where(*conds, *_alive(), *_embed_model(MessageEmbedding.model))
        .order_by(dist)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        Hit(int(r[0]), int(r[1]), 1.0 - float(r[3] or 0), "vector", chat_id=int(r[2])) for r in rows
    ]


async def tune_hnsw(session: AsyncSession) -> None:
    """Vektor kanalidan **oldin** HNSW skanini filtr bilan ishlashga sozlaydi.

    `SET LOCAL` (bu yerda `set_config(..., true)` — bitta statement, bitta
    round trip) tranzaksiya oxirida o'zi qaytadi, ya'ni boshqa so'rovlarga
    ta'sir qilmaydi. Xato bo'lsa (pgvector <0.8, GUC yo'q) kanal baribir
    ishlaydi — faqat eski, kam qamrovli xatti-harakat bilan.
    """
    try:
        await session.execute(_HNSW_TUNE_SQL)
    except Exception as exc:
        log.debug("search.hnsw_tune_failed", error=str(exc)[:200])


async def _query_vector(question: str, llm: LLM | None) -> list[float] | None:
    """So'rov uchun **bitta** embedding (`RETRIEVAL_QUERY`). Xato bo'lsa — None."""
    try:
        res = await (llm or LLM()).embed([question], task_type=EMBED_QUERY)
    except LLMError as exc:
        log.warning("search.embed_failed", error=str(exc)[:200])
        return None
    return list(res.vectors[0]) if res.vectors else None


def rrf_nodes(
    *ranked: Sequence[NodeHit], k: int = 60, weights: Sequence[float] | None = None
) -> list[NodeHit]:
    """`rrf` ning tugunlar uchun varianti — kalit `node_id`."""
    acc: dict[int, NodeHit] = {}
    for idx, hits in enumerate(ranked):
        w = weights[idx] if weights is not None and idx < len(weights) else 1.0
        for i, h in enumerate(hits):
            cur = acc.get(h.node_id)
            if cur is None:
                acc[h.node_id] = NodeHit(
                    h.node_id,
                    h.chat_id,
                    h.level,
                    w / (k + i + 1),
                    h.source,
                    starts_at=h.starts_at,
                    ends_at=h.ends_at,
                    msg_lo=h.msg_lo,
                    msg_hi=h.msg_hi,
                    msg_count=h.msg_count,
                    ctx_prefix=h.ctx_prefix,
                    text=h.text,
                )
            else:
                cur.score += w / (k + i + 1)
                cur.source = "both" if cur.source != h.source else cur.source
    return sorted(acc.values(), key=lambda h: h.score, reverse=True)


def _node_row_to_hit(row: Any, *, score: float, source: str) -> NodeHit:
    return NodeHit(
        node_id=int(row.id),
        chat_id=int(row.chat_id),
        level=int(row.level),
        score=score,
        source=source,
        starts_at=row.starts_at,
        ends_at=row.ends_at,
        msg_lo=int(row.msg_lo or 0),
        msg_hi=int(row.msg_hi or 0),
        msg_count=int(row.msg_count or 0),
        ctx_prefix=row.ctx_prefix or "",
        text=row.text or "",
    )


async def node_search(
    session: AsyncSession,
    account_id: int,
    query_vec_or_text: str | Sequence[float],
    *,
    levels: Sequence[int] = (1, 2, 3),
    chat_ids: Sequence[int] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 20,
    text: str | None = None,
    llm: LLM | None = None,
    weights: Sequence[float] | None = None,
) -> list[NodeHit]:
    """`retrieval_nodes` ustidan ikki kanalli qidiruv (tsv + vektor) → RRF.

    Tugunlar epizod/kun/hafta darajasida — ya'ni "ha, roziman" tipidagi
    xabarlar ham o'z konteksti bilan topiladi. Har topilma `msg_lo..msg_hi`
    oralig'ini beradi, sitata esa **level 0** xabarlarga qo'yiladi.

    Args:
        session: DB sessiyasi. **Parallel ishlatilmaydi** — kanallar ketma-ket.
        account_id: `accounts.id` (tugunlar akkaunt bo'yicha ajratilgan).
        query_vec_or_text: savol matni yoki tayyor so'rov vektori. Vektor
            berilsa leksik kanal uchun `text` ni ham bering.
        levels: qaysi darajalar (sukut: epizod, kun, hafta).
        chat_ids: faqat shu chatlar; `None` — akkauntdagi hammasi.
        since / until: `starts_at` oynasi.
        limit: qaytariladigan tugunlar soni.
        text: leksik kanal uchun matn (vektor berilganda).
        llm: matn berilib vektor berilmaganda embedding uchun.
        weights: `(tsv, vektor)` og'irliklari.

    Returns:
        `NodeHit` ro'yxati, RRF ballari bo'yicha kamayish tartibida.
    """
    from app.db.models import RetrievalNode, RetrievalNodeEmbedding

    qtext = query_vec_or_text if isinstance(query_vec_or_text, str) else text
    qvec: list[float] | None = None
    if not isinstance(query_vec_or_text, str):
        qvec = list(query_vec_or_text)
    elif llm is not None:
        qvec = await _query_vector(query_vec_or_text, llm)

    conds: list[Any] = [RetrievalNode.account_id == account_id]
    if levels:
        conds.append(RetrievalNode.level.in_(list(levels)))
    if chat_ids:
        conds.append(RetrievalNode.chat_id.in_(list(chat_ids)))
    if since is not None:
        conds.append(RetrievalNode.ends_at >= since)
    if until is not None:
        conds.append(RetrievalNode.starts_at < until)

    lex: list[NodeHit] = []
    folded = fold_terms(keywords(qtext)) if qtext else []
    tsq = _tsquery(folded, prefix=True, op=" & ") if folded else ""
    if tsq:
        q = func.to_tsquery("simple", tsq)
        score = func.ts_rank_cd(RetrievalNode.tsv, q).label("score")
        rows = (
            await session.execute(
                select(RetrievalNode, score)
                .where(*conds, RetrievalNode.tsv.op("@@")(q))
                .order_by(score.desc())
                .limit(limit * 2)
            )
        ).all()
        if not rows and len(folded) > 1:
            q = func.to_tsquery("simple", _tsquery(folded, prefix=True, op=" | "))
            score = func.ts_rank_cd(RetrievalNode.tsv, q).label("score")
            rows = (
                await session.execute(
                    select(RetrievalNode, score)
                    .where(*conds, RetrievalNode.tsv.op("@@")(q))
                    .order_by(score.desc())
                    .limit(limit * 2)
                )
            ).all()
        lex = [_node_row_to_hit(r[0], score=float(r[1] or 0), source="node_tsv") for r in rows]

    vec: list[NodeHit] = []
    if qvec is not None:
        await tune_hnsw(session)
        dist = RetrievalNodeEmbedding.vector.cosine_distance(qvec).label("dist")
        rows = (
            await session.execute(
                select(RetrievalNode, dist)
                .join(RetrievalNodeEmbedding, RetrievalNodeEmbedding.node_id == RetrievalNode.id)
                .where(*conds, *_embed_model(RetrievalNodeEmbedding.model))
                .order_by(dist)
                .limit(limit * 2)
            )
        ).all()
        vec = [
            _node_row_to_hit(r[0], score=1.0 - float(r[1] or 0), source="node_vector") for r in rows
        ]

    fused = rrf_nodes(lex, vec, weights=weights or (1.0, 1.0))
    log.debug("search.nodes", account_id=account_id, tsv=len(lex), vector=len(vec))
    return fused[:limit]


async def node_hits_to_messages(
    session: AsyncSession,
    node_hits: Sequence[NodeHit],
    *,
    per_node: int = NODE_CITE_PER_NODE,
    max_nodes: int = NODE_CITE_NODES,
) -> list[Hit]:
    """Tugun topilmalarini **level 0** xabarlarga qaytaradi (sitata uchun).

    Har tugun uchun **alohida** so'rov (`n <= NODE_CITE_NODES`, ya'ni ≤ 12):
    ilgari bitta katta `OR` so'rovi umumiy `LIMIT` bilan kesilardi va u
    `chat_id, tg_msg_id` bo'yicha tartiblangani uchun butun byudjetni birinchi
    chatlar yeb qo'yardi — ro'yxat oxiridagi tugunlarga hech narsa qolmasdi
    (L5). Endi har tugun aynan o'z `per_node` ulushini oladi.

    Tugunlarning RRF tartibi saqlanadi — bu kanal kirishi shu tartibda.
    """
    top = [h for h in list(node_hits)[:max_nodes] if h.msg_hi >= h.msg_lo > 0]
    if not top:
        return []
    out: list[Hit] = []
    seen: set[int] = set()
    for h in top:
        rows = (
            await session.execute(
                select(Message.id, Message.tg_msg_id)
                .where(
                    Message.chat_id == h.chat_id,
                    Message.tg_msg_id >= h.msg_lo,
                    Message.tg_msg_id <= h.msg_hi,
                    *_alive(),
                )
                .order_by(Message.tg_msg_id)
                .limit(per_node + len(seen))
            )
        ).all()
        picked = 0
        for mid, tg_id in rows:
            if int(mid) in seen:
                continue
            out.append(Hit(int(mid), int(tg_id), h.score, "node", chat_id=h.chat_id, level=h.level))
            seen.add(int(mid))
            picked += 1
            if picked >= per_node:
                break
    return out


async def entity_search(
    session: AsyncSession,
    account_id: int,
    entities: Sequence[str],
    *,
    chat_ids: Sequence[int] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 40,
) -> list[Hit]:
    """`node_entities` bo'yicha **aniq tenglik** kanali (phase4/A).

    Planner bergan satrlar (`@nurbek`, `#deploy`, `example.com`,
    `t.me/c/12/34`) shakli bo'yicha turga ajratiladi
    (`services/entities.parse_entity_query`) va indeksdan tenglik bilan
    qidiriladi. Reyting yo'q — bu yerda "qanchalik o'xshash" degan savol
    yo'q, shuning uchun tartib **yangilik** bo'yicha; ball esa RRF uchun
    o'rindan hosil qilinadi.

    Args:
        session: DB sessiyasi.
        account_id: `accounts.id` (indeks kaliti shu bilan boshlanadi).
        entities: xom satrlar; tanilmagani jimgina tashlab yuboriladi.
        chat_ids / since / until: `hybrid_search_multi` bilan bir xil doira.
        limit: natija soni.

    Returns:
        `source="entity"` bo'lgan `Hit` ro'yxati (yangidan eskiga).
    """
    from app.services.entities import resolve_entities

    pairs = resolve_entities(list(entities or []))
    if not pairs:
        return []
    from app.db.models import NodeEntity

    match = or_(
        *[and_(NodeEntity.kind == kind, NodeEntity.value == value) for kind, value in pairs]
    )
    # `published_at` select ro'yxatida — `SELECT DISTINCT` da tartiblash ifodasi
    # chiqishda ham bo'lishi shart (Postgres talabi).
    stmt = (
        select(Message.id, Message.tg_msg_id, Message.chat_id, Message.published_at)
        .join(NodeEntity, NodeEntity.message_id == Message.id)
        .join(Chat, Chat.id == Message.chat_id)
        .where(
            *_scope(account_id, chat_ids, since, until),
            *_alive(),
            NodeEntity.account_id == account_id,
            match,
        )
        .order_by(Message.published_at.desc())
        .distinct()
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    return [
        Hit(
            int(r[0]),
            int(r[1]),
            1.0 / (i + 1),
            "entity",
            chat_id=int(r[2]),
            date=r[3] if len(r) > 3 else None,
        )
        for i, r in enumerate(rows)
    ]


async def hybrid_search_multi(
    session: AsyncSession,
    account_id: int,
    question: str,
    *,
    chat_ids: Sequence[int] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 40,
    expansion: Expansion | None = None,
    expand: bool = False,
    llm: LLM | None = None,
    weights: Sequence[float] | None = None,
    levels: Sequence[int] = (1, 2, 3),
    entities: Sequence[str] | None = None,
) -> list[Hit]:
    """Akkaunt bo'ylab besh kanalli qidiruv: tsv_uz, tsv_ru, trigram, vektor, tugunlar.

    `hybrid_search` bitta chat uchun; bu — "burchakdan burchakka" varianti
    (v2, 2-bo'lim): bir nechta chat, sana oynasi, tugun kanali. Tugun
    topilmalari `node_hits_to_messages` orqali level-0 xabarlarga qaytariladi,
    shuning uchun natija **bir xil** `Hit` turida bo'ladi va sitata har doim
    haqiqiy `#tg_msg_id` ga tushadi.

    Kanallar **ketma-ket** bajariladi: bitta `AsyncSession` ni parallel
    ishlatib bo'lmaydi (asyncpg `InterfaceError`).

    Args:
        session: DB sessiyasi.
        account_id: `accounts.id`.
        question: foydalanuvchi savoli (untrusted matn emas).
        chat_ids: qidiruv doirasi; `None` — akkauntdagi hamma chat.
        since / until: sana oynasi (`published_at`).
        limit: natija soni.
        expansion: tayyor ko'p tilli kengaytma; `expand=True` bo'lsa o'zi oladi.
        llm: LLM fasadi (embedding va kengaytirish uchun).
        weights: besh kanal og'irligi.
        levels: tugun kanalidagi darajalar.
        entities: planner ajratgan identifikatorlar (`@nurbek`, `#deploy`,
            `example.com`). Berilsa — **oltinchi** RRF kirishi
            (`entity_search`, og'irlik `W_ENTITY`); berilmasa kanal umuman
            ishga tushmaydi va og'irliklar besh kanalli holicha qoladi.

    Returns:
        RRF bo'yicha tartiblangan `Hit` ro'yxati (`chat_id` to'ldirilgan).
    """
    base = keywords(question)
    if expansion is None and expand and get_settings().search_expand_enabled:
        from app.services.query_expand import expand as expand_query

        expansion = await expand_query(question, llm=llm)

    uz_terms = list(base)
    ru_terms = [t for t in base if _CYRILLIC_RE.search(t)]
    if expansion is not None:
        uz_terms += [t for t in expansion.latin_terms() if t not in uz_terms]
        ru_terms += [t for t in expansion.cyrillic_terms() if t not in ru_terms]
    ent_terms = list(entities or [])
    if not uz_terms and not ru_terms and not ent_terms:
        return []

    conds = _scope(account_id, chat_ids, since, until)
    wide = limit * 2

    uz: list[Hit] = []
    folded = fold_terms(uz_terms)
    uz_col = getattr(Message, "tsv_uz", None)
    if uz_col is not None and folded:
        uz = await _tsv_multi(
            session,
            uz_col,
            config="simple",
            tsq=_tsquery(folded, prefix=True, op=" & "),
            conds=conds,
            limit=wide,
            source="tsv_uz",
        )
        if not uz and len(folded) > 1:
            uz = await _tsv_multi(
                session,
                uz_col,
                config="simple",
                tsq=_tsquery(folded, prefix=True, op=" | "),
                conds=conds,
                limit=wide,
                source="tsv_uz",
            )

    ru: list[Hit] = []
    ru_col = getattr(Message, "tsv_ru", None)
    ru_clean = [t for t in (_tsq_safe(x.lower()) for x in ru_terms) if len(t) >= 2]
    if ru_col is not None and ru_clean:
        ru = await _tsv_multi(
            session,
            ru_col,
            config="russian",
            tsq=_tsquery(ru_clean, prefix=False, op=" & "),
            conds=conds,
            limit=wide,
            source="tsv_ru",
        )

    tri = await _trigram_multi(session, uz_terms or ru_terms, conds=conds, limit=wide)

    qvec = await _query_vector(question, llm)
    vec = await _vector_multi(session, qvec, conds=conds, limit=wide) if qvec else []

    node_hits = await node_search(
        session,
        account_id,
        qvec if qvec is not None else question,
        text=question,
        levels=levels,
        chat_ids=chat_ids,
        since=since,
        until=until,
        limit=NODE_CITE_NODES,
    )
    nodes = await node_hits_to_messages(session, node_hits)

    ents: list[Hit] = []
    if ent_terms:
        ents = await entity_search(
            session,
            account_id,
            ent_terms,
            chat_ids=chat_ids,
            since=since,
            until=until,
            limit=wide,
        )

    lists = [uz, ru, tri, vec, nodes]
    w = list(weights or DEFAULT_MULTI_RRF_WEIGHTS)
    if ent_terms:
        lists.append(ents)
        if len(w) < len(lists):
            w.append(W_ENTITY)
    fused = rrf(*lists, weights=w)
    log.debug(
        "search.hybrid_multi",
        account_id=account_id,
        chats=len(chat_ids) if chat_ids else 0,
        tsv_uz=len(uz),
        tsv_ru=len(ru),
        trigram=len(tri),
        vector=len(vec),
        nodes=len(node_hits),
        entities=len(ents),
        fused=len(fused),
    )
    return fused[:limit]


# ─── hit metadatasi / vektorlar ──────────────────────────────────────────────


async def hydrate_hits(session: AsyncSession, chat: Chat, hits: list[Hit]) -> list[Hit]:
    """Rerank va dedup uchun matn/sana/jo'natuvchi to'ldiradi (bitta so'rov)."""
    if not hits:
        return hits
    ids = [h.message_id for h in hits]
    rows = (await session.execute(select(Message).where(Message.id.in_(ids)))).scalars().all()
    by_id = {m.id: m for m in rows}
    for h in hits:
        m = by_id.get(h.message_id)
        if m is None:
            continue
        info = _row_to_info(m, chat)
        h.chat_id = chat.id
        h.chat_title = chat.title
        h.sender = info.sender
        h.date = m.published_at
        h.text = info.text
    return hits


async def fetch_vectors(
    session: AsyncSession, message_ids: Sequence[int]
) -> dict[int, list[float]]:
    """`message_embeddings` dan vektorlar (dedup/klaster uchun). Yo'qlari tushib qoladi."""
    if not message_ids:
        return {}
    rows = (
        await session.execute(
            select(MessageEmbedding.message_id, MessageEmbedding.vector).where(
                MessageEmbedding.message_id.in_(list(message_ids)),
                *_embed_model(MessageEmbedding.model),
            )
        )
    ).all()
    out: dict[int, list[float]] = {}
    for mid, vec in rows:
        if vec is None:
            continue
        out[int(mid)] = [float(x) for x in vec]
    return out


# ─── kontekst tanlash ────────────────────────────────────────────────────────


def _row_to_info(m: Message, chat: Chat) -> MessageInfo:
    is_channel = chat.type == ChatType.CHANNEL
    sender = (
        chat.title
        if is_channel and not m.sender_id
        else (f"user:{m.sender_id}" if m.sender_id else "—")
    )
    return MessageInfo(
        msg_id=m.tg_msg_id,
        date=m.published_at,
        sender=sender,
        text=compact_text(m.text or ""),
        media_type=m.media_type,
        views=m.views,
    )


def fit_budget(infos: list[MessageInfo], budget_tokens: int) -> tuple[list[MessageInfo], bool]:
    """Byudjetga sig'diradi — eng yangilarini saqlab (ro'yxat eskidan yangiga)."""
    kept: list[MessageInfo] = []
    used = 0
    for m in reversed(infos):
        cost = est_tokens(m.text) + 12  # sarlavha/vaqt/muallif ustama
        if used + cost > budget_tokens and kept:
            return list(reversed(kept)), True
        kept.append(m)
        used += cost
    return list(reversed(kept)), False


async def _chat_row(session: AsyncSession, account_id: int, peer_id: int) -> Chat | None:
    chat = (
        await session.execute(
            select(Chat).where(Chat.account_id == account_id, Chat.tg_peer_id == peer_id)
        )
    ).scalar_one_or_none()
    if chat is None or not chat.synced_total:
        return None
    return chat


async def recent_context(
    session: AsyncSession, chat: Chat, *, limit: int, budget_tokens: int
) -> ContextBundle:
    rows = await session.execute(
        select(Message)
        .where(Message.chat_id == chat.id)
        .order_by(Message.tg_msg_id.desc())
        .limit(limit)
    )
    infos = [_row_to_info(m, chat) for m in reversed(rows.scalars().all())]
    infos, trunc = fit_budget(infos, budget_tokens)
    return ContextBundle(
        chat.title,
        infos,
        "recent",
        est_tokens=sum(est_tokens(i.text) for i in infos),
        truncated=trunc,
    )


def fit_by_priority(
    ordered: Sequence[tuple[int, MessageInfo]], budget_tokens: int
) -> tuple[list[MessageInfo], bool]:
    """Ustuvorlik tartibida byudjetga sig'diradi, natijani xronologik qaytaradi.

    `fit_budget` "eng yangisini saqla" qoidasi bilan ishlaydi — qidiruv rejimida
    bu **xato** edi: oxirgi 15 xabarlik dum topilgan javobni siqib chiqarardi.
    Bu yerda tartib chaqiruvchidan keladi (avval reranked hit'lar, keyin
    qo'shnilar, oxirida dum), chiqish esa o'qish uchun vaqt bo'yicha saralanadi.
    """
    picked: dict[int, MessageInfo] = {}
    used = 0
    truncated = False
    for key, info in ordered:
        if key in picked:
            continue
        cost = est_tokens(info.text) + 12
        if picked and used + cost > budget_tokens:
            truncated = True
            break
        picked[key] = info
        used += cost
    return [picked[k] for k in sorted(picked)], truncated


async def search_context(
    session: AsyncSession,
    chat: Chat,
    question: str,
    *,
    budget_tokens: int,
    top_k: int = 20,
    llm: LLM | None = None,
    expand: bool = False,
    rerank_enabled: bool = True,
) -> ContextBundle:
    """Mos xabarlar (± qo'shnilari) + qisqa oxirgi dum — xronologik tartibda.

    Quvur: to'rt kanal → RRF → dedup (sha1/simhash) → rerank → byudjet.
    Dedup rerank'dan **oldin**: aks holda model bir xil forwardning o'nta
    nusxasini tartiblaydi va byudjet bitta matn bilan to'ladi.
    """
    # kech import — `cluster`/`rerank` o'z navbatida `search` ni ishlatadi
    from app.services import cluster as CL
    from app.services.rerank import rerank as rerank_hits

    hits = await hybrid_search(session, chat.id, question, limit=top_k, expand=expand, llm=llm)
    hits = await hydrate_hits(session, chat, hits)
    hits = CL.dedup(hits)
    if rerank_enabled and len(hits) > 1:
        hits = await rerank_hits(question, hits, llm=llm)

    hit_ids = [h.tg_msg_id for h in hits]
    neighbour_ids = [h.tg_msg_id + d for h in hits for d in range(-NEIGHBORS, NEIGHBORS + 1) if d]
    tail_rows = await session.execute(
        select(Message.tg_msg_id)
        .where(Message.chat_id == chat.id, *_alive())
        .order_by(Message.tg_msg_id.desc())
        .limit(RECENT_TAIL)
    )
    tail_ids = [int(r[0]) for r in tail_rows.all()]

    wanted = list(dict.fromkeys(hit_ids + neighbour_ids + tail_ids))
    if not wanted:
        return ContextBundle(chat.title, [], "search", hits=0)
    rows = await session.execute(
        select(Message)
        .where(Message.chat_id == chat.id, *_alive(), Message.tg_msg_id.in_(wanted))
        .order_by(Message.tg_msg_id)
    )
    by_tg = {m.tg_msg_id: _row_to_info(m, chat) for m in rows.scalars().all()}
    ordered = [(mid, by_tg[mid]) for mid in wanted if mid in by_tg]
    infos, trunc = fit_by_priority(ordered, budget_tokens)
    return ContextBundle(
        chat.title,
        infos,
        "search",
        est_tokens=sum(est_tokens(i.text) for i in infos),
        truncated=trunc,
        hits=len(hits),
    )


async def window_context(
    session: AsyncSession, chat: Chat, *, since: datetime, until: datetime | None = None
) -> list[MessageInfo]:
    until = until or datetime.now(UTC)
    rows = await session.execute(
        select(Message)
        .where(
            Message.chat_id == chat.id, Message.published_at >= since, Message.published_at <= until
        )
        .order_by(Message.tg_msg_id)
    )
    return [_row_to_info(m, chat) for m in rows.scalars().all()]


# ─── vaqt oynasi aniqlash (savoldan) ─────────────────────────────────────────

_WINDOW_PATTERNS: list[tuple[re.Pattern[str], timedelta]] = [
    (re.compile(r"\b(bugun|сегодня|today)\b", re.I), timedelta(days=1)),
    (re.compile(r"\b(kecha|вчера|yesterday)\b", re.I), timedelta(days=2)),
    (
        re.compile(
            r"\b(bu hafta|shu hafta|haftada|на этой неделе|за неделю|this week|last week|weekly)\b",
            re.I,
        ),
        timedelta(days=7),
    ),
    (
        re.compile(
            r"\b(bu oy|shu oy|oyda|за месяц|в этом месяце|this month|last month|monthly)\b", re.I
        ),
        timedelta(days=31),
    ),
    (
        re.compile(r"\b(bu yil|shu yil|yilda|за год|в этом году|this year)\b", re.I),
        timedelta(days=366),
    ),
]
_SUMMARY_HINT = re.compile(
    r"\b(xulosa|umumlashtir|nima (bo'ldi|muhokama)|обсужда|итог|резюм|summar"
    r"|what (was|happened)|overview|recap|tahlil)\w*",
    re.I,
)


def detect_window(question: str, *, now: datetime | None = None) -> timedelta | None:
    for pat, delta in _WINDOW_PATTERNS:
        if pat.search(question):
            return delta
    return None


def looks_like_summary(question: str) -> bool:
    return bool(_SUMMARY_HINT.search(question))


# ─── map-reduce siqish ───────────────────────────────────────────────────────

_MAP_PROMPT = MAP_DIGEST_PROMPT


async def compact_window(
    infos: list[MessageInfo], *, title: str, llm: LLM | None = None
) -> ContextBundle:
    """Katta oynani bo'laklab arzon model bilan xulosalaydi (map), natija — digest'lar."""
    lines = [_line(m) for m in infos]
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) > MAP_CHUNK_CHARS and cur:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    dropped = 0
    if len(chunks) > MAP_MAX_CHUNKS:
        dropped = len(chunks) - MAP_MAX_CHUNKS
        chunks = chunks[-MAP_MAX_CHUNKS:]  # eng yangi bo'laklar

    client = llm or LLM()
    digests: list[str] = []
    t_in = t_out = 0
    model = ""
    for i, chunk in enumerate(chunks, 1):
        msgs = [
            Msg.system(_MAP_PROMPT),
            Msg.user(
                f'<untrusted_data source="telegram" chat="{U.attr(title)}" '
                f'part="{i}/{len(chunks)}">\n'
                f"{U.neutralize(chunk)}\n</untrusted_data>"
            ),
        ]
        try:
            res = await client.chat(Task.ROUTE, msgs, max_tokens=400)
        except LLMError as exc:
            log.warning("search.map_failed", part=i, error=str(exc)[:200])
            continue
        digests.append(f"[part {i}/{len(chunks)}] {res.text.strip()}")
        t_in += res.usage.tokens_in
        t_out += res.usage.tokens_out
        model = res.model
    return ContextBundle(
        title,
        [],
        "window_compacted",
        est_tokens=sum(est_tokens(d) for d in digests),
        truncated=dropped > 0,
        map_summaries=digests,
        map_tokens_in=t_in,
        map_tokens_out=t_out,
        map_model=model,
    )


async def digest_window(
    session: AsyncSession, chat: Chat, *, since: datetime, llm: LLM | None = None
) -> ContextBundle:
    """Katta oyna → kunlik digest keshi (`services/digests`); yangi kunlar token sarflaydi."""
    from app.services.digests import digests_for_window

    days = await digests_for_window(session, chat, since=since, llm=llm)
    summaries = [f"[{d.day.date()}] ({d.msg_count} msgs) {d.text}" for d in days]
    return ContextBundle(
        chat.title,
        [],
        "window_compacted",
        est_tokens=sum(est_tokens(x) for x in summaries),
        truncated=False,
        map_summaries=summaries,
        map_tokens_in=sum(d.tokens_in for d in days),
        map_tokens_out=sum(d.tokens_out for d in days),
        map_model=next((d.model for d in days if d.model), ""),
    )


def _line(m: MessageInfo) -> str:
    when = m.date.strftime("%Y-%m-%d %H:%M") if m.date else "?"
    body = m.text or (f"[{m.media_type}]" if m.media_type else "")
    views = f" (views: {m.views})" if m.views else ""
    return f"[{when}] #{m.msg_id} {m.sender}{views}: {body}"


# ─── asosiy kirish nuqtasi ───────────────────────────────────────────────────


async def select_context(
    session: AsyncSession,
    *,
    account_id: int,
    peer_id: int,
    question: str,
    strategy: str = "auto",
    limit: int = 50,
    deep: bool = False,
    llm: LLM | None = None,
) -> ContextBundle | None:
    """DB'dan strategiyaga ko'ra kontekst. Chat sinxronlanmagan bo'lsa None (jonli yo'l).

    recent  — oxirgi `limit` xabar (byudjetgacha)
    search  — savolga mos xabarlar (FTS+vektor, ± qo'shni) + qisqa dum
    window  — savoldagi vaqt oynasi (bugun/hafta/oy); katta bo'lsa map-reduce
    auto    — vaqt oynasi bo'lsa window; savol qidiruvga o'xshasa search; aks holda recent
    """
    chat = await _chat_row(session, account_id, peer_id)
    if chat is None:
        return None
    budget = DEEP_CONTEXT_TOKENS if deep else DEFAULT_CONTEXT_TOKENS
    s = get_settings()
    limit = max(5, min(limit, s.web_context_max_messages))

    if strategy == "auto":
        if detect_window(question) is not None:
            strategy = "window"
        elif keywords(question) and not looks_like_summary(question):
            strategy = "search"
        else:
            strategy = "recent"

    if strategy == "search":
        bundle = await search_context(
            session,
            chat,
            question,
            budget_tokens=budget,
            llm=llm,
            expand=s.search_expand_enabled,
            rerank_enabled=s.rerank_provider.lower() != "none",
        )
        if bundle.messages:
            return bundle
        strategy = "recent"  # hech narsa topilmadi — oxirgilar

    if strategy == "window":
        delta = detect_window(question) or timedelta(days=7)
        since = datetime.now(UTC) - delta
        infos = await window_context(session, chat, since=since)
        if not infos:
            return ContextBundle(chat.title, [], "window")
        total = sum(est_tokens(i.text) + 12 for i in infos)
        if total <= budget:
            return ContextBundle(chat.title, infos, "window", est_tokens=total)
        return await digest_window(session, chat, since=since, llm=llm)

    return await recent_context(session, chat, limit=limit, budget_tokens=budget)


__all__ = [
    "DEFAULT_MULTI_RRF_WEIGHTS",
    "DEFAULT_RRF_WEIGHTS",
    "ENTITY_MULTI_RRF_WEIGHTS",
    "HNSW_EF_SEARCH",
    "STRICT_TRGM_THRESHOLD",
    "TRGM_CANDIDATE_CAP",
    "TRGM_THRESHOLD",
    "ContextBundle",
    "Hit",
    "NodeHit",
    "compact_text",
    "compact_window",
    "detect_window",
    "digest_window",
    "entity_search",
    "est_tokens",
    "fetch_vectors",
    "fit_budget",
    "fit_by_priority",
    "fold_terms",
    "fts_ru_search",
    "fts_search",
    "fts_uz_search",
    "hybrid_search",
    "hybrid_search_multi",
    "hydrate_hits",
    "keywords",
    "looks_like_summary",
    "node_hits_to_messages",
    "node_search",
    "rrf",
    "rrf_nodes",
    "search_context",
    "select_context",
    "trigram_search",
    "tune_hnsw",
    "vector_search",
]
