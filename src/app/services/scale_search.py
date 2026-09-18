"""Korpus miqyosidagi kaskad — `deep`/`exhaustive` rejimlarining qidiruv yadrosi.

`deep_search.fan_out` **so'rovdan** boshlaydi: 3-8 subquery, har biri ~40
topilma, jami ≤400 nomzod. Bu "savolga o'xshagan xabar bormi?" degan savolga
javob beradi va arxiv katta bo'lganda yetarli: reyting yuqorisidagi 400 ta
qator butun korpusning ming'dan bir qismi. "Burchakdan burchakka" degani esa
boshqa narsa — **korpusni bo'lib chiqish** va har bo'lakdan olingan nomzodni
arzon model bilan elakdan o'tkazish.

    PLAN (deep_search)            savol → subquery, oyna, chatlar
      ↓
    (a) plan_buckets              chat x ISO-hafta bo'laklari, kosinus bo'yicha
      ↓                             tartiblangan (tugun vektorlari), ≤400 ta
    (b) pull_candidates           har bucket ichida 4 kanalli RRF (parallel
      ↓                             sessiyalar), elbow kesish → ≤3000 nomzod
    (c) dedup_fast                sha1 + **banded** simhash — O(n), 20k uchun
      ↓                             ham chidaydi (cosine bu yerda yugurmaydi)
    (d) filter_batched            Flash-Lite, 50 satrlik partiya, 0..3 ball,
      ↓                             `relevance_cache`, erta to'xtash, narx qopqog'i
    (e) select_evidence           ball bo'yicha + chat/hafta kvotasi, ≤1000
      ↓
    (f) map → reduce              bo'lakdan da'volar (sitatalar bo'lak ichida
      ↓                             tekshiriladi) → yagona da'volar dayjesti
    SYNTH (deep_search)           da'volar + top ~150 dalil, **bitta** konvert

Nega aynan shunday:

* **Elak arzon modelda.** Nomzod soni ikki tartibga o'sadi, shuning uchun
  har nomzodni kuchli modelga ko'rsatib bo'lmaydi. `Task.ROUTE`
  (Flash-Lite) 50 satrni bir chaqiruvda ko'radi va faqat indeks + ball
  qaytaradi — chiqish ~60 token, prompt injection eng yomoni tartibni
  buzadi.
* **Dedup O(n) bo'lishi shart.** `cluster.dedup_simhash` har yangi imzoni
  saqlanganlarning hammasi bilan solishtiradi (O(n²)); 20 000 nomzodda bu
  200M taqqoslash. Bu yerda imzo 4 ta 16-bitli bandga bo'linadi: hamming
  ≤3 bo'lsa kaptarxona tamoyili bo'yicha kamida bitta band **aynan** mos
  keladi, ya'ni nomzodlar faqat shu bandni bo'lishganlar ichidan izlanadi.
  Kosinus/klaster (`cluster.dedup`, `cluster_topics`) faqat filtrdan
  keyingi ≤1000 ta ustida yuguradi — o'sha yerda O(n²) halol.
* **Karantin tomoni.** `filter` va `map` prompt'lari begona matn ko'radi,
  lekin ularning chiqishi mexanik: filtr — indeks va ball, map — da'vo
  matni va **o'sha bo'lakda bor** id'lar. Reja va refleksiya bu matnni
  ko'rmaydi (`deep_search._Taint`, `QUARANTINED_STAGES`).
* **Narx qopqog'i kodda.** `COST_CEILING` filtr va map orasida tekshiriladi;
  oshsa bosqich to'xtaydi, `truncated=True` va `stopped_by="ceiling"`
  qamrovga tushadi. "Javob to'liqroq bo'lsin" deb olib tashlanmaydi.

Konstantalar o'lchovdan keyin qo'yilgan (`evals/perf/bench.py --sections
scale`), o'zgartirishdan oldin qayta o'lchang — **kamida 3 marta**.

TODO (keyingi faza): `exhaustive` hozircha sinxron yuguradi
(`EXHAUSTIVE_WALL_S`). Filtr bosqichi tabiatan partiyali va idempotent —
`filter_batched` ni Gemini Batch API yoki `worker/tasks.py` vazifasiga
ko'chirish uchun seam shu funksiyada: u faqat `(hits, q_hash)` oladi va
`(scores, stats)` qaytaradi, boshqa holatga tegmaydi.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chat, Message, RetrievalNode, RetrievalNodeEmbedding
from app.llm import LLM, LLMError, Msg, Task
from app.llm.base import EMBED_QUERY
from app.llm.pricing import PRICES
from app.logging import get_logger
from app.services import cluster as C
from app.services import deep_search as DS
from app.services import nodes as N
from app.services import search as S
from app.services import untrusted as U
from app.services.prompts_deep import (
    DEEP_SYNTH_PROMPT,
    SCALE_FILTER_PROMPT,
    SCALE_MAP_PROMPT,
    SCALE_REDUCE_PROMPT,
)
from app.services.textnorm import uz_fold

log = get_logger(__name__)

# ─── rejimlar va byudjetlar (kodda, o'zgarmas) ───────────────────────────────

MODES = ("fast", "deep", "exhaustive")
SCALE_MODES = ("deep", "exhaustive")

#: (a) bucket rejasi
BUCKET_MAX = 400  # chat x hafta bo'laklarining qattiq qopqog'i (`exhaustive`)
#: `deep` uchun pastroq qopqoq — **o'lchovdan**, taxmindan emas. 2.66M
#: xabarli jonli korpusda (`tgai-api-1`, faqat o'qish, 3 savol, oyma-oy
#: zaxira bucket'lari) `pull_candidates`:
#:
#:     cap=100  p50 40.1 s   p95 41.3 s
#:     cap=200  p50 34.9 s   p95 40.5 s
#:     cap=400  p50 47.0 s   p95 75.2 s     <- pull byudjetidan chiqadi
#:
#: `deep` da pull byudjeti ≈70 s (120 s devor minus `FAN_OUT_RESERVE_S` va
#: `PULL_RESERVE_S`; o'lchov paytida 53 s edi — zaxira 45 s edi), ya'ni
#: 400 ta bucket p95 da byudjetni yeb, javobni jimgina kesardi. 200 esa
#: p95 da ham sig'adi. `exhaustive` ning devori 600 s — u to'liq 400 ni
#: oladi. Qiziq tafsilot: narx bucket **soniga** chiziqli emas —
#: katta chatlarning bir nechta oyi butun vaqtni yeydi, shuning uchun
#: 100 va 200 orasida farq deyarli yo'q.
BUCKET_MAX_DEEP = 200
BUCKET_NODE_LEVELS = (2, 3)  # kun va hafta tugunlari (epizod juda mayda)
BUCKET_NODE_FETCH = 3 * BUCKET_MAX  # har vektor uchun olinadigan tugun soni
BUCKET_FALLBACK_CHATS = 40  # tugunsiz chatlar uchun (chat, oy) zaxira yo'li
#: `exhaustive` + oynasiz savol: zaxira grid **hamma** nishon chat uchun
#: quriladi (tuguni borlari ham). Jonli yugurishda tugunlar faqat oxirgi
#: haftalarda bor edi, ya'ni reja 5-7 bucket / 1 hafta bilan chegaralanardi —
#: "burchakdan burchakka" degan rejim uchun bu yolg'on.
BUCKET_FALLBACK_CHATS_FULL = 200
PLAN_CHAT_BONUS = 0.10  # reja aniq nomlagan chat — ballga qo'shimcha
FALLBACK_SCORE = 0.35  # (chat, oy) bucket'ining bazaviy bali

#: (b) bucket ichidan nomzod tortish
PER_BUCKET_K = 30
PULL_CONCURRENCY = 6  # parallel sessiyalar (asyncpg: bitta sessiya = bitta task)
POOL_MAX = 20_000
#: Elbow kesish shundan pastga tushmaydi. **200 → 500** (jonli o'lchov): ikkita
#: real yugurishda tirsak 2058→239 va 4170→278 ga kesgan, dalil esa 1-20 ta
#: chiqqan — `FILTER_MAX` (2000) qopqog'i hech qachon ishlamagan va "hech
#: qayerda yo'q" degan javob aslida hovuzning 6% idan kelgan.
ELBOW_MIN = 500
ELBOW_MAX = 3_000  # va shundan yuqoriga ko'tarilmaydi (`deep`)
#: Tirsak hovuzning ko'pi bilan 75% ini tashlaydi. RRF ballari silliq
#: kamayadi, ya'ni "eng katta sakrash" ko'pincha shovqin — qopqoqsiz u
#: hovuzning 90% ini yeb qo'yadi. Haqiqiy shift `FILTER_MAX`.
ELBOW_KEEP_MIN_FRAC = 0.25
#: Filtr/map/sintez uchun devordan qoldiriladigan vaqt. **45 → 30 s**: jonli
#: yugurishlarda pull'dan keyingi ish (dedup + filtr to'lqinlari + tanlov +
#: map/reduce + sintez) eng yomon holatda 30-40 s oldi, ya'ni 45 s zaxira
#: pull byudjetidan 15 s ni behuda olib qo'yardi. Derivatsiya o'sha:
#: `pull_budget = qolgan vaqt - zaxira`, pastdan `PULL_MIN_S` bilan.
PULL_RESERVE_S = 30.0
PULL_MIN_S = 5.0

#: (c) dedup — O(n)
SIMHASH_BANDS = 4  # 64 bit / 4 = 16 bitli band; hamming <=3 -> kamida 1 band mos
#: Bitta band chelagida saqlanadigan imzolar soni. Chegara **kerak**: korpus
#: bir xil shablonli xabarlarga to'la bo'lsa (bot bildirishnomalari, forward)
#: bitta chelak minglab imzoni yig'ib qoladi va band indeksi O(n^2) ga
#: qaytadi. Cheklangan chelak dedupning bir qismini o'tkazib yuborishi
#: mumkin — bu xavfsiz tomonga xato: ortiqcha nomzod filtrda tushadi,
#: kvadratik skan esa butun yugurishni o'ldiradi.
BAND_BUCKET_MAX = 32
CLUSTER_INPUT_MAX = 1_000  # kosinus/klaster faqat shundan kichik to'plamda

#: (d) filtr
FILTER_BATCH = 50
FILTER_PARALLEL = 4
FILTER_KEEP_MIN = 2  # 0..3 ball; 1 = "o'tkinchi eslatma", dalilga kirmaydi
FILTER_LINE_CHARS = 150
FILTER_MAX: dict[str, int] = {"fast": 0, "deep": 2_000, "exhaustive": 10_000}
FILTER_MAX_TOKENS = 900
EARLY_STOP_BATCHES = 5  # oxirgi shuncha partiya bo'yicha qaraladi
EARLY_STOP_KEEP_RATE = 0.05
EARLY_STOP_COVERAGE = 0.90  # bucket qamrovi shundan yuqori bo'lsa to'xtash mumkin

#: (e) tanlov
SOFT_MAX_EVIDENCE = 1_000
PER_CHAT_QUOTA = 5  # har chatdan kamida shuncha (bor bo'lsa)
PER_WEEK_QUOTA = 3  # har haftadan kamida shuncha (bor bo'lsa)

#: (f) map / reduce
MAP_CHUNK_MSGS = 80
#: Bo'lakning **pastki** chegarasi: 10 qatorli dalil 10 ta map chaqirig'iga
#: aylanib ketardi (har (chat, hafta) juftligi o'z bo'lagi edi), ya'ni
#: map narxi dalil soniga emas, uning **tarqoqligiga** bog'liq edi. Endi
#: qo'shni kichik bo'laklar **bitta chat** ichida birlashtiriladi.
MAP_CHUNK_MIN = 5
#: Dalil shundan kam bo'lsa map/reduce umuman yugurmaydi: da'volarni ajratish
#: uchun 15 qator sintezning o'ziga baribir to'liq sig'adi
#: (`SYNTH_EVIDENCE_LINES` = 150), ya'ni map faqat ikkinchi marta to'laydi.
MAP_SKIP_TOTAL = 15
MAP_MAX_CHUNKS = 16
MAP_PARALLEL = 4
MAP_LINE_CHARS = 300
MAP_MAX_TOKENS = 1_600
MAP_MAX_CLAIMS = 12
REDUCE_MAX_TOKENS = 3_000
REDUCE_DIGEST_TOKENS = 6_000  # da'volar bloki sintezga shu byudjet bilan tushadi
REDUCE_MAX_CLAIMS = 40
REDUCE_CITES_MAX = 6
#: Reduce'ni **darhol** `Task.DEEP` ga ko'taradigan niyatlar: bu ikkitasida
#: javob da'volar orasidagi **munosabat** (farq, dinamika), ya'ni arzon model
#: ro'yxatni yig'adi-yu, tartibni buzadi.
DEEP_REDUCE_INTENTS = ("compare", "trend")

#: Sintezga tushadigan dalil satrlari — da'volar bilan **bitta** konvertda.
#: Nega kerak: sitatalar validatori (`validate_citations`) dalil ro'yxatidan
#: qidiradi, ya'ni da'vodagi id konvertda bo'lmasa javobdan tushib qolardi.
SYNTH_EVIDENCE_LINES = 150

#: (g) narx qopqoqlari — kodda, `.env` bekor qilmaydi (bu xavfsizlik chegarasi).
COST_CEILING: dict[str, float] = {"fast": 0.01, "deep": 0.10, "exhaustive": 0.50}
EXHAUSTIVE_WALL_S = 600.0

#: `coverage.stopped_by` qiymatlari — UI shu ro'yxatga qarab matn tanlaydi.
STOP_REASONS = ("complete", "yield", "ceiling", "saturation", "wall", "estimate_only")

#: To'xtash sabablarining **ustunligi**. Bosqichlar o'z sababini alohida
#: yozadi (`pull_stopped_by` / `filter_stopped_by`), umumiy sabab esa
#: shulardan eng "og'iri". Nega kerak: `filter_batched` hovuzni tugatib
#: `complete` deb yozardi va pull byudjeti qo'ygan `wall` ni **bekor
#: qilardi** — qamrovda "reja to'liq skanlandi" chiqardi, holbuki bucket'larning
#: yarmi umuman qaralmagan edi.
STOP_RANK: dict[str, int] = {
    "complete": 0,
    "yield": 1,
    "saturation": 2,
    "wall": 3,
    "ceiling": 4,
    "estimate_only": 5,
}


def combine_stops(*reasons: str) -> str:
    """Bir necha bosqich sababidan eng ustunini tanlaydi (`STOP_RANK`)."""
    best = "complete"
    for reason in reasons:
        if reason and STOP_RANK.get(reason, 0) > STOP_RANK.get(best, 0):
            best = reason
    return best


EVENT_MIN_INTERVAL_S = 1.0  # `pull`/`filter` hodisalari shundan tez-tez kelmaydi

#: (h) baho kalibratsiyasi — jonli yugurishlardan.
#: Baho ilgari **har doim** `FILTER_MAX` ga to'yinardi (`est_messages` =
#: bucket x k), ya'ni 5 ta bucket ham, 200 ta ham bir xil 0.1124 ni ko'rsatardi
#: va foydalanuvchi "nima to'layapman" degan savolga javob olmasdi.
#: `EST_PULL_YIELD` — bucket x k dan dedup va tirsakdan keyin hovuzda
#: qoladigan ulush; `EST_KEEP_RATE` — filtrdan o'tish ulushi (o'lchangan
#: 0.05-0.11 oralig'ining o'rtasi). Baho baribir yuqoridan: pastki chegara
#: `ELBOW_MIN`, yuqorigisi `FILTER_MAX`.
EST_PULL_YIELD = 0.35
EST_KEEP_RATE = 0.08


def elbow_hi_for(mode: str) -> int:
    """Elbow kesishning yuqori chegarasi — rejimga bog'liq.

    `deep` uchun `ELBOW_MAX` (3000). `exhaustive` uchun esa u
    `FILTER_MAX["exhaustive"]` gacha ko'tariladi: aks holda rejimning
    kattaroq byudjeti **hech qachon** ishlatilmasdi — hovuz 3000 da
    kesilib, `FILTER_MAX` 10 000 o'lik konstanta bo'lib qolardi.
    """
    return min(POOL_MAX, max(ELBOW_MAX, FILTER_MAX.get(mode, ELBOW_MAX)))


def bucket_max_for(mode: str) -> int:
    """Rejim bo'yicha bucket qopqog'i — `BUCKET_MAX_DEEP` izohidagi o'lchovdan."""
    return BUCKET_MAX if mode == "exhaustive" else BUCKET_MAX_DEEP


_JSON_RE = re.compile(r"\{.*\}", re.S)
#: Da'vo sitatasi — `#` siz, `to_evidence` yasagan token bilan bir xil naqsh.
_CITE_BODY_RE = re.compile(r"^(?:c\d{1,12}m\d{1,12}|x\d{1,12})$")

#: Baho uchun taxminiy narx (USD/1M) — model jadvalda topilmasa. Kam
#: baholashdan ko'ra ortiq baholagan yaxshi: qopqoq ishlab tursin.
FALLBACK_PRICE: tuple[float, float] = (0.30, 2.50)


# ─── turlar ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Bucket:
    """Korpusning bitta bo'lagi: bitta chat x bitta ISO-hafta (yoki oy)."""

    chat_id: int
    chat_title: str
    since: datetime
    until: datetime
    score: float = 0.0
    est_messages: int = 0
    source: str = "node"  # node | month
    label: str = ""  # `2026-W12` yoki `2026-03`

    @property
    def key(self) -> str:
        return f"c{self.chat_id}|{self.label}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "label": self.label,
            "score": round(self.score, 4),
            "est_messages": self.est_messages,
            "source": self.source,
        }


@dataclass(slots=True)
class Estimate:
    """Skan boshlanishidan **oldingi** baho — bucket sanog'i x k x narx jadvali."""

    mode: str
    buckets: int
    est_messages: int
    est_cost_usd: float  # **skan** narxi — aynan `COST_CEILING` boshqaradigan qism
    ceiling_usd: float
    #: Skan + sintez. Qopqoqqa kirmaydi (sintez rejimga bog'liq emas), lekin
    #: foydalanuvchi to'laydigan raqam shu — UI ikkalasini ko'rsatadi.
    est_total_usd: float = 0.0
    weeks: int = 0  # reja qamragan alohida (ISO-hafta | oy) yorliqlari
    range_from: datetime | None = None
    range_to: datetime | None = None
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "buckets": self.buckets,
            "est_messages": self.est_messages,
            "est_cost_usd": round(self.est_cost_usd, 4),
            "est_total_usd": round(self.est_total_usd, 4),
            "ceiling_usd": self.ceiling_usd,
            "weeks": self.weeks,
            "range": {
                "from": self.range_from.isoformat() if self.range_from else None,
                "to": self.range_to.isoformat() if self.range_to else None,
            },
        }


@dataclass(slots=True)
class Claim:
    """Bitta bo'lakdan ajratilgan atomar da'vo. `cites` — **tekshirilgan** token'lar."""

    text: str
    cites: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "cites": list(self.cites),
            "confidence": round(self.confidence, 2),
        }


@dataclass(slots=True)
class ScanStats:
    """Halol qamrov raqamlari — `Coverage` va `scan_runs` shulardan to'ladi."""

    buckets_planned: int = 0
    buckets_scanned: int = 0
    #: Tirsakdan **oldingi** birlashtirilgan hovuz — "nechta nomzod ko'rildi"
    #: degan savolning halol javobi (`scan_runs.candidates` shundan yoziladi).
    pulled: int = 0
    candidates: int = 0  # dedupdan keyingi hovuz
    filtered: int = 0  # modelga (yoki keshga) ko'rsatilgani
    kept: int = 0  # `FILTER_KEEP_MIN` dan o'tgani
    cache_hits: int = 0
    evidence: int = 0
    claims: int = 0
    contradictions: int = 0
    chunks: int = 0
    quota_trimmed: int = 0
    #: Bosqich sabablari alohida saqlanadi — keyingi bosqich oldingisini
    #: `complete` bilan **bekor qila olmasin** (`combine_stops`).
    pull_stopped_by: str = ""
    filter_stopped_by: str = ""
    stopped_by: str = "complete"
    truncated: bool = False
    scan_cost_usd: float = 0.0
    est_cost_usd: float = 0.0
    est_total_usd: float = 0.0

    def note_stop(self, stage: str, reason: str) -> None:
        """Bosqich sababini yozadi va umumiysini **qayta hisoblaydi**."""
        if not reason:
            return
        if stage == "pull":
            self.pull_stopped_by = combine_stops(self.pull_stopped_by, reason)
        elif stage == "filter":
            self.filter_stopped_by = combine_stops(self.filter_stopped_by, reason)
        self.stopped_by = combine_stops(self.pull_stopped_by, self.filter_stopped_by, reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "buckets_planned": self.buckets_planned,
            "buckets_scanned": self.buckets_scanned,
            "pulled": self.pulled,
            "candidates": self.candidates,
            "filtered": self.filtered,
            "kept": self.kept,
            "cache_hits": self.cache_hits,
            "evidence": self.evidence,
            "claims": self.claims,
            "contradictions": self.contradictions,
            "chunks": self.chunks,
            "quota_trimmed": self.quota_trimmed,
            "pull_stopped_by": self.pull_stopped_by,
            "filter_stopped_by": self.filter_stopped_by,
            "stopped_by": self.stopped_by,
            "truncated": self.truncated,
            "scan_cost_usd": round(self.scan_cost_usd, 6),
        }


@dataclass(slots=True)
class ScaleOutcome:
    """Kaskad natijasi — `deep_search` uni sintezga uzatadi."""

    evidence: list[S.Hit] = field(default_factory=list)
    clusters: list[C.Cluster] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    contradictions: list[Claim] = field(default_factory=list)
    timeline: list[str] = field(default_factory=list)
    pool: list[S.Hit] = field(default_factory=list)
    buckets: list[Bucket] = field(default_factory=list)
    stats: ScanStats = field(default_factory=ScanStats)
    estimate: Estimate | None = None


class _Budget:
    """Narx qopqog'i + devor soati — bitta joyda, bosqichlar shundan so'raydi."""

    __slots__ = ("ceiling", "deadline", "meter", "start_cost")

    def __init__(self, meter: DS._Meter, *, ceiling: float, deadline: float) -> None:
        self.meter = meter
        self.ceiling = ceiling
        self.deadline = deadline  # `time.monotonic()` fazasida
        self.start_cost = meter.cost_usd

    @property
    def spent(self) -> float:
        return max(0.0, self.meter.cost_usd - self.start_cost)

    def over_cost(self) -> bool:
        """Qopqoq **skan** sarfini o'lchaydi, butun yugurishnikini emas.

        Sabab: `estimate()` ham aynan skan narxini qaytaradi va ikkalasi
        bitta hodisada yonma-yon ko'rsatiladi (`est_cost_usd` va
        `ceiling_usd`). Reja/HyDE/embedding va sintez narxi rejim tanloviga
        bog'liq emas, ya'ni ularni qopqoqqa qo'shish foydalanuvchiga
        "skan qimmat" deb yolg'on ko'rsatardi.
        """
        return self.spent >= self.ceiling

    def over_time(self) -> bool:
        return time.monotonic() >= self.deadline

    def left_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def stop_reason(self) -> str:
        if self.over_cost():
            return "ceiling"
        if self.over_time():
            return "wall"
        return ""


class _Throttle:
    """Hodisa oqimini sekinlashtiradi (`pull`/`filter` — ≤1 hodisa/soniya)."""

    __slots__ = ("_last", "interval")

    def __init__(self, interval: float = EVENT_MIN_INTERVAL_S) -> None:
        self.interval = interval
        self._last = 0.0

    def ready(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if force or now - self._last >= self.interval:
            self._last = now
            return True
        return False


# ─── umumiy yordamchilar ─────────────────────────────────────────────────────


def question_hash(question: str) -> str:
    """`md5(uz_fold(savol))` — kesh kaliti.

    `uz_fold` imlo/apostrof/kirill farqini yo'qotadi, shuning uchun
    "to'lov muddati" va "тўлов муддати" bitta keshga tushadi. MD5 bu yerda
    **kriptografik emas** — faqat qisqa, barqaror kalit.
    """
    folded = uz_fold((question or "").strip())
    return hashlib.md5(folded.encode("utf-8"), usedforsecurity=False).hexdigest()


def iso_label(dt: datetime) -> str:
    """`2026-W12` — ISO yil va hafta (kalendar yili bilan farq qilishi mumkin)."""
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def month_label(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def _month_start(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_month(dt: datetime) -> datetime:
    return _month_start(_month_start(dt) + timedelta(days=32))


def _json_obj(raw: str) -> dict[str, Any]:
    m = _JSON_RE.search(raw or "")
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def hit_token(hit: S.Hit) -> str:
    """`c<chat>m<msg>` — `deep_search.to_evidence` yasaydigan token bilan bir xil."""
    return f"c{hit.chat_id}m{hit.tg_msg_id}" if hit.chat_id and hit.tg_msg_id else ""


# ─── (a) bucket rejasi ───────────────────────────────────────────────────────


async def plan_buckets(
    session: AsyncSession,
    account_id: int,
    *,
    plan: DS.DeepPlan,
    vectors: Sequence[Sequence[float]],
    chat_titles: dict[int, str] | None = None,
    limit: int = BUCKET_MAX,
    mode: str = "deep",
) -> list[Bucket]:
    """Korpusni chat x ISO-hafta bo'laklariga bo'ladi va savolga yaqinligi bo'yicha tartiblaydi.

    Manba — `retrieval_nodes` (kun/hafta tugunlari): ular allaqachon
    embedding bilan, ya'ni "qaysi haftada shu mavzu gapirilgan" degan savolga
    **bitta** vektor so'rovi bilan javob beradi. Har savol vektori (standalone
    savol + subquery'lar) uchun alohida so'rov yuguradi, bucket esa ularning
    **maksimumini** oladi: bitta subquery topgan hafta boshqasi topmagani
    uchun pastga tushmasin.

    Tuguni yo'q chatlar (hali kun tuguni qurilmagan yoki juda kichik) uchun
    zaxira yo'l — `messages` bo'yicha `(chat, oy)` sanoq bucket'lari. Ular
    bazaviy ball oladi, ya'ni mos topilgan hafta tugunlaridan pastda turadi,
    lekin **yo'qolmaydi**.

    `mode="exhaustive"` va oynasiz (`plan.since is None`) savolda zaxira grid
    **hamma** nishon chat uchun quriladi — tuguni borlari uchun ham — va
    sinxronlangan butun oraliqni (`min(published_at)` … hozir) qoplaydi.
    Nega: jonli yugurishda tugunlar faqat oxirgi haftalarda bor edi, ya'ni
    "burchakdan burchakka" degan rejim 5-7 bucket / 1 hafta bilan
    chegaralanardi. Grid tugun bucket'lari bilan **birlashtiriladi** (ular
    ustun: bir xil kalitda tugun bali saqlanadi).

    Returns:
        `score` bo'yicha kamayish tartibidagi ≤`limit` ta `Bucket`.
    """
    titles = dict(chat_titles or {})
    plan_chats = set(plan.chats or [])
    by_key: dict[tuple[int, str], Bucket] = {}

    conds: list[Any] = [
        RetrievalNode.account_id == account_id,
        RetrievalNode.level.in_(list(BUCKET_NODE_LEVELS)),
    ]
    if plan.chats:
        conds.append(RetrievalNode.chat_id.in_(list(plan.chats)))
    if plan.since is not None:
        conds.append(RetrievalNode.ends_at >= plan.since)
    if plan.until is not None:
        conds.append(RetrievalNode.starts_at < plan.until)

    for vec in vectors:
        if not vec:
            continue
        dist = RetrievalNodeEmbedding.vector.cosine_distance(list(vec)).label("dist")
        stmt = (
            select(
                RetrievalNode.chat_id,
                RetrievalNode.starts_at,
                RetrievalNode.msg_count,
                dist,
            )
            .join(RetrievalNodeEmbedding, RetrievalNodeEmbedding.node_id == RetrievalNode.id)
            .where(*conds, *S._embed_model(RetrievalNodeEmbedding.model))
            .order_by(dist)
            .limit(BUCKET_NODE_FETCH)
        )
        await S.tune_hnsw(session)
        for chat_id, starts_at, msg_count, distance in (await session.execute(stmt)).all():
            when = starts_at if starts_at.tzinfo else starts_at.replace(tzinfo=UTC)
            start = N.week_start(when)
            key = (int(chat_id), iso_label(start))
            score = 1.0 - float(distance or 0.0)
            found = by_key.get(key)
            if found is None:
                by_key[key] = Bucket(
                    chat_id=int(chat_id),
                    chat_title=titles.get(int(chat_id), ""),
                    since=start,
                    until=start + timedelta(days=7),
                    score=score,
                    est_messages=int(msg_count or 0),
                    source="node",
                    label=iso_label(start),
                )
            else:
                found.score = max(found.score, score)
                found.est_messages = max(found.est_messages, int(msg_count or 0))

    full_grid = mode == "exhaustive" and plan.since is None
    covered = set() if full_grid else {cid for cid, _ in by_key}
    fallback = await _month_buckets(
        session,
        account_id,
        plan=plan,
        exclude=covered,
        titles=titles,
        limit=BUCKET_FALLBACK_CHATS_FULL if full_grid else BUCKET_FALLBACK_CHATS,
    )
    for bucket in fallback:
        by_key.setdefault((bucket.chat_id, bucket.label), bucket)

    out = list(by_key.values())
    for bucket in out:
        if bucket.chat_id in plan_chats:
            bucket.score += PLAN_CHAT_BONUS
        if not bucket.chat_title:
            bucket.chat_title = titles.get(bucket.chat_id, "")
    out.sort(key=lambda b: (-b.score, b.chat_id, b.label))
    log.debug(
        "scale.buckets",
        account_id=account_id,
        mode=mode,
        full_grid=full_grid,
        node=sum(1 for b in out if b.source == "node"),
        month=sum(1 for b in out if b.source == "month"),
        kept=min(len(out), limit),
    )
    return out[:limit]


async def _month_buckets(
    session: AsyncSession,
    account_id: int,
    *,
    plan: DS.DeepPlan,
    exclude: set[int],
    titles: dict[int, str],
    limit: int,
) -> list[Bucket]:
    """Tuguni yo'q chatlar uchun `(chat, oy)` bucket'lari — xabar sanog'idan.

    Nega oy, hafta emas: tugun yo'q degani odatda chat kichik yoki yangi
    sinxronlangan. Hafta darajasida bo'lish ularni yuzlab bo'sh bucket'ga
    aylantirardi va `BUCKET_MAX` ni haqiqiy nomzodsiz to'ldirardi.
    """
    conds: list[Any] = [Chat.account_id == account_id, *S._alive()]
    if plan.chats:
        conds.append(Message.chat_id.in_(list(plan.chats)))
    if plan.since is not None:
        conds.append(Message.published_at >= plan.since)
    if plan.until is not None:
        conds.append(Message.published_at < plan.until)

    month = func.date_trunc("month", Message.published_at).label("m")
    stmt = (
        select(Message.chat_id, month, func.count().label("n"))
        .join(Chat, Chat.id == Message.chat_id)
        .where(*conds)
        .group_by(Message.chat_id, month)
        .order_by(func.count().desc())
        .limit(limit * 12)
    )
    try:
        rows = (await session.execute(stmt)).all()
    except Exception as exc:  # zaxira yo'l — u yo'q bo'lsa ham kaskad ishlaydi
        log.warning("scale.month_buckets_failed", error=str(exc)[:200])
        return []

    out: list[Bucket] = []
    seen_chats: set[int] = set()
    for chat_id, start, n in rows:
        cid = int(chat_id)
        if cid in exclude:
            continue
        if cid not in seen_chats and len(seen_chats) >= limit:
            continue
        seen_chats.add(cid)
        when = start if start.tzinfo else start.replace(tzinfo=UTC)
        out.append(
            Bucket(
                chat_id=cid,
                chat_title=titles.get(cid, ""),
                since=when,
                until=_next_month(when),
                score=FALLBACK_SCORE,
                est_messages=int(n or 0),
                source="month",
                label=month_label(when),
            )
        )
    return out


# ─── (b) bucket ichidan nomzod tortish ───────────────────────────────────────


@dataclass(slots=True)
class PullSpec:
    """Har bucket uchun **bir xil** kirish: atamalar bir marta hisoblanadi."""

    account_id: int
    uz_terms: list[str] = field(default_factory=list)
    ru_terms: list[str] = field(default_factory=list)
    qvec: list[float] | None = None
    k: int = PER_BUCKET_K


def build_pull_spec(
    account_id: int, *, question: str, subqueries: Sequence[str], qvec: Sequence[float] | None
) -> PullSpec:
    """Savol + subquery'lardan leksik atamalar (kanal boshiga bir marta)."""
    terms: list[str] = []
    for text in (question, *subqueries):
        for term in S.keywords(text or ""):
            if term.lower() not in {x.lower() for x in terms}:
                terms.append(term)
    ru = [t for t in terms if S._CYRILLIC_RE.search(t)]
    return PullSpec(
        account_id=account_id,
        uz_terms=terms[:16],
        ru_terms=ru[:16],
        qvec=list(qvec) if qvec else None,
    )


async def pull_bucket(session: AsyncSession, bucket: Bucket, spec: PullSpec) -> list[S.Hit]:
    """Bitta bucket ichida 4 kanalli RRF (tsv_uz · tsv_ru · trigram · vektor).

    Tugun kanali bu yerda **yo'q**: bucket'ning o'zi tugun vektoridan
    tug'ilgan, ya'ni tugunni ikkinchi marta so'rash bir xil signalni
    ikki marta sanardi.
    """
    conds = S._scope(spec.account_id, [bucket.chat_id], bucket.since, bucket.until)
    wide = spec.k * 2

    uz: list[S.Hit] = []
    uz_col = getattr(Message, "tsv_uz", None)
    folded = S.fold_terms(spec.uz_terms)
    if uz_col is not None and folded:
        uz = await S._tsv_multi(
            session,
            uz_col,
            config="simple",
            tsq=S._tsquery(folded, prefix=True, op=" | "),
            conds=conds,
            limit=wide,
            source="tsv_uz",
        )

    ru: list[S.Hit] = []
    ru_col = getattr(Message, "tsv_ru", None)
    ru_clean = [t for t in (S._tsq_safe(x.lower()) for x in spec.ru_terms) if len(t) >= 2]
    if ru_col is not None and ru_clean:
        ru = await S._tsv_multi(
            session,
            ru_col,
            config="russian",
            tsq=S._tsquery(ru_clean, prefix=False, op=" | "),
            conds=conds,
            limit=wide,
            source="tsv_ru",
        )

    tri = await S._trigram_multi(session, spec.uz_terms or spec.ru_terms, conds=conds, limit=wide)
    vec = await S._vector_multi(session, spec.qvec, conds=conds, limit=wide) if spec.qvec else []

    fused = S.rrf(uz, ru, tri, vec, weights=S.DEFAULT_RRF_WEIGHTS)
    return fused[: spec.k]


def elbow_cut(
    scores: Sequence[float],
    *,
    lo: int = ELBOW_MIN,
    hi: int = ELBOW_MAX,
    keep_min_frac: float = ELBOW_KEEP_MIN_FRAC,
) -> int:
    """Kamayuvchi ballar qatoridagi eng katta sakrash — `[lo, hi]` oralig'ida.

    Nega qopqoq bilan: RRF ballari silliq kamayadi, ya'ni "tabiiy" tirsak
    ba'zan 5-chi, ba'zan 8000-chi o'rinda bo'ladi. Pastki chegara javobni
    ochlikdan saqlaydi, yuqorigisi — filtr narxidan.

    `keep_min_frac` — tirsak hovuzning ko'pi bilan `1 - frac` qismini
    tashlaydi. Jonli o'lchovda tirsak 2058 nomzoddan 239 tasini qoldirgan
    (88% tashlangan) va dalil 1-20 ta chiqqan: "sakrash" u yerda shovqin edi.
    Haqiqiy shift — `FILTER_MAX`, tirsak esa faqat quyruqni kesadi.
    """
    n = len(scores)
    if n <= lo:
        return n
    floor_at = max(lo, math.ceil(n * max(0.0, min(1.0, keep_min_frac))))
    top = min(hi, n)
    if floor_at >= top:
        return top
    best_at, best_drop = top, -1.0
    for i in range(floor_at, top):
        drop = float(scores[i - 1]) - float(scores[i])
        if drop > best_drop:
            best_drop, best_at = drop, i
    return best_at


async def pull_candidates(
    session_factory: DS.SessionFactory,
    *,
    buckets: Sequence[Bucket],
    spec: PullSpec,
    events: DS.EventSink | None = None,
    budget_s: float | None = None,
    concurrency: int = PULL_CONCURRENCY,
    pool_max: int = POOL_MAX,
    elbow_hi: int = ELBOW_MAX,
    stats: ScanStats | None = None,
) -> tuple[list[S.Hit], dict[int, str], int]:
    """Bucket'larni parallel aylanib chiqadi → birlashtirilgan hovuz + elbow kesish.

    Har vazifa **o'z sessiyasi** bilan ishlaydi (bitta `AsyncSession`
    parallelga xavfsiz emas), shift esa semafor bilan. Byudjet tugasa
    qolgan bucket'lar bekor qilinadi — **tugaganlari saqlanadi**
    (`fan_out` bilan bir sabab: `gather` bekor qilinganda tugaganini ham
    tashlab yuboradi).

    Returns:
        `(hovuz, message_id → bucket kaliti, skanlangan bucket soni)`.
    """
    if not buckets:
        return [], {}, 0
    sem = asyncio.Semaphore(max(1, concurrency))
    throttle = _Throttle()
    done = 0
    seen = 0  # shu paytgacha tortilgan xom nomzodlar (dedupdan oldin)
    started = time.monotonic()

    async def one(bucket: Bucket) -> tuple[Bucket, list[S.Hit]]:
        nonlocal done, seen
        async with sem, session_factory() as session:
            try:
                hits = await pull_bucket(session, bucket, spec)
            except Exception as exc:
                log.warning("scale.pull_failed", bucket=bucket.key, error=str(exc)[:200])
                hits = []
        done += 1
        seen += len(hits)
        if events is not None and throttle.ready(force=done == len(buckets)):
            await DS._emit(
                events,
                "pull",
                bucket_i=done,
                buckets=len(buckets),
                candidates=seen,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        return bucket, hits

    tasks = [asyncio.ensure_future(one(b)) for b in buckets]
    _, pending = await asyncio.wait(tasks, timeout=budget_s)
    if pending:
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        log.warning("scale.pull_timeout", pending=len(pending), total=len(tasks))

    lists: list[list[S.Hit]] = []
    bucket_of: dict[int, str] = {}
    scanned = 0
    for task in tasks:
        if task.cancelled() or task.exception() is not None:
            continue
        bucket, hits = task.result()
        scanned += 1
        if not hits:
            continue
        for h in hits:
            bucket_of.setdefault(h.message_id, bucket.key)
        lists.append(hits)

    fused = S.rrf(*lists)[:pool_max] if lists else []
    cut = elbow_cut([h.score for h in fused], hi=elbow_hi)
    out = fused[:cut]
    if stats is not None:
        # Tirsakdan **oldingi** hovuz: qamrov hisobotida "nechta nomzod
        # ko'rildi" degan raqam shu bo'lishi kerak, kesilganidan keyingisi emas.
        stats.pulled = len(fused)
    if events is not None:
        await DS._emit(
            events,
            "pull",
            bucket_i=scanned,
            buckets=len(buckets),
            candidates=len(out),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
    log.info(
        "scale.pull",
        buckets=len(buckets),
        scanned=scanned,
        pool=len(fused),
        elbow=cut,
        kept=len(out),
        elbow_hi=elbow_hi,
        elapsed_s=round(time.monotonic() - started, 1),
    )
    return out, bucket_of, scanned


def pull_budget_for(left_s: float, *, reserve_s: float = PULL_RESERVE_S) -> float:
    """Pull bosqichiga ajratiladigan vaqt — derivatsiya bitta joyda.

    `qolgan vaqt - zaxira`, pastdan `PULL_MIN_S` bilan. Zaxira pull'dan
    keyingi butun ishni (dedup, filtr to'lqinlari, tanlov, map/reduce,
    sintez) qoplashi kerak; jonli o'lchovda u eng yomon holatda 30-40 s.
    """
    return max(PULL_MIN_S, left_s - reserve_s)


# ─── (c) dedup — O(n) ────────────────────────────────────────────────────────


def dedup_banded(hits: list[S.Hit], *, max_hamming: int = C.SIMHASH_MAX_HAMMING) -> list[S.Hit]:
    """Simhash dedup **band indeksi** bilan — kutilgan murakkablik O(n).

    64-bitli imzo 4 ta 16-bitli bandga bo'linadi. Ikki imzo orasidagi
    hamming masofa ≤3 bo'lsa, farq bandlarning ko'pi bilan uchtasiga
    tushadi, ya'ni **kamida bitta** band aynan bir xil bo'ladi (kaptarxona
    tamoyili). Shuning uchun har imzo faqat o'zi bilan bitta bandni
    bo'lishgan nomzodlar bilan solishtiriladi — `cluster.dedup_simhash`
    dagi "hamma bilan" o'rniga.
    """
    from app.services.textnorm import hamming64, simhash64

    band_bits = 64 // SIMHASH_BANDS
    mask = (1 << band_bits) - 1
    index: dict[tuple[int, int], list[tuple[int, S.Hit]]] = {}
    kept: list[S.Hit] = []
    for h in hits:
        body = C.normalize(h.text)
        if len(body) < 24:  # qisqa matnda simhash shovqin beradi
            kept.append(h)
            continue
        sig = simhash64(body)
        bands = [(b, (sig >> (b * band_bits)) & mask) for b in range(SIMHASH_BANDS)]
        dup: S.Hit | None = None
        for band in bands:
            for other_sig, other in index.get(band, ()):
                if hamming64(other_sig, sig) <= max_hamming:
                    dup = other
                    break
            if dup is not None:
                break
        if dup is None:
            kept.append(h)
            for band in bands:
                chunk = index.setdefault(band, [])
                if len(chunk) < BAND_BUCKET_MAX:
                    chunk.append((sig, h))
        else:
            C._absorb(dup, h)
    return kept


def dedup_fast(hits: list[S.Hit]) -> list[S.Hit]:
    """Hovuz uchun dedup: sha1 + banded simhash. **Kosinus yo'q** (O(n²))."""
    return dedup_banded(C.dedup_exact(hits))


# ─── (d) filtr — Flash-Lite, partiyalarda ────────────────────────────────────


def filter_line(index: int, hit: S.Hit, *, max_chars: int = FILTER_LINE_CHARS) -> str:
    """`#i [chat | sana | kim] qisqartirilgan matn` — konvert ichidagi bitta satr."""
    when = hit.date.strftime("%Y-%m-%d") if hit.date else "?"
    head = f"{U.attr(hit.chat_title)} | {when} | {U.attr(hit.sender)}"
    body = U.neutralize(S.compact_text(hit.text or "", max_chars=max_chars))
    return f"#{index} [{head}] {body}".strip()


def build_filter_envelope(question: str, batch: Sequence[tuple[int, S.Hit]]) -> str:
    """Bitta konvert, nomzodlar faqat raqam bilan tanaladi (invariant 3)."""
    lines = "\n".join(filter_line(i, h) for i, h in batch)
    return (
        f"<question>{U.neutralize(question)}</question>\n"
        f'<untrusted_data kind="candidates" count="{len(batch)}">\n'
        f"{lines}\n</untrusted_data>"
    )


def parse_keep(raw: str, *, allowed: Sequence[int]) -> dict[int, int]:
    """`{"keep": [[i, score], …]}` → `{indeks: ball}`; noma'lum indeks tashlanadi."""
    data = _json_obj(raw)
    items = data.get("keep")
    if not isinstance(items, list):
        return {}
    ok = set(allowed)
    out: dict[int, int] = {}
    for item in items:
        idx: int | None
        score: int | None
        if isinstance(item, list | tuple) and len(item) >= 2:
            idx, score = _as_int(item[0]), _as_int(item[1])
        elif isinstance(item, dict):
            idx, score = _as_int(item.get("i") or item.get("index")), _as_int(item.get("score"))
        else:
            idx, score = _as_int(item), 3
        if idx is None or idx not in ok or score is None:
            continue
        out[idx] = max(0, min(3, score))
    return out


def batches(pool: Sequence[S.Hit], *, size: int = FILTER_BATCH) -> list[list[tuple[int, S.Hit]]]:
    """Global indeksni saqlab partiyalarga bo'ladi (indeks = hovuzdagi o'rin)."""
    indexed = list(enumerate(pool))
    return [indexed[i : i + size] for i in range(0, len(indexed), size)]


def bucket_coverage(seen: set[str], total: set[str]) -> float:
    """Skanlangan bucket'lar ulushi — erta to'xtash sharti (b) uchun."""
    return (len(seen & total) / len(total)) if total else 1.0


def should_early_stop(
    keep_rates: Sequence[float],
    coverage: float,
    *,
    window: int = EARLY_STOP_BATCHES,
    min_rate: float = EARLY_STOP_KEEP_RATE,
    min_coverage: float = EARLY_STOP_COVERAGE,
) -> bool:
    """Oxirgi `window` partiya deyarli hech narsa bermadi **va** korpus qaraldi.

    Ikkala shart ham kerak: past keep-rate o'zi "savol tugadi" degani emas
    (nomzodlar bucket bo'yicha tartiblangan, oxirgi bucket'lar zaif), bucket
    qamrovi esa "hammasini ko'rdik"ni **kodda** isbotlaydi.
    """
    if len(keep_rates) < window:
        return False
    tail = list(keep_rates)[-window:]
    return (sum(tail) / window) < min_rate and coverage >= min_coverage


async def cache_lookup(
    session: AsyncSession, *, q_hash: str, pairs: Sequence[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    """`relevance_cache` dan tayyor ballar. Jadval yo'q bo'lsa — bo'sh lug'at.

    Model **kech** import qilinadi: kesh ixtiyoriy qatlam (migratsiya 0020
    hali yugurmagan bo'lsa ham kaskad ishlashi kerak), shuning uchun u
    modulning import bog'liqligi bo'lmasin.
    """
    if not pairs:
        return {}
    from app.db.models import RelevanceCache

    chat_ids = sorted({c for c, _ in pairs})
    wanted = set(pairs)
    try:
        stmt = select(RelevanceCache.chat_id, RelevanceCache.tg_msg_id, RelevanceCache.score).where(
            RelevanceCache.q_hash == q_hash, RelevanceCache.chat_id.in_(chat_ids)
        )
        rows = (await session.execute(stmt)).all()
    except Exception as exc:  # migratsiya hali yugurmagan — kesh ixtiyoriy
        log.warning("scale.cache_read_failed", error=str(exc)[:200])
        return {}
    out: dict[tuple[int, int], int] = {}
    for chat_id, tg_msg_id, score in rows:
        key = (int(chat_id), int(tg_msg_id))
        if key in wanted:
            out[key] = int(score or 0)
    return out


async def cache_store(
    session: AsyncSession,
    *,
    account_id: int,
    q_hash: str,
    model: str,
    scores: Sequence[tuple[int, int, int]],
) -> None:
    """Yangi ballarni keshga yozadi (`ON CONFLICT DO NOTHING` — qayta yozilmaydi)."""
    if not scores:
        return
    from app.db.models import RelevanceCache

    rows = [
        {
            "account_id": account_id,
            "q_hash": q_hash,
            "chat_id": int(chat_id),
            "tg_msg_id": int(tg_msg_id),
            "score": int(score),
            "model": (model or "")[:64],
        }
        for chat_id, tg_msg_id, score in scores
        if chat_id and tg_msg_id
    ]
    if not rows:
        return
    try:
        stmt = pg_insert(RelevanceCache).values(rows).on_conflict_do_nothing()
        await session.execute(stmt)
    except Exception as exc:
        log.warning("scale.cache_write_failed", error=str(exc)[:200])


async def filter_batched(
    session_factory: DS.SessionFactory,
    *,
    account_id: int,
    question: str,
    pool: Sequence[S.Hit],
    bucket_of: dict[int, str],
    meter: DS._Meter,
    budget: _Budget,
    taint: DS._Taint,
    events: DS.EventSink | None = None,
    mode: str = "deep",
    stats: ScanStats | None = None,
) -> list[tuple[S.Hit, int]]:
    """Hovuzni arzon model bilan elakdan o'tkazadi → `(hit, ball)` juftlari.

    To'rtta to'xtash sababi bor va hammasi **kodda**: narx qopqog'i
    (`ceiling`), devor soati (`wall`), erta to'yinish (`yield`) va hovuzning
    tugashi (`complete`). Model hech qachon "yetarli" deb ayta olmaydi.

    Kesh (`relevance_cache`) chaqiruvdan **oldin** so'raladi: bir xil savol
    bir xil xabarga bir xil ball beradi, ya'ni ikkinchi marta so'rash token
    sarflaydi-yu, natijani o'zgartirmaydi.
    """
    st = stats or ScanStats()
    limited = list(pool)[: FILTER_MAX.get(mode, FILTER_MAX["deep"])]
    if not limited:
        st.note_stop("filter", "complete")
        return []

    q_hash = question_hash(question)
    pairs = [(h.chat_id, h.tg_msg_id) for h in limited if h.chat_id and h.tg_msg_id]
    async with session_factory() as session:
        cached = await cache_lookup(session, q_hash=q_hash, pairs=pairs)

    scored: dict[int, int] = {}
    todo: list[S.Hit] = []
    for i, hit in enumerate(limited):
        found = cached.get((hit.chat_id, hit.tg_msg_id))
        if found is None:
            todo.append(hit)
        else:
            scored[i] = found
            st.cache_hits += 1
    index_of = {id(h): i for i, h in enumerate(limited)}

    all_buckets = {bucket_of.get(h.message_id, "") for h in limited} - {""}
    seen_buckets: set[str] = set()
    keep_rates: list[float] = []
    fresh: list[tuple[int, int, int]] = []
    throttle = _Throttle()
    groups = batches(todo)
    processed = 0
    model_name = ""
    stopped = ""

    async def run_batch(batch: list[tuple[int, S.Hit]]) -> dict[int, int]:
        nonlocal model_name
        prompt = build_filter_envelope(question, batch)
        taint.assert_clean("filter", prompt)
        try:
            res = await meter.chat(
                Task.ROUTE,
                [Msg.system(SCALE_FILTER_PROMPT), Msg.user(prompt)],
                json_mode=True,
                max_tokens=FILTER_MAX_TOKENS,
            )
        except LLMError as exc:
            log.warning("scale.filter_failed", error=str(exc)[:200])
            return {}
        model_name = res.model or model_name
        return parse_keep(res.text, allowed=[i for i, _ in batch])

    for wave_at in range(0, len(groups), FILTER_PARALLEL):
        reason = budget.stop_reason()
        if reason:
            stopped = reason
            break
        wave = groups[wave_at : wave_at + FILTER_PARALLEL]
        # Partiya indeksi hovuzdagi **global** o'rinni saqlashi kerak:
        # `todo` keshdan o'tganlarni tashlab ketgan, ya'ni lokal raqam
        # boshqa xabarga tegishli bo'lardi.
        renumbered = [[(index_of[id(h)], h) for _, h in batch] for batch in wave]
        results = await asyncio.gather(
            *(run_batch(batch) for batch in renumbered), return_exceptions=True
        )
        for batch, result in zip(renumbered, results, strict=True):
            processed += len(batch)
            if isinstance(result, BaseException):
                log.warning("scale.filter_batch_error", error=str(result)[:200])
                keep_rates.append(0.0)
                continue
            kept_here = 0
            for idx, score in result.items():
                scored[idx] = score
                if score >= FILTER_KEEP_MIN:
                    kept_here += 1
            for idx, hit in batch:
                seen_buckets.add(bucket_of.get(hit.message_id, ""))
                fresh.append((hit.chat_id, hit.tg_msg_id, scored.get(idx, 0)))
            keep_rates.append(kept_here / max(1, len(batch)))
        st.filtered = processed + st.cache_hits
        st.kept = sum(1 for v in scored.values() if v >= FILTER_KEEP_MIN)
        if events is not None and throttle.ready(force=wave_at + FILTER_PARALLEL >= len(groups)):
            await DS._emit(
                events,
                "filter",
                done=processed,
                total=len(todo),
                kept=st.kept,
                keep_rate=round(st.kept / max(1, st.filtered), 4),
                cost_usd=round(budget.spent, 6),
                cached=st.cache_hits,
            )
        if should_early_stop(keep_rates, bucket_coverage(seen_buckets, all_buckets)):
            stopped = "yield"
            log.info("scale.filter_early_stop", processed=processed, total=len(todo))
            break

    if not stopped and processed < len(todo):
        stopped = budget.stop_reason() or "wall"
    # Oldingi bosqich (pull) allaqachon to'xtash sababini yozgan bo'lishi
    # mumkin — uni "complete" bilan **bekor qilmaymiz** (`combine_stops`:
    # wall/ceiling yield/complete dan ustun turadi).
    st.note_stop("filter", stopped or "complete")
    st.truncated = st.truncated or bool(stopped) or len(pool) > len(limited)
    st.filtered = processed + st.cache_hits
    st.scan_cost_usd = round(budget.spent, 6)

    if fresh:
        async with session_factory() as session:
            await cache_store(
                session, account_id=account_id, q_hash=q_hash, model=model_name, scores=fresh
            )

    out = [(limited[i], score) for i, score in scored.items() if score >= FILTER_KEEP_MIN]
    out.sort(key=lambda pair: (-pair[1], -pair[0].score))
    st.kept = len(out)
    log.info(
        "scale.filter",
        pool=len(pool),
        limited=len(limited),
        cached=st.cache_hits,
        processed=processed,
        kept=len(out),
        pull_stopped_by=st.pull_stopped_by,
        filter_stopped_by=st.filter_stopped_by,
        stopped_by=st.stopped_by,
        cost_usd=st.scan_cost_usd,
    )
    return out


# ─── (e) tanlov — ball + chat/hafta kvotasi ──────────────────────────────────


def week_key(hit: S.Hit) -> str:
    return iso_label(hit.date) if hit.date else "?"


def select_evidence(
    scored: Sequence[tuple[S.Hit, int]],
    *,
    soft_max: int = SOFT_MAX_EVIDENCE,
    per_chat: int = PER_CHAT_QUOTA,
    per_week: int = PER_WEEK_QUOTA,
) -> tuple[list[S.Hit], int]:
    """Ball bo'yicha, lekin har chat/haftaga **kafolatlangan** ulush bilan.

    Nega kvota: sof ball tartibi bitta faol chatni butun byudjet bilan
    to'ldiradi va "qaerda gapirilgan" degan savolga javob yo'qoladi. Kvota
    avval band qilinadi, qolgan joy esa ball bo'yicha to'ldiriladi.

    Returns:
        `(tanlangan hitlar, kvota tufayli tashlab yuborilgan soni)`.
    """
    ordered = sorted(scored, key=lambda p: (-p[1], -p[0].score))
    reserved: list[S.Hit] = []
    picked: set[int] = set()
    per: dict[str, int] = {}

    def take(key: str, hit: S.Hit, cap: int) -> None:
        if hit.message_id in picked or per.get(key, 0) >= cap:
            return
        per[key] = per.get(key, 0) + 1
        picked.add(hit.message_id)
        reserved.append(hit)

    for hit, _score in ordered:
        take(f"c{hit.chat_id}", hit, per_chat)
    for hit, _score in ordered:
        take(f"w{week_key(hit)}", hit, per_week)

    trimmed = 0
    if len(reserved) > soft_max:
        rank = {h.message_id: i for i, (h, _s) in enumerate(ordered)}
        reserved.sort(key=lambda h: rank.get(h.message_id, 1 << 30))
        trimmed = len(reserved) - soft_max
        reserved = reserved[:soft_max]
        picked = {h.message_id for h in reserved}

    out = list(reserved)
    for hit, _score in ordered:
        if len(out) >= soft_max:
            break
        if hit.message_id not in picked:
            picked.add(hit.message_id)
            out.append(hit)

    rank = {h.message_id: i for i, (h, _s) in enumerate(ordered)}
    out.sort(key=lambda h: rank.get(h.message_id, 1 << 30))
    return out, trimmed


# ─── (f) map / reduce ────────────────────────────────────────────────────────


def map_chunks(
    evidence: Sequence[S.Hit],
    *,
    chunk_size: int = MAP_CHUNK_MSGS,
    max_chunks: int = MAP_MAX_CHUNKS,
    chunk_min: int = MAP_CHUNK_MIN,
    skip_below: int = MAP_SKIP_TOTAL,
) -> list[list[S.Hit]]:
    """Bo'laklar **bir jinsli**: har biri bitta chat (vaqt bo'yicha tartiblangan).

    Nega bitta chat: map prompti "shu suhbatda nima deyildi" degan savolga
    javob beradi. Aralash bo'lakda model ikki chatning ikki mavzusini bitta
    da'voga qo'shib yuborardi va sitata tekshiruvi buni tuta olmasdi —
    id'lar ikkalasidan ham haqiqiy.

    Nega haftalar birlashadi: bo'lish avval `(chat, hafta)` juftligi bo'yicha
    edi, ya'ni 10 qatorli tarqoq dalil **10 ta** map chaqirig'iga aylanardi
    (jonli yugurishdan). Endi qo'shni kichik bo'laklar bir chat ichida
    `chunk_min` gacha birlashtiriladi; `skip_below` dan kam dalilda map
    umuman yugurmaydi — u qatorlar sintezga baribir to'liq tushadi.
    """
    if len(evidence) <= max(0, skip_below):
        return []
    groups: dict[tuple[int, str], list[S.Hit]] = {}
    order: dict[tuple[int, str], int] = {}
    for i, hit in enumerate(evidence):
        key = (hit.chat_id, week_key(hit))
        groups.setdefault(key, []).append(hit)
        order.setdefault(key, i)
    raw: list[tuple[int, list[S.Hit]]] = []
    for key in sorted(groups, key=lambda k: order[k]):
        rows = sorted(groups[key], key=lambda h: h.date or datetime.min.replace(tzinfo=UTC))
        for i in range(0, len(rows), chunk_size):
            raw.append((key[0], rows[i : i + chunk_size]))
    merged: list[tuple[int, list[S.Hit]]] = []
    for chat_id, rows in raw:
        if (
            merged
            and merged[-1][0] == chat_id
            and len(merged[-1][1]) < chunk_min
            and len(merged[-1][1]) + len(rows) <= chunk_size
        ):
            merged[-1][1].extend(rows)
            continue
        merged.append((chat_id, list(rows)))
    return [rows for _chat_id, rows in merged][:max_chunks]


def build_map_envelope(question: str, chunk: Sequence[S.Hit]) -> tuple[str, set[str]]:
    """Bitta konvert + o'sha bo'lakda **ruxsat etilgan** token'lar to'plami."""
    lines: list[str] = []
    allowed: set[str] = set()
    for hit in chunk:
        token = hit_token(hit)
        if not token:
            continue
        allowed.add(token)
        when = hit.date.strftime("%Y-%m-%d %H:%M") if hit.date else "?"
        head = f"{U.attr(hit.chat_title)} | {when} | {U.attr(hit.sender)}"
        body = U.neutralize(S.compact_text(hit.text or "", max_chars=MAP_LINE_CHARS))
        lines.append(f"#{token} [{head}] {body}")
    body_all = "\n".join(lines) or "(no evidence)"
    envelope = (
        f"<question>{U.neutralize(question)}</question>\n"
        f'<untrusted_data source="telegram" kind="evidence" count="{len(allowed)}">\n'
        f"{body_all}\n</untrusted_data>"
    )
    return envelope, allowed


def parse_claims(
    raw: str, *, allowed: set[str] | None = None, max_claims: int = MAP_MAX_CLAIMS
) -> list[Claim]:
    """Model chiqishini da'volarga aylantiradi; **noma'lum sitata tashlanadi**.

    `allowed` berilganda (map bosqichi) sitata aynan shu bo'lakdagi
    id'lardan bo'lishi shart. Sitatasi qolmagan da'vo ham tashlanadi:
    tekshirib bo'lmaydigan da'vo javobga kirmaydi.
    """
    data = _json_obj(raw)
    items = data.get("claims")
    if not isinstance(items, list):
        return []
    out: list[Claim] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or len(text.strip()) < 3:
            continue
        cites: list[str] = []
        raw_cites = item.get("cites")
        if isinstance(raw_cites, list):
            for c in raw_cites:
                if not isinstance(c, str):
                    continue
                token = c.strip().lstrip("#")
                if not _CITE_BODY_RE.match(token):
                    continue
                if allowed is not None and token not in allowed:
                    continue
                if token not in cites:
                    cites.append(token)
        if not cites:
            continue
        conf = item.get("confidence")
        numeric = isinstance(conf, int | float) and not isinstance(conf, bool)
        confidence = float(conf) if numeric else 0.5  # type: ignore[arg-type]
        out.append(
            Claim(
                text=U.neutralize(text).strip()[:400],
                cites=cites[:REDUCE_CITES_MAX],
                confidence=max(0.0, min(1.0, confidence)),
            )
        )
        if len(out) >= max_claims:
            break
    return out


async def map_claims(
    *,
    question: str,
    chunks: Sequence[Sequence[S.Hit]],
    meter: DS._Meter,
    budget: _Budget,
    taint: DS._Taint,
    events: DS.EventSink | None = None,
    stats: ScanStats | None = None,
) -> list[Claim]:
    """Har bo'lakdan da'volar — parallel, sitatalar **map vaqtida** tekshiriladi."""
    st = stats or ScanStats()
    out: list[Claim] = []
    total = len(chunks)
    if not total:
        return out

    async def one(index: int, chunk: Sequence[S.Hit]) -> list[Claim]:
        envelope, allowed = build_map_envelope(question, chunk)
        taint.assert_clean("map", envelope)
        try:
            res = await meter.chat(
                Task.ROUTE,
                [Msg.system(SCALE_MAP_PROMPT), Msg.user(envelope)],
                json_mode=True,
                max_tokens=MAP_MAX_TOKENS,
            )
        except LLMError as exc:
            log.warning("scale.map_failed", chunk=index, error=str(exc)[:200])
            return []
        claims = parse_claims(res.text, allowed=allowed)
        await DS._emit(events, "map", chunk=index + 1, chunks=total, claims=len(claims))
        return claims

    for wave_at in range(0, total, MAP_PARALLEL):
        reason = budget.stop_reason()
        if reason:
            st.note_stop("map", reason)
            st.truncated = True
            log.info("scale.map_stopped", reason=reason, done=wave_at, total=total)
            break
        wave = list(range(wave_at, min(total, wave_at + MAP_PARALLEL)))
        results = await asyncio.gather(*(one(i, chunks[i]) for i in wave), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                log.warning("scale.map_chunk_error", error=str(result)[:200])
                continue
            out.extend(result)
        st.chunks = min(total, wave_at + MAP_PARALLEL)
    st.claims = len(out)
    st.scan_cost_usd = round(budget.spent, 6)
    return out


def reduce_task(intent: str, *, contradictions: int = 0) -> Task:
    """`compare`/`trend` yoki qarama-qarshilik → `DEEP`; aks holda `SEARCH`.

    Qoida **kodda**, chunki u narx qarori: DEEP chaqiruvi SEARCH'dan ~5x
    qimmat va uni faqat da'volar **orasidagi munosabat** muhim bo'lganda
    to'laymiz (farq, dinamika, ziddiyat). Oddiy `lookup` da reduce shunchaki
    ro'yxatni yig'adi.
    """
    if intent in DEEP_REDUCE_INTENTS or contradictions > 0:
        return Task.DEEP
    return Task.SEARCH


def build_reduce_envelope(question: str, claims: Sequence[Claim]) -> str:
    """Da'volar ham **ishonchsiz**: ular begona matndan model yasagan xulosa."""
    lines = [
        f"{i}. {U.neutralize(c.text)} [cites: {', '.join(c.cites)}] (conf {c.confidence:.2f})"
        for i, c in enumerate(claims, 1)
    ]
    body = "\n".join(lines) or "(no claims)"
    return (
        f"<question>{U.neutralize(question)}</question>\n"
        f'<untrusted_data kind="claims" count="{len(claims)}">\n'
        f"{body}\n</untrusted_data>"
    )


async def reduce_claims(
    *,
    question: str,
    claims: Sequence[Claim],
    intent: str,
    meter: DS._Meter,
    budget: _Budget,
    taint: DS._Taint,
    events: DS.EventSink | None = None,
    stats: ScanStats | None = None,
) -> tuple[list[Claim], list[Claim], list[str]]:
    """Da'volarni yagona dayjestga yig'adi → `(claims, contradictions, timeline)`.

    Birinchi urinish `reduce_task(intent)` bilan. Agar u `SEARCH` bo'lgan va
    natijada **ziddiyat** topilgan bo'lsa — byudjet ruxsat bersa bir marta
    `DEEP` bilan qaytariladi: ziddiyatni saralash aynan kuchli model ishi.
    """
    st = stats or ScanStats()
    if not claims:
        return [], [], []

    async def once(task: Task) -> tuple[list[Claim], list[Claim], list[str]]:
        envelope = build_reduce_envelope(question, claims)
        taint.assert_clean("reduce", envelope)
        try:
            res = await meter.chat(
                task,
                [Msg.system(SCALE_REDUCE_PROMPT), Msg.user(envelope)],
                json_mode=True,
                max_tokens=REDUCE_MAX_TOKENS,
            )
        except LLMError as exc:
            log.warning("scale.reduce_failed", error=str(exc)[:200])
            return list(claims)[:REDUCE_MAX_CLAIMS], [], []
        data = _json_obj(res.text)
        merged = parse_claims(res.text, max_claims=REDUCE_MAX_CLAIMS)
        contra = parse_claims(
            json.dumps({"claims": data.get("contradictions") or []}, ensure_ascii=False),
            max_claims=REDUCE_MAX_CLAIMS,
        )
        raw_timeline = data.get("timeline")
        timeline = [
            U.neutralize(x).strip()[:300]
            for x in (raw_timeline if isinstance(raw_timeline, list) else [])
            if isinstance(x, str) and x.strip()
        ][:40]
        return (merged or list(claims)[:REDUCE_MAX_CLAIMS]), contra, timeline

    first_task = reduce_task(intent)
    merged, contra, timeline = await once(first_task)
    if first_task is Task.SEARCH and contra and not budget.stop_reason():
        merged, contra, timeline = await once(Task.DEEP)
    st.claims = len(merged)
    st.contradictions = len(contra)
    st.scan_cost_usd = round(budget.spent, 6)
    await DS._emit(events, "reduce", claims_in=len(claims), contradictions=len(contra))
    return merged, contra, timeline


def render_claims_block(
    claims: Sequence[Claim],
    contradictions: Sequence[Claim] = (),
    timeline: Sequence[str] = (),
    *,
    budget_tokens: int = REDUCE_DIGEST_TOKENS,
) -> str:
    """Da'volar dayjesti — sintezga **bitta** dalil konverti ichida tushadi.

    Alohida konvert ochilmaydi (invariant: bitta konvert). Blok konvertning
    boshida turadi, dalil satrlari esa ostida — shuning uchun da'vodagi
    `#c…m…` token'lari validatorda ham topiladi.
    """
    if not claims and not contradictions and not timeline:
        return ""
    max_chars = int(budget_tokens * S.CHARS_PER_TOKEN)
    lines: list[str] = ["## claims (extracted from the messages below)"]
    total = len(lines[0])
    for c in claims:
        cites = " ".join(f"#{t}" for t in c.cites)
        line = f"- {c.text} {cites}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    if contradictions:
        lines.append("## contradictions")
        for c in contradictions:
            cites = " ".join(f"#{t}" for t in c.cites)
            line = f"- {c.text} {cites}"
            if total + len(line) > max_chars:
                break
            lines.append(line)
            total += len(line) + 1
    if timeline:
        lines.append("## timeline")
        for row in timeline:
            if total + len(row) > max_chars:
                break
            lines.append(f"- {row}")
            total += len(row) + 1
    return "\n".join(lines)


# ─── (h) baho — **skan LLM chaqiruvlaridan oldin** ───────────────────────────


def _price_for(task: Task) -> tuple[float, float]:
    """Vazifa modelining (input, output) narxi — topilmasa ehtiyotkor sukut."""
    try:
        from app.llm.router import resolve

        _provider, model = resolve(task)
    except Exception:  # provider sozlanmagan (test/CI) — baho baribir kerak
        return FALLBACK_PRICE
    from app.llm.pricing import _normalize

    return PRICES.get(_normalize(model), FALLBACK_PRICE)


def estimate(
    buckets: Sequence[Bucket],
    *,
    mode: str = "deep",
    per_bucket_k: int = PER_BUCKET_K,
    prices: dict[str, tuple[float, float]] | None = None,
) -> Estimate:
    """Skan narxining **oldindan** bahosi: bucket sanog'i x k x narx jadvali.

    Hech bir LLM chaqirilmaydi — shuning uchun `exhaustive` uchun
    foydalanuvchidan tasdiq so'rash mumkin (`confirm`) va u tasdiq
    bermaguncha bitta token ham sarflanmaydi.

    `est_cost_usd` — **skan** narxi (filtr + map + reduce), ya'ni aynan
    `COST_CEILING` boshqaradigan qism; `est_total_usd` esa sintez bilan
    birga — foydalanuvchi to'laydigan raqam. Ikkalasi alohida, chunki
    `est_cost_usd` va `ceiling_usd` bir hodisada yonma-yon turadi, ya'ni
    ular **solishtiriladigan** bo'lishi kerak.

    Nomzod sanog'i **kalibrlangan**, `bucket x k` emas: xom ko'paytma
    har doim `FILTER_MAX` ga to'yinardi va baho 5 ta bucket'da ham, 200
    tada ham bir xil chiqardi (jonli yugurishlarda doim 0.1124). Endi
    `bucket x k x EST_PULL_YIELD`, pastdan `ELBOW_MIN`, yuqoridan
    `FILTER_MAX` — ya'ni baho bucket soniga qarab **o'sadi** va qopqoqdan
    oshmaydi. Keep-rate ham o'lchovdan (`EST_KEEP_RATE`).
    """
    p = dict(
        prices
        or {
            "filter": _price_for(Task.ROUTE),
            "map": _price_for(Task.ROUTE),
            "reduce": _price_for(Task.SEARCH),
        }
    )
    p.setdefault("synth", _price_for(Task.DEEP))
    planned = len(buckets)
    expected = int(planned * per_bucket_k * EST_PULL_YIELD)
    est_messages = min(
        FILTER_MAX.get(mode, FILTER_MAX["deep"]),
        elbow_hi_for(mode),
        max(ELBOW_MIN, expected),
    )
    if not planned:
        est_messages = 0

    n_batches = math.ceil(est_messages / FILTER_BATCH) if est_messages else 0
    # Bir satr ≈ 150 belgi matn + ~45 belgi sarlavha; rubrika har partiyada.
    filter_in = n_batches * (
        int(len(SCALE_FILTER_PROMPT) / S.CHARS_PER_TOKEN)
        + int(FILTER_BATCH * (FILTER_LINE_CHARS + 45) / S.CHARS_PER_TOKEN)
    )
    filter_out = n_batches * 120

    kept = min(SOFT_MAX_EVIDENCE, int(est_messages * EST_KEEP_RATE))
    chunks = 0 if kept <= MAP_SKIP_TOTAL else min(MAP_MAX_CHUNKS, math.ceil(kept / MAP_CHUNK_MSGS))
    map_in = chunks * (
        int(len(SCALE_MAP_PROMPT) / S.CHARS_PER_TOKEN)
        + int(MAP_CHUNK_MSGS * (MAP_LINE_CHARS + 45) / S.CHARS_PER_TOKEN)
    )
    map_out = chunks * MAP_MAX_TOKENS

    claims_in = chunks * MAP_MAX_CLAIMS
    reduce_rubric = int(len(SCALE_REDUCE_PROMPT) / S.CHARS_PER_TOKEN)
    reduce_in = (reduce_rubric + claims_in * 60) if chunks else 0
    reduce_out = REDUCE_MAX_TOKENS if chunks else 0

    # Sintez — bitta `Task.DEEP` chaqiruvi: rubrika + da'volar dayjesti +
    # `SYNTH_EVIDENCE_LINES` ta dalil satri, chiqishi `SYNTH_MAX_TOKENS`.
    synth_in = (
        int(len(DEEP_SYNTH_PROMPT) / S.CHARS_PER_TOKEN)
        + int(SYNTH_EVIDENCE_LINES * (MAP_LINE_CHARS + 45) / S.CHARS_PER_TOKEN)
        + (REDUCE_DIGEST_TOKENS if chunks else 0)
    )
    synth_out = DS.SYNTH_MAX_TOKENS

    def cost(stage: str, tin: int, tout: int) -> float:
        price_in, price_out = p[stage]
        return (tin * price_in + tout * price_out) / 1_000_000

    breakdown = {
        "filter": round(cost("filter", filter_in, filter_out), 6),
        "map": round(cost("map", map_in, map_out), 6),
        "reduce": round(cost("reduce", reduce_in, reduce_out), 6),
        "synth": round(cost("synth", synth_in, synth_out), 6),
    }
    scan = breakdown["filter"] + breakdown["map"] + breakdown["reduce"]
    return Estimate(
        mode=mode,
        buckets=planned,
        est_messages=est_messages,
        est_cost_usd=round(scan, 6),
        est_total_usd=round(scan + breakdown["synth"], 6),
        ceiling_usd=COST_CEILING.get(mode, COST_CEILING["deep"]),
        weeks=len({b.label for b in buckets}),
        range_from=min((b.since for b in buckets), default=None),
        range_to=max((b.until for b in buckets), default=None),
        breakdown=breakdown,
    )


# ─── quvur ───────────────────────────────────────────────────────────────────


async def query_vectors(
    meter: DS._Meter, *, question: str, subqueries: Sequence[str], limit: int = 5
) -> list[list[float]]:
    """Savol + subquery vektorlari — **bitta** embed chaqirig'ida.

    Bucket reytingi ularning maksimumini oladi, `pull` esa birinchisini
    (standalone savol) ishlatadi.
    """
    texts = [question, *[s for s in subqueries if s and s != question]][:limit]
    try:
        res = await meter.embed(texts, task_type=EMBED_QUERY)
    except LLMError as exc:
        log.warning("scale.embed_failed", error=str(exc)[:200])
        return []
    return [list(v) for v in res.vectors]


async def run_scale(
    session_factory: DS.SessionFactory,
    *,
    account_id: int,
    plan: DS.DeepPlan,
    question: str,
    mode: str,
    meter: DS._Meter,
    taint: DS._Taint,
    events: DS.EventSink | None = None,
    chat_titles: dict[int, str] | None = None,
    deadline: float | None = None,
    confirm: bool = True,
    locale: str = "uz",
) -> ScaleOutcome:
    """Kaskadning to'liq yugurishi — `deep_search._run_pipeline` shu yerga kiradi.

    `confirm=False` va `mode="exhaustive"` bo'lsa **faqat baho** qaytadi:
    bucket rejasi quriladi, `estimate` hodisasi chiqadi va bosqichlar
    boshlanmaydi (bitta filtr tokeni ham sarflanmaydi).
    """
    started = time.monotonic()
    stats = ScanStats()
    outcome = ScaleOutcome(stats=stats)
    ceiling = COST_CEILING.get(mode, COST_CEILING["deep"])
    wall = deadline if deadline is not None else started + EXHAUSTIVE_WALL_S
    budget = _Budget(meter, ceiling=ceiling, deadline=wall)

    vectors = await query_vectors(meter, question=question, subqueries=plan.subqueries)
    async with session_factory() as session:
        buckets = await plan_buckets(
            session,
            account_id,
            plan=plan,
            vectors=vectors,
            chat_titles=chat_titles,
            limit=bucket_max_for(mode),
            mode=mode,
        )
    outcome.buckets = buckets
    stats.buckets_planned = len(buckets)

    est = estimate(buckets, mode=mode)
    outcome.estimate = est
    stats.est_cost_usd = est.est_cost_usd
    stats.est_total_usd = est.est_total_usd
    await DS._emit(events, "estimate", **est.to_dict())
    if not buckets:
        stats.note_stop("filter", "complete")
        return outcome

    await DS._emit(
        events,
        "buckets",
        planned=len(buckets),
        chats=len({b.chat_id for b in buckets}),
        weeks=len({b.label for b in buckets}),
        range={
            "from": min(b.since for b in buckets).isoformat(),
            "to": max(b.until for b in buckets).isoformat(),
        },
    )
    if mode == "exhaustive" and not confirm:
        # Faqat baho — hech bir skan tokeni sarflanmadi. Buni "reja to'liq
        # skanlandi" deb yozish yolg'on bo'lardi: `scan_runs` da ham, qamrovda
        # ham alohida sabab turadi.
        stats.stopped_by = "estimate_only"
        return outcome

    # ── (b) tortish ─────────────────────────────────────────────────────────
    spec = build_pull_spec(
        account_id,
        question=question,
        subqueries=plan.subqueries,
        qvec=vectors[0] if vectors else None,
    )
    pull_budget = pull_budget_for(budget.left_s())
    pool, bucket_of, scanned = await pull_candidates(
        session_factory,
        buckets=buckets,
        spec=spec,
        events=events,
        budget_s=pull_budget,
        elbow_hi=elbow_hi_for(mode),
        stats=stats,
    )
    stats.buckets_scanned = scanned
    if scanned < len(buckets):
        # Pull byudjeti tugadi — qolgan bucket'lar **umuman** qaralmadi.
        # Buni jim o'tkazish "hammasini ko'rdim" degan yolg'on bo'lardi.
        stats.note_stop("pull", "wall")
        stats.truncated = True
    if not pool:
        stats.note_stop("filter", budget.stop_reason() or "complete")
        return outcome

    # ── gidratatsiya + (c) dedup ────────────────────────────────────────────
    async with session_factory() as session:
        pool = await DS._hydrate(session, pool, account_id=account_id)
    taint.add_hits(pool)  # begona matn quvurga kirdi — reja/refleksiya uchun qopqoq
    pool = dedup_fast(pool)
    stats.candidates = len(pool)
    stats.pulled = max(stats.pulled, len(pool))
    outcome.pool = pool

    # ── (d) filtr ───────────────────────────────────────────────────────────
    scored = await filter_batched(
        session_factory,
        account_id=account_id,
        question=question,
        pool=pool,
        bucket_of=bucket_of,
        meter=meter,
        budget=budget,
        taint=taint,
        events=events,
        mode=mode,
        stats=stats,
    )
    if not scored:
        return outcome

    # ── (e) tanlov ──────────────────────────────────────────────────────────
    evidence, trimmed = select_evidence(scored)
    stats.quota_trimmed = trimmed
    stats.evidence = len(evidence)
    outcome.evidence = evidence
    await DS._emit(
        events,
        "select",
        evidence=len(evidence),
        chats=len({h.chat_id for h in evidence}),
        weeks=len({week_key(h) for h in evidence}),
        quota_trimmed=trimmed,
    )

    # Kosinus dedup va mavzu klasterlari **faqat shu yerda** — to'plam ≤1000,
    # ya'ni O(n²) halol narx (hovuzda u 20k² bo'lardi).
    head = evidence[:CLUSTER_INPUT_MAX]
    async with session_factory() as session:
        vecs = await S.fetch_vectors(session, [h.message_id for h in head])
    head = C.dedup_cosine(head, vecs)
    outcome.clusters = C.cluster_topics(
        head[: C.MAX_CLUSTER_INPUT], vecs, distance_threshold=C.TOPIC_DISTANCE_THRESHOLD
    )
    with contextlib.suppress(Exception):
        outcome.clusters = await C.label_clusters(
            outcome.clusters, llm=cast(LLM, meter), locale=locale
        )

    # ── (f) map → reduce ────────────────────────────────────────────────────
    chunks = map_chunks(evidence)
    claims = await map_claims(
        question=question,
        chunks=chunks,
        meter=meter,
        budget=budget,
        taint=taint,
        events=events,
        stats=stats,
    )
    merged, contra, timeline = await reduce_claims(
        question=question,
        claims=claims,
        intent=plan.intent,
        meter=meter,
        budget=budget,
        taint=taint,
        events=events,
        stats=stats,
    )
    outcome.claims = merged
    outcome.contradictions = contra
    outcome.timeline = timeline
    if budget.stop_reason():
        stats.note_stop("map", budget.stop_reason())
        stats.truncated = True
    stats.scan_cost_usd = round(budget.spent, 6)
    log.info(
        "scale.done",
        mode=mode,
        buckets=stats.buckets_planned,
        scanned=stats.buckets_scanned,
        pulled=stats.pulled,
        candidates=stats.candidates,
        kept=stats.kept,
        evidence=stats.evidence,
        claims=stats.claims,
        pull_stopped_by=stats.pull_stopped_by,
        filter_stopped_by=stats.filter_stopped_by,
        stopped_by=stats.stopped_by,
        cost_usd=stats.scan_cost_usd,
        elapsed_s=round(time.monotonic() - started, 1),
    )
    return outcome


__all__ = [
    "BUCKET_MAX",
    "BUCKET_MAX_DEEP",
    "COST_CEILING",
    "ELBOW_KEEP_MIN_FRAC",
    "ELBOW_MIN",
    "EST_KEEP_RATE",
    "EST_PULL_YIELD",
    "EXHAUSTIVE_WALL_S",
    "FILTER_BATCH",
    "FILTER_KEEP_MIN",
    "FILTER_MAX",
    "MAP_CHUNK_MIN",
    "MAP_CHUNK_MSGS",
    "MAP_MAX_CHUNKS",
    "MAP_SKIP_TOTAL",
    "MODES",
    "PER_BUCKET_K",
    "POOL_MAX",
    "PULL_CONCURRENCY",
    "PULL_RESERVE_S",
    "SCALE_MODES",
    "SOFT_MAX_EVIDENCE",
    "STOP_RANK",
    "STOP_REASONS",
    "Bucket",
    "Claim",
    "Estimate",
    "PullSpec",
    "ScaleOutcome",
    "ScanStats",
    "batches",
    "bucket_coverage",
    "bucket_max_for",
    "build_filter_envelope",
    "build_map_envelope",
    "build_pull_spec",
    "build_reduce_envelope",
    "cache_lookup",
    "cache_store",
    "combine_stops",
    "dedup_banded",
    "dedup_fast",
    "elbow_cut",
    "elbow_hi_for",
    "estimate",
    "filter_batched",
    "filter_line",
    "iso_label",
    "map_chunks",
    "map_claims",
    "parse_claims",
    "parse_keep",
    "plan_buckets",
    "pull_bucket",
    "pull_budget_for",
    "pull_candidates",
    "question_hash",
    "reduce_claims",
    "reduce_task",
    "render_claims_block",
    "run_scale",
    "select_evidence",
    "should_early_stop",
    "week_key",
]
