"""Dedup va mavzu klasterlari — to'rt pog'ona, sof kod (arxitektura v2, 6-bo'lim).

Nega kerak: forward va sitata arxivda ko'p. Bitta post o'nta kanalda bir xil
matn bilan yotadi; RRF ularning hammasini yuqoriga chiqaradi va byudjet bitta
matnning nusxalari bilan to'ladi. Dedup shu nusxalarni **bitta vakil** ostiga
yig'adi, qolganlari esa `also_seen_in` ro'yxatida qoladi — "bu xabar 4 chatda
uchradi" javobda qimmatli ma'lumot, uni yo'qotmaymiz.

Pog'onalar arzondan qimmatga:

1. `dedup_exact`   — sha1(normalized): aynan bir xil matn, vektorsiz;
2. `dedup_simhash` — Hamming ≤3: kichik tahrir/prefiks farqi, vektorsiz;
3. `dedup_cosine`  — cosine ≥0.93: qayta yozilgan, lekin ayni ma'no;
4. `cluster_topics` — agglomerative, average-linkage, cosine masofa 0.20.

numpy loyihaning bog'liqligi emas, shuning uchun hammasi sof Python. n ≤ 400
uchun bu yetarli (O(n²) masofa matritsasi ≈ 80k juft).

Yorliqlash **bitta** arzon chaqiruv (`Task.ROUTE`), kirish
`<untrusted_data kind="labels">` konvertida, chiqish esa imperativ naqshlardan
tozalanadi: klaster yorlig'i UI'da ko'rsatiladi, u yerdan begona odam
foydalanuvchiga ko'rsatma bera olmasin.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from operator import mul as _mul
from typing import TYPE_CHECKING, Any

from app.llm import LLM, LLMError, Msg, Task
from app.logging import get_logger
from app.services import untrusted as U
from app.services.prompts_retrieval import CLUSTER_LABEL_PROMPT
from app.services.textnorm import APOSTROPHES, hamming64, simhash64, uz_fold

if TYPE_CHECKING:
    from app.services.search import Hit

log = get_logger(__name__)

SIMHASH_MAX_HAMMING = 3
COSINE_DUP_THRESHOLD = 0.93
TOPIC_DISTANCE_THRESHOLD = 0.20
MAX_CLUSTER_INPUT = 400  # sof Python O(n²) uchun halol chegara
MAX_LABEL_WORDS = 8
LABEL_SNIPPET_CHARS = 160
LABEL_MEMBERS = 3

_WS_RE = re.compile(r"\s+")
_JSON_RE = re.compile(r"\{.*\}", re.S)

# Yorliq — mavzu nomi, ko'rsatma emas. Bu naqshlar boshidan qirqiladi.
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:"
    r"please|you\s+must|you\s+should|now\b|ignore\b|disregard\b|forget\b|"
    r"click\b|open\b|send\b|write\b|reply\b|call\b|visit\b|buy\b|"
    rf"iltimos|yubor\w*|bos\w*|yoz\w*|o[c{re.escape(APOSTROPHES)}]\w*|qil\w*|"
    r"пожалуйста|нажми\w*|отправ\w*|напиши\w*|перейди\w*|игнорируй\w*"
    r")[\s:,-]*",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://\S+|@[\w_]{4,}")


@dataclass(slots=True)
class Cluster:
    """Bitta mavzu — vakil hit va a'zolar."""

    hits: list[Hit]
    label: str = ""
    centroid: list[float] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.hits)

    @property
    def representative(self) -> Hit:
        return self.hits[0]


# ─── yordamchi ───────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Dedup uchun matn: fold + bo'shliqlarni yig'ish."""
    return _WS_RE.sub(" ", uz_fold(text or "")).strip()


def text_sha1(text: str) -> str:
    return hashlib.sha1(normalize(text).encode(), usedforsecurity=False).hexdigest()


hamming = hamming64  # `textnorm` bilan yagona amalga oshirish


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    ab = dot(a, b)
    na = math.sqrt(dot(a, a))
    nb = math.sqrt(dot(b, b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return ab / (na * nb)


def unit(vec: Sequence[float]) -> list[float]:
    """Vektorni birlik uzunlikka keltiradi; nol vektor — bo'sh ro'yxat.

    Normallashtirilgan vektorlarda cosine **oddiy skalyar ko'paytma**, ya'ni
    juftlik uchun uch o'tish o'rniga bitta. 400 ta 768-o'lchovli vektorda bu
    farq sezilarli: `dedup`/`cluster_topics` 80k juftlikni ko'radi (o'lchov
    `docs/architecture-v2.md` §11.1).
    """
    n = math.sqrt(dot(vec, vec))
    return [x / n for x in vec] if n else []


def unit_map(vectors: dict[int, list[float]], ids: Iterable[int]) -> dict[int, list[float]]:
    """`{message_id: birlik vektor}` — faqat bir marta, kerakli id'lar uchun."""
    out: dict[int, list[float]] = {}
    for mid in ids:
        vec = vectors.get(mid)
        if not vec:
            continue
        u = unit(vec)
        if u:
            out[mid] = u
    return out


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Skalyar ko'paytma — birlik vektorlar uchun bu aynan cosine.

    `map(operator.mul, …)` ataylab: generator ifodasiga nisbatan sezilarli
    tezroq, chunki sikl C tomonda qoladi (numpy loyihaning bog'liqligi emas).
    """
    return float(sum(map(_mul, a, b)))


def _absorb(keeper: Hit, dup: Hit) -> None:
    """Dublikatni vakil ostiga yig'adi — manba yo'qolmaydi."""
    mark = (dup.chat_id, dup.tg_msg_id, dup.date)
    if mark not in keeper.also_seen_in and dup.message_id != keeper.message_id:
        keeper.also_seen_in.append(mark)
    for extra in dup.also_seen_in:
        if extra not in keeper.also_seen_in:
            keeper.also_seen_in.append(extra)


# ─── 1-pog'ona: aynan bir xil ────────────────────────────────────────────────


def dedup_exact(hits: list[Hit]) -> list[Hit]:
    kept: list[Hit] = []
    by_hash: dict[str, Hit] = {}
    for h in hits:
        body = normalize(h.text)
        if not body:  # matnsiz (media) xabarlar bir-birining dublikati emas
            kept.append(h)
            continue
        key = text_sha1(h.text)
        first = by_hash.get(key)
        if first is None:
            by_hash[key] = h
            kept.append(h)
        else:
            _absorb(first, h)
    return kept


# ─── 2-pog'ona: simhash ──────────────────────────────────────────────────────


def dedup_simhash(hits: list[Hit], *, max_hamming: int = SIMHASH_MAX_HAMMING) -> list[Hit]:
    kept: list[Hit] = []
    sigs: list[tuple[int, Hit]] = []
    for h in hits:
        body = normalize(h.text)
        if len(body) < 24:  # qisqa matnda simhash shovqin beradi
            kept.append(h)
            continue
        sig = simhash64(body)
        dup = next((k for s, k in sigs if hamming(s, sig) <= max_hamming), None)
        if dup is None:
            sigs.append((sig, h))
            kept.append(h)
        else:
            _absorb(dup, h)
    return kept


# ─── 3-pog'ona: cosine near-dup ──────────────────────────────────────────────


def dedup_cosine(
    hits: list[Hit],
    vectors: dict[int, list[float]],
    *,
    threshold: float = COSINE_DUP_THRESHOLD,
) -> list[Hit]:
    # Normallashtirish **bir marta**: ilgari har juftlikda ikkala vektorning
    # normi qayta hisoblanardi (juftlikka uch o'tish o'rniga bitta).
    units = unit_map(vectors, (h.message_id for h in hits))
    kept: list[Hit] = []
    #: Faqat vektori bor vakillar — taqqoslash shular bilan yuradi.
    reps: list[tuple[Hit, list[float]]] = []
    for h in hits:
        vec = units.get(h.message_id)
        if not vec:
            kept.append(h)
            continue
        dup = next((k for k, kv in reps if dot(vec, kv) >= threshold), None)
        if dup is None:
            kept.append(h)
            reps.append((h, vec))
        else:
            _absorb(dup, h)
    return kept


def dedup(hits: list[Hit], vectors: dict[int, list[float]] | None = None) -> list[Hit]:
    """To'liq quvur: sha1 → simhash → (vektor bo'lsa) cosine."""
    out = dedup_simhash(dedup_exact(hits))
    return dedup_cosine(out, vectors) if vectors else out


# ─── 4-pog'ona: mavzu klasterlari ────────────────────────────────────────────


def cluster_topics(
    hits: list[Hit],
    vectors: dict[int, list[float]],
    *,
    distance_threshold: float = TOPIC_DISTANCE_THRESHOLD,
) -> list[Cluster]:
    """Agglomerative, average-linkage, cosine masofa. Vektorsiz hit — o'z klasteri."""
    pool = hits[:MAX_CLUSTER_INPUT]
    with_vec = [h for h in pool if vectors.get(h.message_id)]
    without = [h for h in pool if not vectors.get(h.message_id)]
    if not with_vec:
        return [Cluster([h]) for h in pool]

    groups: list[list[int]] = [[i] for i in range(len(with_vec))]
    vecs = [vectors[h.message_id] for h in with_vec]
    # Juftlik masofalari bir marta hisoblanadi, birlashtirish o'rtachani
    # yangilaydi. Vektorlar avval **birlik** uzunlikka keltiriladi — shundan
    # keyin cosine bitta skalyar ko'paytma (§11.1 o'lchovi: 400 hit uchun
    # 5.5 s → 0.9 s).
    units = [unit(v) for v in vecs]
    dist: dict[tuple[int, int], float] = {}
    for i in range(len(units)):
        ui = units[i]
        for j in range(i + 1, len(units)):
            dist[(i, j)] = 1.0 - dot(ui, units[j])

    def pair(a: int, b: int) -> float:
        return dist[(a, b)] if a < b else dist[(b, a)]

    def linkage(g1: list[int], g2: list[int]) -> float:
        return sum(pair(a, b) for a in g1 for b in g2) / (len(g1) * len(g2))

    while len(groups) > 1:
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = linkage(groups[i], groups[j])
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None or best[0] > distance_threshold:
            break
        _, i, j = best
        groups[i] = groups[i] + groups[j]
        groups.pop(j)

    clusters = [
        Cluster([with_vec[i] for i in sorted(g)], centroid=_centroid([vecs[i] for i in g]))
        for g in groups
    ]
    clusters.sort(key=lambda c: c.size, reverse=True)
    clusters.extend(Cluster([h]) for h in without)
    return clusters


def _centroid(vecs: list[list[float]]) -> list[float]:
    if not vecs:
        return []
    n = len(vecs)
    return [sum(v[i] for v in vecs) / n for i in range(len(vecs[0]))]


# ─── yorliqlash ──────────────────────────────────────────────────────────────


def sanitize_label(raw: str) -> str:
    """Yorliq — mavzu nomi: imperativ boshlanish, havola va uzunlik qirqiladi."""
    text = U.neutralize(raw or "")
    text = _URL_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip(" \"'`*#-:;.,")
    prev = None
    while prev != text:  # "please ignore ..." kabi zanjirlar
        prev = text
        text = _IMPERATIVE_RE.sub("", text).strip(" \"'`*#-:;.,")
    words = text.split()
    return " ".join(words[:MAX_LABEL_WORDS])


def build_label_envelope(clusters: list[Cluster], *, locale: str = "uz") -> str:
    from app.services.search import compact_text

    blocks: list[str] = []
    for i, c in enumerate(clusters):
        snippets = [
            compact_text(h.text or "", max_chars=LABEL_SNIPPET_CHARS)
            for h in c.hits[:LABEL_MEMBERS]
            if (h.text or "").strip()
        ]
        body = " / ".join(U.neutralize(s) for s in snippets) or "(no text)"
        blocks.append(f"{i}. ({c.size} msgs) {body}")
    return (
        f'<untrusted_data kind="labels" locale="{U.attr(locale)}" count="{len(clusters)}">\n'
        + "\n".join(blocks)
        + "\n</untrusted_data>"
    )


def parse_labels(raw: str, *, n: int) -> dict[int, str]:
    m = _JSON_RE.search(raw or "")
    if not m:
        return {}
    try:
        data: Any = json.loads(m.group(0))
    except (ValueError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    labels = data.get("labels")
    if not isinstance(labels, dict):
        return {}
    out: dict[int, str] = {}
    for key, value in labels.items():
        try:
            idx = int(key)
        except (ValueError, TypeError):
            continue
        if 0 <= idx < n and isinstance(value, str):
            clean = sanitize_label(value)
            if clean:
                out[idx] = clean
    return out


async def label_clusters(
    clusters: list[Cluster], *, llm: LLM | None = None, locale: str = "uz"
) -> list[Cluster]:
    """Bitta `Task.ROUTE` chaqiruvi. Nosozlikda yorliqlar bo'sh qoladi."""
    if not clusters:
        return clusters
    msgs = [
        Msg.system(CLUSTER_LABEL_PROMPT),
        Msg.user(build_label_envelope(clusters, locale=locale)),
    ]
    try:
        res = await (llm or LLM()).chat(Task.ROUTE, msgs, json_mode=True, max_tokens=500)
    except LLMError as exc:
        log.warning("cluster.label_failed", error=str(exc)[:200])
        return clusters
    for idx, label in parse_labels(res.text, n=len(clusters)).items():
        clusters[idx].label = label
    return clusters


__all__ = [
    "COSINE_DUP_THRESHOLD",
    "MAX_CLUSTER_INPUT",
    "SIMHASH_MAX_HAMMING",
    "TOPIC_DISTANCE_THRESHOLD",
    "Cluster",
    "build_label_envelope",
    "cluster_topics",
    "cosine",
    "dedup",
    "dedup_cosine",
    "dedup_exact",
    "dedup_simhash",
    "dot",
    "hamming",
    "label_clusters",
    "normalize",
    "parse_labels",
    "sanitize_label",
    "text_sha1",
    "unit",
    "unit_map",
]
