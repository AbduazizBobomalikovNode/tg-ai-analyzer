"""Pozitsiyaga asoslangan retrieval metrikalari — sof funksiyalar.

Hech qanday I/O yo'q (DB/LLM chaqirilmaydi), shuning uchun
``tests/test_evals.py`` qo'lda hisoblangan qiymatlar bilan tekshiradi.

Hammasi ikkilik (binary) relevance bilan ishlaydi: ``gold`` — to'g'ri javob
xabar id'lari to'plami (``gold_msg_ids``, tg_msg_id fazosida), ``retrieved`` —
qidiruv qaytargan id'lar, mosligi bo'yicha kamayish tartibida (eng yaqindan
uzoqqa).

``CLAUDE.md`` dagi "Narx qopqoqlari" bandi: LLM tebranadi, bitta yugurish
hech narsani isbotlamaydi — shuning uchun ``median_spread`` / ``intervals_overlap``
kamida uch marta yugurilgan natijalarni taqqoslash uchun.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "intervals_overlap",
    "median_spread",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
]


def recall_at_k(retrieved: Sequence[int], gold: Sequence[int], k: int) -> float:
    """Top-``k`` orasida topilgan ``gold`` id'lar ulushi (0..1).

    Bo'sh ``gold`` — o'lchanadigan narsa yo'q, ``1.0`` qaytadi (hech narsa
    yo'qotilmagan, deb hisoblanadi) — bu holat chaqiruvchi tomonida odatda
    oldindan filtrlanadi (golden set bo'sh gold bilan yozilmaydi).

    Examples:
        >>> recall_at_k([3, 1, 2], [1, 2], 3)
        1.0
        >>> recall_at_k([3, 1, 2], [1, 2], 1)
        0.0
    """
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    top = set(retrieved[:k])
    return len(top & gold_set) / len(gold_set)


def _dcg_at_k(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg_at_k(retrieved: Sequence[int], gold: Sequence[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain, ikkilik relevance bilan.

    Ideal tartib — barcha ``gold`` id'lar (``min(len(gold), k)`` tasi)
    ro'yxat boshida turgan holat. Bo'sh ``gold`` → ``1.0``.

    Examples:
        >>> round(ndcg_at_k([9, 1, 9], [1], 3), 4)
        0.6309
        >>> ndcg_at_k([1, 9, 9], [1], 3)
        1.0
    """
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    gains = [1.0 if doc_id in gold_set else 0.0 for doc_id in retrieved[:k]]
    dcg = _dcg_at_k(gains, k)
    ideal_gains = [1.0] * min(len(gold_set), k)
    idcg = _dcg_at_k(ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mrr_at_k(retrieved: Sequence[int], gold: Sequence[int], k: int) -> float:
    """Birinchi to'g'ri javobning teskari o'rni (``1/rank``).

    Top-``k`` ichida gold topilmasa — ``0.0``. Bo'sh ``gold`` → ``1.0``.

    Examples:
        >>> mrr_at_k([9, 1, 2], [1], 3)
        0.5
        >>> mrr_at_k([9, 9, 9], [1], 3)
        0.0
    """
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def median_spread(values: Sequence[float]) -> tuple[float, float, float]:
    """``(median, min, max)`` — kamida 3 marta yugurilgan qiymatlar uchun.

    Bo'sh ``values`` — chaqiruvchi xatosi, ``ValueError`` ko'tariladi (aks
    holda "0 yugurishdan 0.0 median" xato ishonch beradi).

    Examples:
        >>> median_spread([0.4, 0.6, 0.5])
        (0.5, 0.4, 0.6)
        >>> median_spread([0.2, 0.8])
        (0.5, 0.2, 0.8)
    """
    if not values:
        raise ValueError("median_spread: bo'sh qiymatlar ro'yxati")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return median, ordered[0], ordered[-1]


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Ikki ``(min, max)`` oraliq kesishadimi.

    Taqqoslashda ishlatiladi: kesishsa — natija "inconclusive" (bitta
    yugurish to'plami boshqasidan **aniq** ustun emas).

    Examples:
        >>> intervals_overlap((0.1, 0.3), (0.25, 0.5))
        True
        >>> intervals_overlap((0.1, 0.2), (0.3, 0.4))
        False
    """
    a_lo, a_hi = min(a), max(a)
    b_lo, b_hi = min(b), max(b)
    return a_lo <= b_hi and b_lo <= a_hi
