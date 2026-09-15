"""Taxminiy narxlar (USD / 1M token) — dashboard uchun xarajat bahosi.

Bu **taxmin**: provider narxlari o'zgaradi, cache/batch chegirmalari hisobga
olinmaydi. `claude_code` — obuna, marginal xarajat 0. Noma'lum model → None.
Yangilash: shu jadval; kod boshqa joyda narx bilmaydi.
"""

from __future__ import annotations

# (input, output) USD per 1M token
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Google
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-embedding-001": (0.15, 0.0),
    # Rasm generatsiya. Gemini narxni **rasm boshiga** e'lon qiladi, bizning
    # jadval esa token bo'yicha — shuning uchun output tokeni sifatida
    # e'lon qilingan "rasm = 1290 output token" ekvivalenti ishlatiladi:
    # $30/1M * 1290 = ~$0.039 / rasm. Matn kiritish oddiy flash narxida.
    # `services/images.py` shu taxmin bilan `IMAGE_OUTPUT_TOKENS` yozadi.
    "gemini-2.5-flash-image": (0.30, 30.0),
    # phase4/B — `gemini-embedding-2` uchun **taxmin** (arxitektura hujjatidagi
    # $0.20/M input; batch yarmi — `BATCH_PRICES`). Narx e'lon qilinganda
    # yangilang. MUHIM: kalit — API javobidagi **qaytgan** model nomi bo'lishi
    # kerak; alias qaytsa (`-preview`) va jadvalda bo'lmasa `estimate_cost`
    # `None` qaytaradi va kunlik xarajat limiti jimgina ishlamay qoladi.
    "gemini-embedding-2": (0.20, 0.0),
    "gemini-embedding-2-preview": (0.20, 0.0),
    # DeepSeek — peak narxlar (off-peak yarmi: 01:00-04:00 va 06:00-10:00 UTC
    # tashqarisi). Byudjet qopqog'i uchun ataylab qimmatrog'i olingan: kam
    # baholash limitni ishlamay qoldiradi.
    "deepseek-v4-flash": (0.44, 1.32),
    "deepseek-v4-pro": (1.32, 3.96),
    # Eski aliaslar — API ikkalasini ham v4-flash'ga yo'naltiradi. Javobdagi
    # `model` maydoni konkret nomni qaytaradi, shuning uchun bular kamdan-kam
    # ishlatiladi; narx v4-flash bilan bir xil.
    "deepseek-chat": (0.44, 1.32),
    "deepseek-reasoner": (0.44, 1.32),
}


# Batch (asinxron) rejim narxi, USD / 1M input token — **taxmin**, jonli
# narxning yarmi. Faqat shu jadvaldagi modellar uchun; qolgani jonli narxda
# hisoblanadi (kam baholashdan ko'ra ortiq baholagan yaxshi: byudjet qopqog'i
# ishlab tursin).
BATCH_PRICES: dict[str, float] = {
    # phase4/C — Gemini Batch API e'lon qilingan narxi: jonli $0.15/M ning
    # yarmi. `embed_batches.poll_batches` shu bilan `embed_batches.cost_usd`
    # ni yozadi.
    "gemini-embedding-001": 0.075,
    "gemini-embedding-2": 0.10,
    "gemini-embedding-2-preview": 0.10,
}


def estimate_cost(
    provider: str, model: str, tokens_in: int, tokens_out: int, *, batch: bool = False
) -> float | None:
    """Taxminiy xarajat. `batch=True` — Batch API (arzonroq, `BATCH_PRICES`)."""
    if provider == "claude_code":
        return 0.0
    key = _normalize(model)
    price = PRICES.get(key)
    if price is None:
        return None
    price_in = BATCH_PRICES[key] if batch and key in BATCH_PRICES else price[0]
    return round((tokens_in * price_in + tokens_out * price[1]) / 1_000_000, 6)


def _normalize(model: str) -> str:
    """`claude-haiku-4-5-20251001` → `claude-haiku-4-5`; `models/gemini-2.5-flash` → …"""
    m = model.split("/")[-1].strip().lower()
    for key in sorted(PRICES, key=len, reverse=True):
        if m == key or m.startswith(key + "-") or m.startswith(key + "@"):
            return key
    return m
