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
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}


def estimate_cost(provider: str, model: str, tokens_in: int, tokens_out: int) -> float | None:
    if provider == "claude_code":
        return 0.0
    key = _normalize(model)
    price = PRICES.get(key)
    if price is None:
        return None
    return round((tokens_in * price[0] + tokens_out * price[1]) / 1_000_000, 6)


def _normalize(model: str) -> str:
    """`claude-haiku-4-5-20251001` → `claude-haiku-4-5`; `models/gemini-2.5-flash` → …"""
    m = model.split("/")[-1].strip().lower()
    for key in sorted(PRICES, key=len, reverse=True):
        if m == key or m.startswith(key + "-") or m.startswith(key + "@"):
            return key
    return m
