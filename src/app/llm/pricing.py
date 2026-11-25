"""Taxminiy narxlar (USD / 1M token) — dashboard uchun xarajat bahosi.

Bu **taxmin**: provider narxlari o'zgaradi. Batch (`BATCH_PRICES`), kontekst keshi
(`CACHED_PRICES`), thinking (output narxida) va Flex tier hisobga olinadi;
`claude_code` — obuna, marginal xarajat 0. Noma'lum model → None.
Yangilash: shu jadval; kod boshqa joyda narx bilmaydi.
"""

from __future__ import annotations

from typing import Any

# (input, output) USD per 1M token
PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    # Google — 2.5 avlod (2026-09'dan yangi mijozlarga yopiq, lekin joriy
    # deploy'lar/eski override'lar hali shu nomlarni ishlatishi mumkin —
    # qator o'chirilmaydi, aks holda ularning xarajati kuzatilmay qoladi).
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    # 3.x avlod — ai.google.dev/gemini-api/docs/pricing, 2026-09-16 tekshirilgan.
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-flash": (1.50, 9.00),
    # Narx 200k tokengacha bo'lgan bosqich uchun ($4.00/$18.00 — 200k dan
    # yuqori). Bizning jadval bitta narx qabul qiladi — arzonroq (odatiy)
    # bosqich olingan, kam baholashdan ko'ra son quyi chiqadi (qopqoq baribir
    # katta kontekstda ham ishlaydi, faqat biroz kam baholaydi).
    "gemini-3.1-pro-preview": (2.00, 12.00),
    # `gemini-pro-latest` — hozircha `gemini-3.1-pro-preview`ga alias
    # (2026-09-16 tekshirilgan, `model_version` shu nomni qaytaradi). Alias
    # boshqa modelga o'tsa narx eskiradi — # taxmin 2026-09-16.
    "gemini-pro-latest": (2.00, 12.00),
    # `gemini-flash-latest` / `gemini-flash-lite-latest` — alias, hozir mos
    # ravishda 3.8-flash / 3.5-flash-lite'ga ko'rsatadi. Google bularni
    # muddatsiz "eng so'nggi" modelga almashtirib turadi — # taxmin 2026-09-16,
    # narx alias siljiganda eskiradi.
    "gemini-flash-latest": (0.75, 3.75),
    "gemini-flash-lite-latest": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.7-flash": (0.75, 3.75),
    "gemini-3.8-flash": (0.75, 3.75),
    # Rasm generatsiya — `gemini-2.5-flash-image` bilan bir xil konvensiya
    # (pastdagi izohga qarang): 1K piksel rasm ≈ $0.067, `IMAGE_OUTPUT_TOKENS`
    # (1290) ekvivalenti bilan $0.067/1290*1e6 ≈ $51.94/M output. Matn kiritish
    # standart narxda ($0.25/M). # taxmin 2026-09-16 (rasmiy narx sahifasida
    # to'g'ridan-to'g'ri token narxi yo'q, faqat rasm-boshiga narx e'lon qilingan).
    "gemini-3.1-flash-image": (0.25, 51.94),
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
    # Suhbat modellari (3.x, ai.google.dev/gemini-api/docs/pricing,
    # 2026-09-16) — batch input ham jonli narxning yarmi. DIQQAT: bu jadval
    # faqat **input**ni chegirmalaydi (`estimate_cost` output'ni doim jonli
    # narxda hisoblaydi) — hozircha zarari yo'q, chunki bu loyihada chat
    # vazifalari uchun batch yo'li yo'q (faqat embedding), lekin kelajakda
    # chat batch qo'shilsa shu funksiya ham yangilanishi kerak.
    "gemini-3.5-flash-lite": 0.15,
    "gemini-3.5-flash": 0.75,
    "gemini-3.1-pro-preview": 1.00,
    "gemini-3.1-flash-lite": 0.125,
    "gemini-3.6-flash": 0.375,
    "gemini-3.7-flash": 0.375,
    "gemini-3.8-flash": 0.375,
}


# Kontekst keshi (implicit/explicit) — **o'qilgan** kesh tokeni narxi, USD / 1M.
# Manba: ai.google.dev/gemini-api/docs/pricing, 2026-09-25 tekshirilgan
# ("Context caching price", Standard, ≤200k prompt). Gemini'da
# `cached_content_token_count` `prompt_token_count` ning **qismi** — ya'ni
# keshlangan tokenlar input narxi o'rniga shu narxda hisoblanadi.
# Jadvalda yo'q model (masalan 3.5-flash-lite: "Not available", Claude —
# `input_tokens` kesh o'qishini o'z ichiga olmaydi) → kesh tokeni to'liq
# input narxida qoladi va jarayonda bir marta `pricing.no_cached_price` log'i.
CACHED_PRICES: dict[str, float] = {
    "gemini-3.5-flash": 0.15,
    "gemini-3.1-pro-preview": 0.20,
    "gemini-pro-latest": 0.20,
    "gemini-3.1-flash-lite": 0.025,
    "gemini-3.6-flash": 0.075,
    "gemini-3.7-flash": 0.075,
    "gemini-3.8-flash": 0.075,
    "gemini-flash-latest": 0.075,
}

#: Flex tier (`service_tier="flex"`) — Google e'lon qilgan narx Standard'ning
#: **yarmi** (input, output+thinking, kesh — hammasi), 2026-09-25 tekshirilgan.
#: Faqat Gemini'da; boshqa provider'da tier e'tiborsiz.
FLEX_MULTIPLIER = 0.5
TIER_STANDARD = "standard"
TIER_FLEX = "flex"

_no_cached_warned: set[str] = set()


def estimate_cost(
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    *,
    batch: bool = False,
    tokens_thinking: int = 0,
    tokens_cached: int = 0,
    tier: str = TIER_STANDARD,
) -> float | None:
    """Taxminiy xarajat. `batch=True` — Batch API (arzonroq, `BATCH_PRICES`).

    * `tokens_thinking` — **output narxida** (Google: "output price includes
      thinking tokens"). Gemini `candidates_token_count` ularni o'z ichiga
      olmaydi, shuning uchun alohida qo'shiladi.
    * `tokens_cached` — `tokens_in` ning keshdan o'qilgan qismi; narx
      `CACHED_PRICES` da bo'lsa shu narxda, bo'lmasa to'liq input narxida.
    * `tier="flex"` — Gemini'da hammasi `FLEX_MULTIPLIER` bilan.
    """
    if provider == "claude_code":
        return 0.0
    key = _normalize(model)
    price = PRICES.get(key)
    if price is None:
        return None
    price_in = BATCH_PRICES[key] if batch and key in BATCH_PRICES else price[0]
    cached = max(0, min(int(tokens_cached or 0), int(tokens_in or 0)))
    cached_price = CACHED_PRICES.get(key)
    if cached and cached_price is None:
        if key not in _no_cached_warned:
            _no_cached_warned.add(key)
            from app.logging import get_logger

            get_logger(__name__).info("pricing.no_cached_price", model=key, cached=cached)
        cached = 0
    cost_in = (tokens_in - cached) * price_in + cached * (cached_price or 0.0)
    cost_out = (tokens_out + max(0, int(tokens_thinking or 0))) * price[1]
    total = (cost_in + cost_out) / 1_000_000
    if tier == TIER_FLEX and provider == "gemini":
        total *= FLEX_MULTIPLIER
    return round(total, 6)


def usage_cost(provider: str, model: str, usage: Any, *, tier: str = TIER_STANDARD) -> float | None:
    """`Usage` obyektidan to'g'ridan-to'g'ri narx (thinking va kesh bilan)."""
    return estimate_cost(
        provider,
        model,
        int(getattr(usage, "tokens_in", 0) or 0),
        int(getattr(usage, "tokens_out", 0) or 0),
        tokens_thinking=int(getattr(usage, "thinking_tokens", 0) or 0),
        tokens_cached=int(getattr(usage, "cached_tokens", 0) or 0),
        tier=tier,
    )


def result_cost(res: Any) -> float | None:
    """`ChatResult` narxi — provider, model, usage va ishlatilgan tier bilan."""
    return usage_cost(
        res.provider, res.model, res.usage, tier=getattr(res, "service_tier", "") or TIER_STANDARD
    )


def _normalize(model: str) -> str:
    """`claude-haiku-4-5-20251001` → `claude-haiku-4-5`; `models/gemini-2.5-flash` → …"""
    m = model.split("/")[-1].strip().lower()
    for key in sorted(PRICES, key=len, reverse=True):
        if m == key or m.startswith(key + "-") or m.startswith(key + "@"):
            return key
    return m
