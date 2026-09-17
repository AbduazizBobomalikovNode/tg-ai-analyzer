"""`services/textnorm.py` — sof Python testlari.

SQL tomonidagi egizak (`uz_norm` / `uz_fold` Postgres funksiyalari) bilan
paritet `tests/test_db_integration.py` da tekshiriladi — u yerda shu fayldagi
`FOLD_CORPUS` ishlatiladi, ya'ni korpus **bitta joyda** turadi.
"""

from __future__ import annotations

import pytest

from app.services.stopwords import STOP_ALL, is_stopword, strip_stopwords
from app.services.textnorm import (
    APOSTROPHES,
    hamming64,
    simhash64,
    strip_invisible,
    uz_fold,
    uz_norm,
)

# ─── paritet korpusi (SQL testi ham shu ro'yxatni ishlatadi) ─────────────────

#: uz-lotin (apostrofning 6 xil kodlashi bilan), uz-kirill, rus, ingliz,
#: emoji, aralash, chegaraviy hollar. ≥40 ta satr.
FOLD_CORPUS: tuple[str, ...] = (
    # uz-lat, apostrof variantlari
    "O'zbekiston",
    "Oʻzbekiston",
    "O‘zbekiston",
    "O’zbekiston",
    "O´zbekiston",
    "O`zbekiston",
    "Oʽzbekiston",
    "Oʹzbekiston",
    "O‛zbekiston",
    "gʻalaba",
    "g'alaba",
    "g‘alaba",
    "ma'lumot",
    "maʼlumot",
    "sanʼat",
    "to'lov qachon bo'ladi?",
    "Toʻlov 20-sanaga koʻchirildi",
    "Assalomu alaykum, yaxshimisiz?",
    "Bugun ishga bormadim, kasal boʻlib qoldim",
    # uz-cyr
    "Ўзбекистон",
    "ЎЗБЕКИСТОН",
    "ғалаба",
    "Тўлов қачон бўлади?",
    "Ассалому алайкум",
    "Бугун ишга бормадим",
    "мақсад ва вазифалар",
    "ҳужжат тайёр",
    # ru
    "Привет, как дела?",
    "ПРИВЕТ",
    "Оплата будет завтра, я уже отправил документы",
    "Съешь ещё этих мягких французских булок",
    "Цена — 1 000 000 сум",
    "Щука, чай, шар, жук, южный, ясный, ёлка",
    "Объявление: подъезд закрыт",
    # en
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog",
    "PAYMENT DUE 2026-03-14",
    # emoji / belgilar / chegaraviy
    "👍👍👍",
    "salom 👋 qalaysan?",
    "",
    " ",
    "\n\t",
    "1234567890",
    "!@#$%^&*()_+-=[]{};:,.<>/?|~",
    "https://t.me/c/123456/789",
    "@nurbek_dev #tolov",
    # aralash yozuv
    "Salom, привет, hello — bir xabarda uch til",
    "Oʻzbekistonda GDP o'sdi (Узбекистан, GDP growth)",
    "kelishuv: to'lov 20-sanagacha, иначе штраф",
    "Нурбек said: to'lovni koʻchiramiz",
    # aksent / boshqa lotin
    "café naïve résumé",
    "Ñandú",
)


# ─── strip_invisible ─────────────────────────────────────────────────────────


def test_strip_invisible_removes_zero_width_and_bidi() -> None:
    dirty = "sa​lom‎‮dunyo⁦!﻿"
    assert strip_invisible(dirty) == "salomdunyo!"


def test_strip_invisible_removes_tags_block() -> None:
    hidden = "hello" + "".join(chr(cp) for cp in range(0xE0041, 0xE0046))
    assert strip_invisible(hidden) == "hello"


def test_strip_invisible_keeps_newline_and_tab_drops_other_c0() -> None:
    assert strip_invisible("a\nb\tc\rd\x00e\x07f\x7fg") == "a\nb\tcdefg"


def test_strip_invisible_normalizes_to_nfc() -> None:
    decomposed = "ŭzbek"  # u + combining breve → ŭ (U+016D)
    assert strip_invisible(decomposed) == "ŭzbek"


def test_strip_invisible_empty() -> None:
    assert strip_invisible("") == ""


def test_strip_invisible_is_idempotent() -> None:
    for s in FOLD_CORPUS:
        once = strip_invisible(s)
        assert strip_invisible(once) == once


# ─── uz_norm / uz_fold ───────────────────────────────────────────────────────


def test_uz_norm_drops_every_apostrophe_variant() -> None:
    for ch in APOSTROPHES:
        assert uz_norm(f"o{ch}zbek") == "ozbek"


def test_uz_norm_keeps_cyrillic() -> None:
    assert uz_norm("Ўзбекистон") == "ўзбекистон"


@pytest.mark.parametrize(
    "variant",
    ["O'zbek", "Oʻzbek", "O‘zbek", "O’zbek", "Ozbek", "OZBEK", "ўзбек", "Ўзбек", "ЎЗБЕК"],
)
def test_uz_fold_collapses_all_spellings(variant: str) -> None:
    assert uz_fold(variant) == "ozbek"


def test_uz_fold_has_no_apostrophes_and_no_cyrillic() -> None:
    for s in FOLD_CORPUS:
        folded = uz_fold(s)
        assert not any(ch in folded for ch in APOSTROPHES), s
        assert not any("Ѐ" <= ch <= "ӿ" for ch in folded), s


def test_uz_fold_russian_transliteration() -> None:
    assert uz_fold("Привет") == "privet"
    assert uz_fold("Щука чай шар жук южный ясный ёлка цена") == (
        "shuka chay shar juk yujniy yasniy yolka tsena"
    )
    assert uz_fold("Объявление подъезд") == "obyavlenie podezd"


def test_uz_fold_uzbek_cyrillic_letters() -> None:
    assert uz_fold("ғқҳў") == "gqho"


def test_uz_fold_is_idempotent() -> None:
    for s in FOLD_CORPUS:
        once = uz_fold(s)
        assert uz_fold(once) == once


def test_uz_fold_empty() -> None:
    assert uz_fold("") == ""
    assert uz_norm("") == ""


def test_uz_fold_leaves_latin_and_digits_alone() -> None:
    assert uz_fold("Payment 2026-03-14") == "payment 2026-03-14"


# ─── simhash ─────────────────────────────────────────────────────────────────


def test_simhash_is_deterministic() -> None:
    text = "Toʻlov 20-sanaga koʻchirildi, kelishdikmi?"
    assert simhash64(text) == simhash64(text)


def test_simhash_ignores_script_and_apostrophe_differences() -> None:
    assert simhash64("Oʻzbekiston") == simhash64("ЎЗБЕКИСТОН")


def test_simhash_short_text_is_zero() -> None:
    assert simhash64("") == 0
    assert simhash64("ab") == 0


def test_simhash_fits_in_64_bits() -> None:
    for s in FOLD_CORPUS:
        assert 0 <= simhash64(s) < 2**64


def test_simhash_near_duplicates_are_close() -> None:
    a = simhash64("Ertaga soat 10 da uchrashamiz, hujjatlarni olib keling")
    b = simhash64("Ertaga soat 10 da uchrashamiz, hujjatlarni olib keling!")
    far = simhash64("Bugun havo juda issiq, dengizga boramiz")
    assert hamming64(a, b) <= 8
    assert hamming64(a, b) < hamming64(a, far)


# ─── stopwords ───────────────────────────────────────────────────────────────


def test_stopwords_are_folded() -> None:
    for word in STOP_ALL:
        assert uz_fold(word) == word


def test_is_stopword_script_and_case_insensitive() -> None:
    assert is_stopword("Va") and is_stopword("ва") and is_stopword("VA")
    assert is_stopword("И") and is_stopword("The")
    assert not is_stopword("to'lov")
    assert not is_stopword("Нурбек")


def test_strip_stopwords_keeps_input_when_everything_is_noise() -> None:
    assert strip_stopwords(["va", "bu", "the"]) == ["va", "bu", "the"]
    assert strip_stopwords(["va", "to'lov"]) == ["to'lov"]


def test_stopword_lists_are_non_trivial() -> None:
    assert len(STOP_ALL) >= 200
