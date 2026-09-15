"""Matn normalizatsiyasi — saqlash chegarasidagi himoya va indeks poydevori.

Uch vazifa, uchalasi ham **sof funksiya** (I/O yo'q, LLM yo'q, deterministik):

``strip_invisible``
    Ingest vaqtida bir marta: NFC + ko'rinmas belgilarni olib tashlash
    (zero-width, bidi, Unicode Tags bloki, C0 boshqaruvlari). Bu
    ``services/untrusted.neutralize`` ni **almashtirmaydi** — u prompt
    chegarasida ishlaydi, bu esa saqlash chegarasida: DB'ga ko'rinmas
    prompt injection tashuvchisi yozilmasin.

``uz_norm`` / ``uz_fold``
    Qidiruv uchun "fold". O'zbek tilida Postgres lug'ati yo'q, matn lotin va
    kirillda aralash, apostrof esa kamida o'n xil kodlanadi. Fold hammasini
    bitta fazoga tushiradi: ``oʻzbek`` / ``o'zbek`` / ``o‘zbek`` / ``ўзбек``
    → ``ozbek``.

    **Diqqat:** bu funksiyalarning aynan nusxasi Postgres'da ham bor
    (migratsiya ``0008``, ``uz_norm(text)`` / ``uz_fold(text)``) —
    ``messages.text_fold`` va ``messages.tsv_uz`` generated column'lari o'sha
    SQL versiyasi bilan quriladi. Ikkalasi bir-biridan **jimgina ajralib
    ketsa indeks buziladi**: Python tomonida qurilgan so'rov SQL tomonida
    qurilgan indeksga tushmay qoladi. Shu sababli:

    * bu yerda o'zgarish qilsangiz, migratsiyadagi SQL'ni ham o'zgartiring
      (yoki yangi migratsiya bilan ``CREATE OR REPLACE`` qiling va
      generated column'larni qayta hisoblang);
    * ``tests/test_db_integration.py::test_uz_fold_sql_python_parity`` ikkala
      amalni bir xil korpusda solishtiradi va CI'da Postgres service bilan
      yuradi.

    Parite uchun ataylab **``casefold()`` emas, ``lower()``** ishlatiladi:
    Postgres'da ``casefold`` yo'q, ``lower()`` esa (UTF-8 lokalda) Python
    ``str.lower()`` bilan bir xil natija beradi. Chetga chiqarilgan hollar:
    ``ß`` (``lower`` uni ``ss`` qilmaydi — ikkala tomonda ham ``ß`` bo'lib
    qoladi, ya'ni parite saqlanadi, faqat "fold" to'liq emas) va turkcha
    nuqtali ``İ`` (ikkala tomon ham ``i`` + U+0307 beradi, lekin bu o'zbek
    korpusida uchramaydi). Postgres ``C``/``POSIX`` lokalda ishga
    tushirilgan bo'lsa ``lower()`` faqat ASCII'ni pasaytiradi — u holda
    parite buziladi, shuning uchun DB UTF-8 lokalda bo'lishi shart.

``simhash64``
    64-bitli simhash (3-belgili shingle) — forward/sitata takrorlarini
    vektorsiz topish uchun (Hamming ≤3 → dublikat nomzod).
"""

from __future__ import annotations

import hashlib
import unicodedata

__all__ = [
    "APOSTROPHES",
    "hamming64",
    "simhash64",
    "strip_invisible",
    "uz_fold",
    "uz_norm",
]

# ─── ko'rinmas belgilar ──────────────────────────────────────────────────────

#: Olib tashlanadigan kod nuqtalari (inclusive oraliqlar).
_INVISIBLE_RANGES: tuple[tuple[int, int], ...] = (
    (0x0000, 0x0008),  # C0 — \t (0x09) va \n (0x0A) dan oldingilar
    (0x000B, 0x001F),  # C0 — \n dan keyingilar (\r ham shu yerda)
    (0x007F, 0x007F),  # DEL
    (0x200B, 0x200F),  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    (0x202A, 0x202E),  # bidi: LRE, RLE, PDF, LRO, RLO
    (0x2066, 0x2069),  # bidi izolyatsiya: LRI, RLI, FSI, PDI
    (0xFEFF, 0xFEFF),  # BOM / ZWNBSP
    (0xE0000, 0xE007F),  # Tags bloki — ko'rinmas prompt injection tashuvchisi
)

#: ``str.translate`` uchun jadval: har bir ko'rinmas belgi → ``None`` (o'chirish).
_INVISIBLE_TABLE: dict[int, None] = {
    cp: None for lo, hi in _INVISIBLE_RANGES for cp in range(lo, hi + 1)
}


def strip_invisible(text: str) -> str:
    """NFC normalizatsiya + ko'rinmas belgilarni olib tashlash.

    Olib tashlanadi: zero-width (U+200B–U+200F, U+FEFF), bidi boshqaruvlari
    (U+202A–U+202E, U+2066–U+2069), Unicode Tags bloki (U+E0000–U+E007F),
    C0 boshqaruv belgilari (``\\n`` va ``\\t`` dan tashqari) va DEL.

    Args:
        text: xom matn (Telegram xabari).

    Returns:
        Tozalangan, NFC ko'rinishidagi matn. Bo'sh kirish → bo'sh chiqish.

    Examples:
        >>> strip_invisible("sa\\u200blom\\ufeff")
        'salom'
        >>> strip_invisible("a\\rb")
        'ab'
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).translate(_INVISIBLE_TABLE)


# ─── o'zbekcha fold ──────────────────────────────────────────────────────────

#: Apostrof sinfi — o'zbek lotinida ``oʻ``/``gʻ`` va tutuq belgisi uchun
#: ishlatiladigan barcha kodlashlar. Fold'da butunlay **o'chiriladi**.
APOSTROPHES: str = (
    "'"  # U+0027 APOSTROPHE
    "‘"  # ‘ LEFT SINGLE QUOTATION MARK
    "’"  # ’ RIGHT SINGLE QUOTATION MARK
    "‛"  # ‛ SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "ʻ"  # ʻ MODIFIER LETTER TURNED COMMA  (rasmiy oʻzbek belgisi)
    "ʼ"  # ʼ MODIFIER LETTER APOSTROPHE    (rasmiy tutuq belgisi)
    "ʹ"  # ʹ MODIFIER LETTER PRIME
    "´"  # ´ ACUTE ACCENT
    "`"  # U+0060 GRAVE ACCENT
    "ʽ"  # ʽ MODIFIER LETTER REVERSED COMMA
)

_APOSTROPHE_TABLE: dict[int, None] = {ord(ch): None for ch in APOSTROPHES}

#: Kirill → lotin. Manbalar **bitta belgi**, nishonlar sof lotin — shuning
#: uchun bir o'tishda qo'llash SQL'dagi "avval ko'p belgililar, keyin 1:1"
#: tartibi bilan aynan bir xil natija beradi.
_CYR_TO_LAT: dict[str, str] = {
    # ko'p belgili (SQL'da `replace()` zanjiri)
    "ё": "yo",
    "ю": "yu",
    "я": "ya",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sh",
    # 1:1 (SQL'da `translate()`)
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "x",
    "ы": "i",
    "э": "e",
    "ғ": "g",  # U+0493 — o'zbek kirill
    "қ": "q",  # U+049B
    "ҳ": "h",  # U+04B3
    "ў": "o",  # U+045E
    # o'chiriladi (SQL'da `translate()` nishonsiz qismi)
    "ъ": "",
    "ь": "",
}

_CYR_TABLE: dict[int, str] = {ord(src): dst for src, dst in _CYR_TO_LAT.items()}


def uz_norm(text: str) -> str:
    """NFC → ``lower()`` → apostrof sinfini butunlay olib tashlash.

    Transliteratsiya **qilinmaydi** — kirill kirill bo'lib qoladi.
    ``uz_fold`` ning birinchi bosqichi; alohida ham foydali (masalan
    keshning kaliti: ``md5(uz_norm(query))``).

    Args:
        text: xom yoki qisman tozalangan matn.

    Returns:
        Normallashtirilgan matn; apostrof sinfidagi belgilar yo'q.

    Examples:
        >>> uz_norm("Oʻzbek")
        'ozbek'
        >>> uz_norm("O'zbek") == uz_norm("O‘zbek")
        True
    """
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).lower().translate(_APOSTROPHE_TABLE)


def uz_fold(text: str) -> str:
    """``uz_norm`` + o'zbek kirill → lotin transliteratsiyasi.

    Natijada apostrof **bo'lmaydi** va kirill yozuv qolmaydi, shuning uchun
    ``oʻzbek``, ``o'zbek``, ``ozbek`` va ``ўзбек`` — to'rttasi ham
    ``"ozbek"`` ga tushadi. Indeks (``messages.text_fold``, ``tsv_uz``) va
    so'rov shu funksiyadan o'tadi.

    Args:
        text: xom yoki qisman tozalangan matn.

    Returns:
        Fold qilingan matn (lotin, apostrofsiz, lowercase).

    Examples:
        >>> uz_fold("Oʻzbekiston")
        'ozbekiston'
        >>> uz_fold("Ўзбекистон")
        'ozbekiston'
        >>> uz_fold("Привет") == 'privet'
        True
    """
    if not text:
        return ""
    return uz_norm(text).translate(_CYR_TABLE)


# ─── simhash ─────────────────────────────────────────────────────────────────

_SIMHASH_BITS = 64
_SHINGLE = 3


def _shingle_hash(shingle: str) -> int:
    """Shingle → 64-bit butun son (barqaror: blake2b, tuz yo'q)."""
    digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def simhash64(text: str) -> int:
    """64-bitli simhash, 3-belgili shingle ustida.

    Dedup'ning ikkinchi pog'onasi (birinchisi — ``sha1`` aynan mosligi):
    ikkita matn orasidagi Hamming masofasi ≤3 bo'lsa — dublikat nomzod.
    Vektor kerak emas, shuning uchun forward va sitata takrorlarini
    embedding'siz topadi.

    Matn avval ``uz_fold`` dan o'tkaziladi — ya'ni yozuv va apostrof
    farqlari natijaga ta'sir qilmaydi. Bo'sh (yoki 3 belgidan qisqa) matn
    uchun ``0`` qaytadi.

    Args:
        text: taqqoslanadigan matn.

    Returns:
        ``0 <= value < 2**64`` oraliqdagi deterministik butun son.

    Examples:
        >>> simhash64("") == 0
        True
        >>> simhash64("Oʻzbekiston") == simhash64("ўзбекистон")
        True
    """
    folded = uz_fold(text)
    if len(folded) < _SHINGLE:
        return 0
    weights = [0] * _SIMHASH_BITS
    seen: dict[str, int] = {}
    for i in range(len(folded) - _SHINGLE + 1):
        seen[folded[i : i + _SHINGLE]] = seen.get(folded[i : i + _SHINGLE], 0) + 1
    for shingle, count in seen.items():
        h = _shingle_hash(shingle)
        for bit in range(_SIMHASH_BITS):
            weights[bit] += count if (h >> bit) & 1 else -count
    value = 0
    for bit in range(_SIMHASH_BITS):
        if weights[bit] > 0:
            value |= 1 << bit
    return value


def hamming64(a: int, b: int) -> int:
    """Ikki 64-bitli simhash orasidagi Hamming masofasi (0..64)."""
    return int((a ^ b).bit_count())
