"""`<untrusted_data>` konvertiga tushadigan matnni zararsizlantirish (invariant 3).

Muammo: Telegram matni konvert ichiga **xom** qo'yilsa, begona odam o'z
xabarida `</untrusted_data>` yozib konvertni "yopishi" va qolgan matnni
ko'rsatma sifatida ko'rsatishi mumkin (prompt injection). Shu bilan birga
ko'rinmas/bidi belgilar (U+200B…U+200F, U+202A…U+202E, U+2066…U+2069, U+FEFF)
matnni modelga bir xil, odamga boshqa ko'rsatadi.

Yechim — **bitta** joy: konvertga tushadigan har qanday kontent
`neutralize()` dan, atributlar `attr()` dan o'tadi. Konvert teglarining
o'zini kod yozadi, kontent esa hech qachon yangi teg yarata olmaydi.

Qoidalar:
  * `<untrusted_data` / `</untrusted_data` (registrga bog'liq emas, `<` bilan
    teg nomi orasida bo'shliq bo'lsa ham): `<` o'rniga U+2039 belgisi. Matn
    o'qilishi saqlanadi, teg esa teg bo'lmay qoladi.
  * ko'rinmas va bidi boshqaruv belgilari olib tashlanadi;
  * C0 boshqaruv belgilari (`\n` va `\t` dan tashqari) olib tashlanadi.
"""

from __future__ import annotations

import re

# `<` + (ixtiyoriy bo'shliq) + (ixtiyoriy `/`) + `untrusted_data`
_TAG_RE = re.compile(r"<(?=\s*/?\s*untrusted_data)", re.IGNORECASE)

# ko'rinmas/bidi + C0 (\n, \t qoladi) + DEL
_INVISIBLE_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff\x00-\x08\x0b-\x1f\x7f]")

# Deep search sitata tokeni (`#c<chat>m<msg>` / `#x<n>`) \u2014 begona matn ichida
# uchrasa u **token bo'lmasligi** kerak: aks holda begona odam o'z xabarida
# `#c1m42` yozib javobdagi sitatani boshqa xabarga yo'naltira olardi (H3/L3).
#
# **Yagona ta'rif** (M4): `deep_search` aynan shu naqsh bilan sitata
# **ajratadi** (`CITE_TOKEN_RE`), bu yerda esa u **o'ldiriladi**. Ilgari ikkita
# nusxa bor edi va ular bir xil emasdi \u2014 zararsizlantirish lookahead'ida `\b`
# turardi, ajratishda esa yo'q: `#c1m42x` zararsizlantirishdan **o'tib**
# ketardi (`\b` `2`\u2194`x` orasida yo'q), ajratuvchi esa undan `c1m42` ni olardi.
# Shuning uchun bu yerda ham `\b` yo'q: prefiksi mos kelgan hamma narsa
# o'ldiriladi (ortiqcha o'ldirish xavfsiz tomonga xato).
_CITE_TOKEN_BODY = r"c\d{1,12}m\d{1,12}|x\d{1,12}"  # noqa: S105 — parol emas, regex

#: Sitata tokeni \u2014 `deep_search._extract_citations` shu bilan ajratadi.
CITE_TOKEN_RE = re.compile(rf"#({_CITE_TOKEN_BODY})")

_CITE_LOOKALIKE_RE = re.compile(rf"#(?=(?:{_CITE_TOKEN_BODY}))", re.IGNORECASE)
_ZWSP = "\u200b"  # `#` dan keyin qo'yiladi \u2014 ko'rinishi bir xil, token emas

_FAKE_LT = "\u2039"  # teg ochmaydigan "<" o'rnini bosuvchi
_FAKE_GT = "\u203a"  # teg yopmaydigan ">" o'rnini bosuvchi
ATTR_MAX_CHARS = 128


def neutralize(text: str | None) -> str:
    """Konvert ichiga tushadigan kontent — teg va sitata tokeni yarata olmaydi.

    Tartib muhim: avval ko'rinmas belgilar olib tashlanadi, **keyin** sitata
    tokeniga o'xshagan naqshga U+200B qo'yiladi — aks holda qo'yilgan belgining
    o'zi darhol o'chib ketardi.
    """
    if not text:
        return ""
    out = _TAG_RE.sub(_FAKE_LT, _INVISIBLE_RE.sub("", str(text)))
    return _CITE_LOOKALIKE_RE.sub("#" + _ZWSP, out)


def attr(value: str | None, *, max_chars: int = ATTR_MAX_CHARS) -> str:
    """Konvert atributi uchun qiymat (chat sarlavhasi, jo'natuvchi nomi)."""
    out = (
        neutralize(value)
        .replace('"', "'")
        .replace("<", _FAKE_LT)
        .replace(">", _FAKE_GT)
        .replace("\n", " ")
        .replace("\t", " ")
    )
    return out[:max_chars].strip()
