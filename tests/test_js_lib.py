"""Frontend modullari (`static/lib/*.js`) — node bo'lsa ishga tushadi.

Qo'riqlaydi:
  * `lib/palette.js::uzFold` ↔ `services/textnorm.uz_fold` **pariteti**.
    Bu uchinchi nusxa (Python, Postgres `uz_fold()`, brauzer): birortasi
    jimgina ajralib ketsa qidiruv natijalari serverda va brauzerda
    boshqacha bo'ladi — va buni hech kim sezmaydi.
  * fuzzy (subsequence) mosligining asosiy xossalari.
  * `peerColor` determinizmi va 1..7 oralig'i.
  * har bir JS faylning sintaksisi (`node --check`).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pytest

from app.services.textnorm import uz_fold

STATIC = Path(__file__).resolve().parents[1] / "src/app/web/static"
PALETTE = STATIC / "lib/palette.js"
SHELL_JS = STATIC / "lib/shell.js"
FMT_JS = STATIC / "lib/fmt.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node yo'q")

#: Parite korpusi — apostrofning har xil kodlashi, kirill, aralash matn.
CORPUS: list[str] = [
    "",
    "O\u02bbzbekiston",
    "O'zbekiston",
    "O\u2018zbekiston",
    "O`zbekiston",
    "Ўзбекистон",
    "o\u02bbquv g\u02bbalaba",
    "ЎҚУВ ҒАЛАБА",
    "Привет, мир",
    "Ёлка ЮЛА Яша Цена Чай Шум Щи",
    "to\u02bclov shartnoma",
    "Ish <b>chat</b>",
    "Иш чат",
    "a\u02bczo a'zo a\u02bbzo",
    "MiXeD CaSe Тест",
    "ъь твёрдый мягкий",
    "hello world 123 @username",
    "Тошкент shahri",
    "Xalqaro Ҳамкорлик",
    "  bo'sh   joylar  ",
]


def _node_module(script: str, payload: object) -> object:
    """ESM bo'lagini ishga tushiradi; `INPUT` — stdin'dagi JSON."""
    assert NODE is not None
    src = (
        f"const P = await import({json.dumps(PALETTE.as_uri())});\n"
        "const { readFileSync } = await import('node:fs');\n"
        "const INPUT = JSON.parse(readFileSync(0, 'utf8'));\n"
        f"{script}\n"
    )
    proc = subprocess.run(  # noqa: S603
        [NODE, "--input-type=module", "-e", src],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node xatosi: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def test_uz_fold_matches_python() -> None:
    """JS `uzFold` == Python `uz_fold` (belgi-ma-belgi)."""
    out = _node_module(
        "process.stdout.write(JSON.stringify(INPUT.map((s) => P.uzFold(s))));",
        CORPUS,
    )
    assert isinstance(out, list)
    expected = [uz_fold(s) for s in CORPUS]
    mismatch = [(s, a, b) for s, a, b in zip(CORPUS, out, expected, strict=True) if a != b]
    assert not mismatch, f"fold ajralib ketdi: {mismatch}"


def test_uz_fold_collapses_the_four_spellings() -> None:
    """Fold'ning maqsadi: to'rt xil yozuv bitta kalitga tushadi."""
    out = _node_module(
        "process.stdout.write(JSON.stringify(INPUT.map((s) => P.uzFold(s))));",
        ["O\u02bbzbek", "O'zbek", "Ozbek", "Ўзбек"],
    )
    assert set(out) == {"ozbek"}


def test_fuzzy_subsequence_and_ranking() -> None:
    script = """
const cases = INPUT.map(([n, h]) => {
  const m = P.fuzzy(n, h);
  return m === null ? null : Math.round(m.score);
});
process.stdout.write(JSON.stringify(cases));
"""
    cases = [
        ["ozb", "O\u02bbzbekiston yangiliklari"],  # ketma-ketlik + so'z boshi
        ["oyan", "O\u02bbzbekiston yangiliklari"],  # tarqoq ketma-ketlik
        ["zzz", "O\u02bbzbekiston"],  # mos emas
        ["ўзб", "O\u02bbzbekiston"],  # kirill so'rov, lotin nomzod
        ["", "har qanday"],  # bo'sh so'rov — hamma mos
    ]
    scores = _node_module(script, cases)
    assert scores[0] is not None and scores[1] is not None
    assert scores[0] > scores[1], "aniq prefiks tarqoq mosllikdan yuqori turishi kerak"
    assert scores[2] is None
    assert scores[3] is not None, "kirill so'rov fold orqali lotin nomzodga tushishi kerak"
    assert scores[4] == 0


def test_peer_color_is_deterministic_and_in_range() -> None:
    script = """
const out = INPUT.map((id) => P.peerColor(id));
process.stdout.write(JSON.stringify({ out, repeat: INPUT.map((id) => P.peerColor(id)) }));
"""
    ids = [0, 1, 7, 8, 777000, -1001234567890, "1001234567890", None, "abc"]
    res = _node_module(script, ids)
    assert res["out"] == res["repeat"]
    assert all(1 <= v <= 7 for v in res["out"])
    # 7 ta rangning hech bo'lmaganda bir nechtasi ishlatiladi (hammasi 1 emas)
    assert len(set(res["out"])) > 1


def test_initials_are_grapheme_safe() -> None:
    script = "process.stdout.write(JSON.stringify(INPUT.map((s) => P.initials(s))));"
    names = ["Ali Vali", "O\u02bbzbekiston", "", "   ", "Иван Петров", "🎯 kanal"]
    out = _node_module(script, names)
    assert out[0] == "AV"
    assert out[2] == "?" and out[3] == "?"
    assert out[4] == "ИП"
    assert out[5].startswith("🎯")


def test_initials_skip_digits_and_punctuation() -> None:
    """L6: raqam/tinish belgisi so'z boshida bo'lsa ham harf topiladi;
    faqat raqam/tinish belgili so'zlar (yoki so'zlar) o'tkazib yuboriladi."""
    script = "process.stdout.write(JSON.stringify(INPUT.map((s) => P.initials(s))));"
    names = ["1-A guruh", "I | Telegram", ",,,"]
    out = _node_module(script, names)
    assert out == ["AG", "IT", "•"]


def test_palette_chats_fetched_and_ranked() -> None:
    """H3: `lib/shell.js::fetchPaletteChats` mocked `/api/me` + chats bilan
    ishlaydi, natija `rank()` orqali "gor" so'roviga "Gorizont"ni topadi —
    ya'ni ⌘K "Chatlar" bo'limi endi sahifa alohida yuklamasa ham to'ladi.
    """
    assert NODE is not None
    rows = [
        {"id": 1, "title": "Ish chati"},
        {"id": 2, "title": "Gorizont"},
        {"id": 3, "title": "Oila"},
    ]
    src = f"""
const S = await import({json.dumps(SHELL_JS.as_uri())});
const P = await import({json.dumps(PALETTE.as_uri())});
const ROWS = {json.dumps(rows)};
const calls = [];
const getFn = async (path) => {{
  calls.push(path);
  if (path === "/api/me") return {{ accounts: [{{ id: 9, status: "active" }}] }};
  if (path === "/api/accounts/9/chats?limit=1000") return {{ items: ROWS }};
  throw new Error("unexpected path: " + path);
}};
const out = await S.fetchPaletteChats(getFn);
const ranked = P.rank("gor", out, (c) => c.title, 5);
process.stdout.write(JSON.stringify({{
  calls, out, top: ranked.length ? ranked[0].item.title : null,
}}));
"""
    proc = subprocess.run(  # noqa: S603
        [NODE, "--input-type=module", "-e", src],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node xatosi: {proc.stderr[-2000:]}")
    result = json.loads(proc.stdout)
    assert result["calls"] == ["/api/me", "/api/accounts/9/chats?limit=1000"]
    assert result["out"] == rows
    assert result["top"] == "Gorizont"


def test_uz_dates_are_formatted_manually_never_negative() -> None:
    """H2: Chromium'ning ba'zi ICU to'plamlarida `uz-UZ` uchun oy/kun nomi
    yo'q (`Intl` "M09 15, Tue" qaytaradi) — `fmt.js` `web.fmt.*`
    lug'atidan qo'lda tuzatadi. Nisbiy vaqt hech qachon manfiy chiqmaydi.
    """
    assert NODE is not None
    en_to_uz = {
        "Monday": "Dushanba",
        "Tuesday": "Seshanba",
        "Wednesday": "Chorshanba",
        "Thursday": "Payshanba",
        "Friday": "Juma",
        "Saturday": "Shanba",
        "Sunday": "Yakshanba",
    }
    weekday_uz = en_to_uz[date(2026, 9, 15).strftime("%A")]
    strings = {
        "web.fmt.now": "hozir",
        "web.fmt.yesterday": "kecha",
        "web.fmt.rel_minutes": "{n} daqiqa oldin",
        "web.fmt.rel_hours": "{n} soat oldin",
        "web.fmt.rel_days": "{n} kun oldin",
        "web.fmt.rel_weeks": "{n} hafta oldin",
        "web.fmt.rel_months": "{n} oy oldin",
        "web.fmt.rel_years": "{n} yil oldin",
        "web.fmt.month_short.9": "sen",
        "web.fmt.month_full.9": "sentyabr",
        f"web.fmt.weekday_full.{date(2026, 9, 15).isoweekday() % 7}": weekday_uz,
    }
    src = f"""
globalThis.window = {{ TGAI: {{ lang: "uz", strings: {json.dumps(strings)} }} }};
const F = await import({json.dumps(FMT_JS.as_uri())});
const d = new Date(2026, 8, 15, 14, 32);   // 15-sen-2026, mahalliy vaqt
const fixedNow = d.getTime();
const fourWeeksAgo = new Date(fixedNow - 28 * 86400 * 1000);
process.stdout.write(JSON.stringify({{
  date: F.date(d),
  dateTime: F.dateTime(d),
  weekdayDate: F.weekdayDate(d),
  relWeeks: F.relTime(fourWeeksAgo, fixedNow),
  relNow: F.relTime(new Date(fixedNow - 5000), fixedNow),
  usdZero: F.usd(0),
  usdTiny: F.usd(0.004),
  usdNormal: F.usd(1.5),
}}));
"""
    proc = subprocess.run(  # noqa: S603
        [NODE, "--input-type=module", "-e", src],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node xatosi: {proc.stderr[-2000:]}")
    out = json.loads(proc.stdout)
    assert out["date"] == "15 sen 2026"
    assert out["dateTime"] == "15 sen, 14:32"
    assert out["weekdayDate"] == f"{weekday_uz}, 15 sentyabr"
    assert out["relWeeks"] == "4 hafta oldin"
    assert "-" not in out["relWeeks"]
    assert out["relNow"] == "hozir"
    assert out["usdZero"] == "$0.00"
    assert out["usdTiny"] == "$0.0040"
    assert out["usdNormal"] == "$1.50"


def test_every_js_file_parses() -> None:
    """`node --check` — har bir fayl (klassik ham, modul ham)."""
    assert NODE is not None
    files = files_js(STATIC)
    assert len(files) >= 15, "fayllar topilmadi — yo'l o'zgarganmi?"
    bad: list[str] = []
    for path in files:
        proc = subprocess.run(  # noqa: S603
            [NODE, "--check", str(path)], capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            bad.append(f"{path.name}: {proc.stderr.splitlines()[:3]}")
    assert not bad, bad


def test_sprite_has_every_icon_the_templates_use() -> None:
    """Shablon/JS `#name` bilan so'ragan har bir ikonka sprite'da bor."""
    import re

    sprite = (STATIC / "icons.svg").read_text(encoding="utf-8")
    have = set(re.findall(r'<symbol id="([a-z0-9-]+)"', sprite))
    assert len(have) >= 60

    root = Path(__file__).resolve().parents[1] / "src/app/web"
    used: set[str] = set()
    for path in list((root / "templates").glob("*.html")) + files_js(STATIC):
        text = path.read_text(encoding="utf-8")
        used |= set(re.findall(r'm\.icon\("([a-z0-9-]+)"', text))
        used |= set(re.findall(r'ico\("([a-z0-9-]+)"', text))
        used |= set(re.findall(r"icons\.svg#([a-z0-9-]+)", text))
    assert used, "hech qanday ikonka ishlatilmayapti — regex eskirganmi?"
    assert not (used - have), f"sprite'da yo'q ikonkalar: {sorted(used - have)}"


def files_js(base: Path) -> list[Path]:
    """Barcha frontend skriptlari: ildiz (klassik) · `lib/` · `pages/` (ui/R2).

    `pages/` rekursiv — 2-bosqichda sahifa modullari kichik fayllarga
    bo'linadi (`pages/ask/*.js`), ular ham `node --check` dan o'tishi kerak.
    """
    return (
        sorted(base.glob("*.js"))
        + sorted((base / "lib").glob("*.js"))
        + sorted((base / "pages").rglob("*.js"))
    )


# ─── ui/R2a: `/chats` sof funksiyalari ───────────────────────────────────────

CHATS_JS = STATIC / "pages/chats.js"


def _chats_module(script: str, payload: object) -> object:
    assert NODE is not None
    src = (
        f"const M = await import({json.dumps(CHATS_JS.as_uri())});\n"
        "const { readFileSync } = await import('node:fs');\n"
        "const INPUT = JSON.parse(readFileSync(0, 'utf8'));\n"
        f"{script}\n"
    )
    proc = subprocess.run(  # noqa: S603
        [NODE, "--input-type=module", "-e", src],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node xatosi: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def test_spark_path_builds_a_polyline_within_the_viewbox() -> None:
    """M3: qator sparklayni `d` atributi — DOM'siz, sof funksiya."""
    script = "process.stdout.write(JSON.stringify(INPUT.map((v) => M.sparkPath(v))));"
    cases = [
        [],
        [0, 0, 0],
        [1, 5, 2, 8, 0],
        [3],
    ]
    out = _chats_module(script, cases)
    assert out[0] == ""  # bo'sh — chizig'i yo'q
    assert out[1].startswith("M") and "L" in out[1]
    assert out[2].startswith("M0.00,")  # birinchi nuqta x=0 dan boshlanadi
    assert out[2].count("L") == 4  # 5 nuqta = 1 M + 4 L
    assert out[3].startswith("M24.00,")  # bitta nuqta — markazga (48/2)


# ─── ui/R2b: `/ask` sof funksiyalari ─────────────────────────────────────────
#
# `deep.js::citeParts` va `state.js::safeHref` — DOM'siz, shuning uchun
# node'da to'g'ridan-to'g'ri import qilinadi. Ikkalasi ham xavfsizlik
# chegarasida turadi: birinchisi javob matnidagi sitata tokenlarini
# ajratadi (HTML qayta parse qilinmaydi), ikkinchisi `href` ga faqat
# http(s)/tg sxemasini o'tkazadi.

ASK = STATIC / "pages/ask"


def _ask_module(module: str, script: str, payload: object) -> object:
    assert NODE is not None
    src = (
        f"const M = await import({json.dumps((ASK / module).as_uri())});\n"
        "const { readFileSync } = await import('node:fs');\n"
        "const INPUT = JSON.parse(readFileSync(0, 'utf8'));\n"
        f"{script}\n"
    )
    proc = subprocess.run(  # noqa: S603
        [NODE, "--input-type=module", "-e", src],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node xatosi: {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def test_citation_tokens_split_into_chips() -> None:
    """`#c<chat>m<msg>` / `#x<n>` → chip; qolgan hamma narsa matn bo'lib qoladi."""
    script = "process.stdout.write(JSON.stringify(INPUT.map((s) => M.citeParts(s))));"
    cases = [
        "To'lov kelishildi #c12m345 va tasdiqlandi.",
        "Ikkita manba: #c1m2 #x7 oxiri",
        "Sitata yo'q — oddiy matn #hashtag va #123",
        "",
        "#c99999999999999m1",  # 14 raqam — token emas (qopqoq 12)
    ]
    out = _ask_module("deep.js", script, cases)

    first = out[0]
    assert [p.get("token") for p in first if "token" in p] == ["c12m345"]
    assert first[0]["text"].startswith("To'lov")
    hit = next(p for p in first if p.get("token"))
    assert hit["chatId"] == 12 and hit["msgId"] == 345

    tokens = [p["token"] for p in out[1] if "token" in p]
    assert tokens == ["c1m2", "x7"]
    assert next(p for p in out[1] if p.get("token") == "x7")["chatId"] is None

    assert all("token" not in p for p in out[2]), "hashtag sitata bo'lib qolmasin"
    assert out[3] == []
    assert all("token" not in p for p in out[4]), "juda uzun raqam token emas"


def test_cite_parts_rebuild_the_original_text() -> None:
    """Bo'laklar yo'qotmasdan qayta yig'ilishi kerak — matn buzilmaydi."""
    script = """
const out = INPUT.map((s) => M.citeParts(s).map((p) => p.text ?? p.raw).join(""));
process.stdout.write(JSON.stringify(out));
"""
    cases = ["a #c1m2 b #x3", "#c1m2", "hech narsa", "  bo'sh  "]
    assert _ask_module("deep.js", script, cases) == cases


def test_safe_href_allows_only_http_and_tg() -> None:
    """`javascript:` va `data:` hech qachon `href` ga tushmaydi."""
    script = "process.stdout.write(JSON.stringify(INPUT.map((u) => M.safeHref(u))));"
    cases = [
        "https://t.me/chan/12",
        "http://example.org",
        "tg://resolve?domain=x",
        "HTTPS://T.ME/x",
        "javascript:alert(1)",
        "  javascript:alert(1)",
        "\njavascript:alert(1)",
        "data:text/html,<script>",
        "//evil.example",
        "/api/images/abc",
        "",
        None,
    ]
    out = _ask_module("state.js", script, cases)
    assert out[:4] == [
        "https://t.me/chan/12",
        "http://example.org",
        "tg://resolve?domain=x",
        "HTTPS://T.ME/x",
    ]
    assert out[4:] == [None] * (len(cases) - 4)


def test_page_modules_only_use_icons_that_exist() -> None:
    """Sahifa modullari `icon("x")` / `iconButton("x", …)` bilan chaqiradi.

    Eski test faqat `m.icon(...)` va `ico(...)` naqshlarini biladi; ES
    modullar `lib/icons.js` dagi `icon`/`iconButton` ni ishlatadi, ya'ni
    yo'q ikonka jimgina bo'sh `<use>` bo'lib qolardi.
    """
    import re

    sprite = (STATIC / "icons.svg").read_text(encoding="utf-8")
    have = set(re.findall(r'<symbol id="([a-z0-9-]+)"', sprite))

    used: set[str] = set()
    for path in sorted((STATIC / "pages").rglob("*.js")):
        text = path.read_text(encoding="utf-8")
        used |= set(re.findall(r'\bicon(?:Button)?\(\s*"([a-z0-9-]+)"', text))
    assert used, "sahifa modullarida ikonka ishlatilmayapti — regex eskirganmi?"
    assert not (used - have), f"sprite'da yo'q ikonkalar: {sorted(used - have)}"
