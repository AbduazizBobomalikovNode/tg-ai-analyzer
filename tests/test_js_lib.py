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

import itertools
import json
import re
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


#: Top-level `import`/`export` — modul belgisi. Faqat qator boshida (ichki
#: matn/izohlardagi so'zlarni emas).
_MODULE_SYNTAX = re.compile(r"(?m)^\s*(?:import\s|export\s|export\{)")


def test_every_js_file_parses() -> None:
    """`node --check` — har bir fayl (klassik ham, modul ham).

    `lib/**` va `pages/**` doim `<script type="module">` orqali yuklanadi
    (yoki bir-biridan `import`/`export` qiladi); ildizdagi bir nechta fayl
    (`briefings.js`, `dashboard.js`) ham xuddi shunday. Muammo: plain
    `node --check FILE` — agar faylda top-level `import` bo'lsa-yu, u
    haqiqiy modul sifatida **to'liq** tekshirilmaydi (Node'ning CJS/ESM
    avto-aniqlashi import bor faylni "modulga o'xshaydi" deb jimgina
    o'tkazib yuboradi — sintaksisi butunlay buzilgan bo'lsa ham, masalan
    ochilmagan qavs — `rc=0`!). Import yo'q klassik skriptda esa
    modul-ko'lam xatolari sezilmaydi: bir xil nom bir marta `function`,
    keyin yana `const` bilan top-level'da e'lon qilinsa (`accountLabel` —
    I5) CommonJS skriptda jim o'tadi, ES modulda esa
    `SyntaxError: Identifier 'x' has already been declared` beradi.

    Shuning uchun `import`/`export` ishlatadigan har bir fayl (papkasidan
    qat'i nazar) **ES modul** sifatida tekshiriladi — `--input-type=module`,
    stdin orqali (fayl nomi/yo'li kerak emas, chunki `--check` importlarni
    yechmaydi, faqat sintaksisni tekshiradi — va bu yo'l real fayl yo'liga
    bog'liq CJS/ESM avto-aniqlash xatosidan xoli).
    """
    assert NODE is not None
    files = files_js(STATIC)
    assert len(files) >= 15, "fayllar topilmadi — yo'l o'zgarganmi?"
    bad: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if _MODULE_SYNTAX.search(text):
            proc = subprocess.run(  # noqa: S603
                [NODE, "--input-type=module", "--check"],
                input=text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
        else:
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


#: `/api/deep/stream` ning korpus rejimi (`services/scale_search.py`) uchun
#: namunaviy hodisa ketma-ketligi — shartnomadagi tartibda.
_SCALE_EVENTS: list[dict[str, object]] = [
    {
        "kind": "plan",
        "t": 0,
        "data": {
            "subqueries": ["a", "b"],
            "standalone_question": "To'lov qachon kelishildi?",
            "intent_follow_up": "deeper",
        },
    },
    {
        "kind": "estimate",
        "t": 500,
        "data": {
            "mode": "deep",
            "buckets": 186,
            "est_messages": 14200,
            "est_cost_usd": 0.07,
            "est_total_usd": 0.09,
            "ceiling_usd": 0.1,
            "weeks": 30,
            "range": {"from": "2026-01-01", "to": "2026-09-01"},
        },
    },
    {
        "kind": "buckets",
        "t": 900,
        "data": {
            "planned": 186,
            "chats": 24,
            "weeks": 30,
            "range": {"from": "2026-01-01", "to": "2026-09-01"},
        },
    },
    {"kind": "pull", "t": 1500, "data": {"bucket_i": 40, "buckets": 186, "candidates": 900}},
    {"kind": "pull", "t": 2500, "data": {"bucket_i": 186, "buckets": 186, "candidates": 2000}},
    {
        "kind": "filter",
        "t": 3500,
        "data": {
            "done": 1000,
            "total": 2000,
            "kept": 320,
            "keep_rate": 0.32,
            "cost_usd": 0.01,
            "cached": False,
        },
    },
    {
        "kind": "filter",
        "t": 4500,
        "data": {
            "done": 2000,
            "total": 2000,
            "kept": 640,
            "keep_rate": 0.32,
            "cost_usd": 0.02,
            "cached": False,
        },
    },
    {
        "kind": "select",
        "t": 5000,
        "data": {
            "evidence": 512,
            "chats": 18,
            "weeks": 26,
            "quota_trimmed": 128,
        },
    },
    {"kind": "map", "t": 6000, "data": {"chunk": 1, "chunks": 4, "claims": 12}},
    {"kind": "map", "t": 8000, "data": {"chunk": 4, "chunks": 4, "claims": 48}},
    {"kind": "reduce", "t": 9000, "data": {"claims_in": 48, "contradictions": 3}},
    {"kind": "token", "t": 9500, "data": {"text": "…"}},
    {
        "kind": "coverage",
        "t": 12000,
        "data": {
            "chats_searched": 18,
            "chats_total": 24,
            "messages_scanned": 2000,
            "buckets_scanned": 186,
            "buckets_planned": 186,
            "filtered": 2000,
            "kept": 640,
            "stopped_by": "saturation",
            "pull_stopped_by": "wall",
            "filter_stopped_by": "saturation",
            "scan_cost_usd": 0.02,
        },
    },
    {"kind": "done", "t": 12500, "data": {"citations": []}},
]


def test_stage_model_folds_a_scale_run() -> None:
    """`deep.js::stageModel` — hodisalar → bosqich holati va voronka sonlari.

    Bu reduktor sof: DOM ham, i18n ham yo'q, shuning uchun node'da
    to'g'ridan-to'g'ri o'lchanadi. U yuzaning **yagona haqiqati** — rail,
    progress barlar va qamrov paneli faqat shu modelning ko'rinishi, ya'ni
    kech kelgan yoki siqilgan hodisa (`pull`/`filter` ≤1/s) bosqichlarni
    chalkashtira olmaydi.
    """
    script = """
const m = M.stageModel(INPUT, { mode: "deep" });
const states = {};
for (const [k, v] of Object.entries(m.stages)) states[k] = v.state;
const elapsed = {};
for (const [k, v] of Object.entries(m.stages)) elapsed[k] = (v.endedAt ?? 0) - v.startedAt;
process.stdout.write(JSON.stringify({
  states, elapsed, funnel: m.funnel, stopped: m.stoppedBy,
  active: m.active, needsConfirm: m.needsConfirm,
  standalone: m.plan && m.plan.standalone_question,
  intent: m.plan && m.plan.intent_follow_up,
  estimate: m.estimate,
  coverage: m.coverage,
}));
"""
    out = _ask_module("deep.js", script, _SCALE_EVENTS)

    # Hamma bosqich yopildi, yozish `done` bilan tugadi.
    assert out["states"] == {
        "plan": "done",
        "estimate": "done",
        "buckets": "done",
        "pull": "done",
        "filter": "done",
        "select": "done",
        "map": "done",
        "reduce": "done",
        "write": "done",
    }
    assert out["active"] is None and out["needsConfirm"] is False

    f = out["funnel"]
    # Voronka: 2000 saralandi → 640 mos → 512 tahlilga (kvota 128 ni kesdi).
    assert (f["filtered"], f["filterTotal"], f["kept"]) == (2000, 2000, 640)
    assert f["keepRate"] == 0.32
    assert (f["selected"], f["trimmed"]) == (512, 128)
    assert (f["bucketsDone"], f["bucketsPlanned"]) == (186, 186)
    assert f["candidates"] == 2000
    assert (f["chunk"], f["chunks"], f["claims"]) == (4, 4, 48)
    assert (f["claimsIn"], f["contradictions"]) == (48, 3)
    assert f["scanCost"] == 0.02
    assert out["stopped"] == "saturation"

    # Reja kartasi uchun: qayta yozilgan savol va davom turi saqlanadi.
    assert out["standalone"] == "To'lov qachon kelishildi?"
    assert out["intent"] == "deeper"
    assert out["estimate"]["est_messages"] == 14200
    # `estimate` endi hafta soni, sana oralig'i va jami narxni ham olib keladi
    # (shartnoma: `weeks`, `range:{from,to}`, `est_total_usd`) — baho satri va
    # tor-oyna ogohlantirishi shulardan chiziladi.
    assert out["estimate"]["weeks"] == 30
    assert out["estimate"]["range"] == {"from": "2026-01-01", "to": "2026-09-01"}
    assert out["estimate"]["est_total_usd"] == 0.09

    # Qamrov endi tortish va saralashning **alohida** to'xtash sababini olib
    # keladi (`pull_stopped_by`/`filter_stopped_by`) — bu ikkisi farq qilganda
    # panel ikkalasini ham ko'rsatadi.
    assert out["coverage"]["pull_stopped_by"] == "wall"
    assert out["coverage"]["filter_stopped_by"] == "saturation"

    # Davomiylik: bosqich keyingisi kelganda yopiladi — `pull` 900→3500,
    # `filter` 3500→5000 (`select` kelganda).
    assert out["elapsed"]["pull"] == 2600
    assert out["elapsed"]["filter"] == 1500


def test_stage_model_handles_confirm_and_abort() -> None:
    """Tasdiq kutilayotgan `exhaustive` — bosqichlar `done` bo'lib qolmaydi.

    Server `estimate` dan keyin `done {needs_confirm:true}` yuboradi va
    javob yozilmaydi; UI shu holatda "tugadi" deb ko'rsatsa foydalanuvchi
    javobni behuda kutadi. `error` esa faol bosqichni `failed` qiladi.
    """
    script = """
const confirmRun = M.stageModel(INPUT.confirm, { mode: "exhaustive" });
const failRun = M.stageModel(INPUT.fail, { mode: "deep" });
const pick = (m) => {
  const s = {};
  for (const [k, v] of Object.entries(m.stages)) s[k] = v.state;
  return s;
};
process.stdout.write(JSON.stringify({
  confirmStates: pick(confirmRun),
  needsConfirm: confirmRun.needsConfirm,
  estimate: confirmRun.estimate,
  failStates: pick(failRun),
  error: failRun.error && failRun.error.code,
}));
"""
    payload = {
        "confirm": [
            {
                "kind": "estimate",
                "t": 0,
                "data": {
                    "mode": "exhaustive",
                    "buckets": 900,
                    "est_messages": 120000,
                    "est_cost_usd": 0.48,
                    "ceiling_usd": 0.5,
                },
            },
            {"kind": "done", "t": 10, "data": {"needs_confirm": True}},
        ],
        "fail": [
            {"kind": "estimate", "t": 0, "data": {"buckets": 4, "est_messages": 100}},
            {"kind": "buckets", "t": 5, "data": {"planned": 4}},
            {"kind": "pull", "t": 9, "data": {"bucket_i": 1, "buckets": 4, "candidates": 10}},
            {"kind": "error", "t": 20, "data": {"code": "chat.err.budget"}},
        ],
    }
    out = _ask_module("deep.js", script, payload)

    assert out["needsConfirm"] is True
    assert out["estimate"]["ceiling_usd"] == 0.5
    # Tasdiq kutilmoqda — hech bir bosqich "tugagan" emas.
    assert out["confirmStates"]["buckets"] == "pending"
    assert "write" not in out["confirmStates"]

    assert out["failStates"]["pull"] == "failed"
    assert out["failStates"]["buckets"] == "done"
    assert out["error"] == "chat.err.budget"


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


# ─── ui/R4: `/dashboard` grafik yordamchilari ────────────────────────────────
#
# `niceMax` / `tickValues` / `tickIndices` / `linePath` / `deltaPct` —
# DOM'siz sof funksiyalar, shuning uchun node'da to'g'ridan-to'g'ri
# o'lchanadi. Ular grafikning o'qini va yorliq zichligini belgilaydi:
# xato bo'lsa yorliqlar bir-birining ustiga chiqadi yoki o'q "3,7k" kabi
# o'qib bo'lmaydigan sonlar bilan to'ladi.

DASH_JS = STATIC / "pages/dashboard.js"


def _dash_module(script: str, payload: object) -> object:
    assert NODE is not None
    src = (
        f"const M = await import({json.dumps(DASH_JS.as_uri())});\n"
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


def test_nice_max_rounds_the_axis_to_a_readable_number() -> None:
    out = _dash_module(
        "process.stdout.write(JSON.stringify(INPUT.map((v) => M.niceMax(v))));",
        [0, -5, 1, 3, 7, 12, 23, 47, 99, 1234, 0.004],
    )
    assert out[0] == 1 and out[1] == 1  # nol/manfiy — bo'sh grafik ham chiziladi
    assert out == [1, 1, 1, 5, 10, 20, 25, 50, 100, 2000, 0.005]
    # Har doim qiymatdan kichik bo'lmasligi shart (ustun kesilib qolmasin).
    for value, top in zip([1, 3, 7, 12, 23, 47, 99, 1234], out[2:10], strict=True):
        assert top >= value


def test_tick_values_pick_a_readable_step_not_a_fixed_count() -> None:
    """Qadam "chiroyli" bo'lmasa o'q "0 · 1,7 · 3,3 · 5" ko'rinishida
    chiqadi — sonlar o'qilmaydi. Chiziqlar soni shunga moslashadi."""
    out = _dash_module(
        "process.stdout.write(JSON.stringify(INPUT.map(([m, o]) => M.tickValues(m, 4, o))));",
        [[7, {}], [5, {"integer": True}], [100, {"integer": True}], [500000, {}]],
    )
    for ticks in out:
        assert ticks[0] == 0
        assert ticks == sorted(ticks)
    assert out[0][-1] == 10 and 2.5 in out[0]  # 0 · 2,5 · 5 · 7,5 · 10
    assert out[1] == [0, 1, 2, 3, 4, 5]  # savollar — butun sonlar
    assert out[2] == [0, 25, 50, 75, 100]
    assert all(float(v).is_integer() for v in out[3])


def test_tick_indices_thin_labels_so_they_never_collide() -> None:
    """Yorliqlar orasidagi masofa **kamida** bitta yorliq kengligicha."""
    out = _dash_module(
        "process.stdout.write(JSON.stringify(INPUT.map(([n, w]) => M.tickIndices(n, w))));",
        [[0, 6], [7, 6], [30, 6], [30, 2], [90, 8]],
    )
    assert out[0] == []
    assert out[1] == [0, 2, 4, 6]  # 7 kun, 6 yorliq o'rni → har ikkinchisi
    assert len(out[2]) <= 6 and out[2][0] == 0
    assert out[3] == [0, 15]  # tor konteyner — ikkitagina yorliq
    for indices in out[1:]:
        # tartib o'sadi va qadam bir xil (yorliqlar "sakramaydi")
        steps = {b - a for a, b in itertools.pairwise(indices)}
        assert len(steps) <= 1, indices
        assert indices == sorted(indices)
    # "oxirgisi majburiy" yo'q: 30 kun / 6 yorliq → oxirgi indeks 29 emas
    assert 29 not in out[2] or len(out[2]) == 30


def test_line_path_breaks_where_a_day_has_no_measurement() -> None:
    """`null` kun — chiziq uziladi (nolga tushgandek ko'rinmasin)."""
    out = _dash_module(
        "process.stdout.write(JSON.stringify(INPUT.map((pts) => M.linePath(pts))));",
        [
            [],
            [{"x": 0, "y": 10}, {"x": 5, "y": 2}],
            [{"x": 0, "y": 1}, None, {"x": 9, "y": 3}],
        ],
    )
    assert out[0] == ""
    assert out[1] == "M0.00,10.00L5.00,2.00"
    assert out[2].count("M") == 2 and "L" not in out[2]


def test_delta_pct_is_null_without_a_previous_period() -> None:
    out = _dash_module(
        "process.stdout.write(JSON.stringify(INPUT.map(([a, b]) => M.deltaPct(a, b))));",
        [[10, 0], [10, 5], [5, 10], [0, 4], [7, 7]],
    )
    assert out[0] is None  # bo'lish yo'q — sahifa matn bilan aytadi
    assert out[1] == 1.0
    assert out[2] == -0.5
    assert out[3] == -1.0
    assert out[4] == 0.0


def test_place_popover_clamps_to_viewport_and_flips_above_when_no_room() -> None:
    """ⓘ popover: `left` doim `[8, innerWidth - width - 8]` ichida qoladi
    (mobil QA'da 57px chapga chiqib ketgan edi) va pastda joy tugasa
    tugma **ustiga** o'tadi (desktop QA'da KPI qiymatini yopib turardi)."""
    cases = [
        # markazda, hamma tomondan joy bor — pastda, markazlashtirilgan
        (
            {"left": 500, "top": 100, "right": 540, "bottom": 120, "width": 40, "height": 20},
            {"width": 200, "height": 80},
            {"width": 1200, "height": 800},
        ),
        # chap chetga yaqin — `left` 8px'dan kichik bo'lmasin
        (
            {"left": 10, "top": 100, "right": 50, "bottom": 120, "width": 40, "height": 20},
            {"width": 300, "height": 80},
            {"width": 400, "height": 800},
        ),
        # o'ng chetga yaqin — `left` o'ngdan 8px qoldirsin
        (
            {"left": 380, "top": 100, "right": 420, "bottom": 120, "width": 40, "height": 20},
            {"width": 300, "height": 80},
            {"width": 400, "height": 800},
        ),
        # ekran tagida — pastda joy yo'q, tepaga o'tadi
        (
            {"left": 500, "top": 750, "right": 540, "bottom": 770, "width": 40, "height": 20},
            {"width": 200, "height": 80},
            {"width": 1200, "height": 800},
        ),
        # popover viewport'dan balandroq — pastda qoladi, lekin `top` 8px'ga tortiladi
        (
            {"left": 100, "top": 50, "right": 140, "bottom": 70, "width": 40, "height": 20},
            {"width": 200, "height": 300},
            {"width": 1200, "height": 200},
        ),
    ]
    out = _dash_module(
        "process.stdout.write(JSON.stringify(INPUT.map(([a, p, v]) => M.placePopover(a, p, v))));",
        cases,
    )
    assert out[0] == {"left": 420, "top": 126, "placement": "below"}
    assert out[1] == {"left": 8, "top": 126, "placement": "below"}
    assert out[2] == {"left": 92, "top": 126, "placement": "below"}
    assert out[3] == {"left": 420, "top": 664, "placement": "above"}
    assert out[4] == {"left": 20, "top": 8, "placement": "below"}
    # Invariant: har holatda `left` viewport ichida — kartaning chap/o'ng
    # chetidan hech qachon chiqmasin (mobil QA'da aynan shu buzilgan edi).
    for (_anchor, pop, viewport), result in zip(cases, out, strict=True):
        assert 0 <= result["left"] <= viewport["width"] - pop["width"]
        assert result["top"] >= 0
        assert result["placement"] in ("above", "below")


def test_stage_model_follows_the_stages_event() -> None:
    """Reyk **shu turn** yuguradigan bosqichlardan quriladi (`stages` hodisasi).

    Qidiruvsiz turn ("salom", "nimalar qila olasan") `plan → write` beradi:
    baho kartasi ham, skan bosqichlari ham chizilmaydi. Hodisa kelmasa
    (eski server) rejim bo'yicha zaxira ro'yxat ishlaydi — eski xulq.
    """
    script = """
const direct = M.stageModel(INPUT.direct, { mode: "exhaustive" });
const late = M.stageModel(INPUT.late, { mode: "exhaustive" });
const pick = (m) => {
  const s = {};
  for (const [k, v] of Object.entries(m.stages)) s[k] = v.state;
  return s;
};
process.stdout.write(JSON.stringify({
  directStates: pick(direct),
  directVisible: direct.visible,
  isDirect: direct.direct,
  intent: direct.intent,
  estimate: direct.estimate,
  lateStates: pick(late),
  lateVisible: late.visible,
}));
"""
    payload = {
        # Qidiruvsiz turn: `stages` → `plan` → `token` → `done {direct}`.
        "direct": [
            {"kind": "stages", "t": 0, "data": {"stages": ["plan", "write"], "direct": True}},
            {"kind": "plan", "t": 5, "data": {"intent_follow_up": "meta", "subqueries": []}},
            {"kind": "token", "t": 20, "data": {"text": "Salom!"}},
            {"kind": "done", "t": 30, "data": {"direct": True, "intent": "meta"}},
        ],
        # Haqiqiy savol: ro'yxat reja **hodisasidan keyin** kengayadi —
        # navbatdagi bosqich (`estimate`) o'shanda yonadi.
        "late": [
            {"kind": "stages", "t": 0, "data": {"stages": ["plan"]}},
            {"kind": "plan", "t": 5, "data": {"subqueries": ["a", "b"]}},
            {
                "kind": "stages",
                "t": 6,
                "data": {
                    "stages": [
                        "plan",
                        "estimate",
                        "buckets",
                        "pull",
                        "filter",
                        "select",
                        "map",
                        "reduce",
                        "write",
                    ]
                },
            },
        ],
    }
    out = _ask_module("deep.js", script, payload)

    assert out["directVisible"] == ["plan", "write"]
    assert out["directStates"] == {"plan": "done", "write": "done"}
    assert out["isDirect"] is True and out["intent"] == "meta"
    assert out["estimate"] is None, "baho kartasi `estimate` hodisasisiz chizilmaydi"

    assert out["lateStates"]["plan"] == "done"
    assert out["lateStates"]["estimate"] == "active"
    assert len(out["lateVisible"]) == 9


def test_stage_model_stats_and_window_estimate_follow_the_planned_stages() -> None:
    """Sanoq yo'li (`plan → stats → write`) va oyna bahosi reja ro'yxatiga ergashadi.

    Jonli regressiya: oyna yo'lida `estimate` hodisasi qattiq yozilgan
    `buckets` bosqichini yoqardi va reykda "Rejalashtirish 12 062 bucket ·
    0 chat · 0 hafta" chiqardi — bu yo'lda bucket rejasi umuman yo'q.
    """
    script = """
const stats = M.stageModel(INPUT.stats, { mode: "exhaustive" });
const win = M.stageModel(INPUT.window, { mode: "exhaustive" });
const pick = (m) => {
  const s = {};
  for (const [k, v] of Object.entries(m.stages)) s[k] = v.state;
  return s;
};
process.stdout.write(JSON.stringify({
  statsStates: pick(stats), statsFunnel: stats.funnel,
  winStates: pick(win), winEstimate: win.estimate, winFunnel: win.funnel,
}));
"""
    payload = {
        "stats": [
            {"kind": "stages", "t": 0, "data": {"stages": ["plan", "stats", "write"]}},
            {"kind": "plan", "t": 5, "data": {"subqueries": []}},
            {
                "kind": "stats",
                "t": 9,
                "data": {"total": 4210, "chats": 87, "direction": "out", "top": 20},
            },
            {"kind": "token", "t": 20, "data": {"text": "1. Nurbek — 600"}},
            {"kind": "done", "t": 30, "data": {}},
        ],
        "window": [
            {
                "kind": "stages",
                "t": 0,
                "data": {"stages": ["plan", "estimate", "pull", "select", "map", "write"]},
            },
            {"kind": "plan", "t": 5, "data": {"subqueries": []}},
            {
                "kind": "estimate",
                "t": 8,
                "data": {
                    "mode": "exhaustive",
                    "buckets": 12062,
                    "buckets_total": 12062,
                    "chats": 214,
                    "weeks": 39,
                    "days": 268,
                    "grain": "week",
                    "est_messages": 667360,
                    "retrieval_kind": "window",
                },
            },
        ],
    }
    out = _ask_module("deep.js", script, payload)

    assert out["statsStates"] == {"plan": "done", "stats": "done", "write": "done"}
    assert out["statsFunnel"]["statsTotal"] == 4210
    assert out["statsFunnel"]["chats"] == 87
    assert out["statsFunnel"]["statsDirection"] == "out"

    assert "buckets" not in out["winStates"], out["winStates"]
    assert out["winStates"]["estimate"] == "done"
    assert out["winStates"]["pull"] == "active"
    assert out["winEstimate"]["weeks"] == 39
    assert out["winEstimate"]["grain"] == "week"
    assert out["winFunnel"]["chats"] == 214


def test_estimate_card_shows_in_deep_mode_only_when_it_costs_something() -> None:
    """`deep` da baho kartasi faqat jami > $0.05 bo'lsa; `exhaustive` da doim."""
    script = """
const out = INPUT.map(([est, mode]) => M.estimateCardVisible(est, mode));
process.stdout.write(JSON.stringify(out));
"""
    cases = [
        [{"mode": "deep", "est_total_usd": 0.064}, "deep"],
        [{"mode": "deep", "est_total_usd": 0.02}, "deep"],
        [{"mode": "exhaustive", "est_total_usd": 0.01}, "exhaustive"],
        [{"est_total_usd": 0.2}, "fast"],
        [None, "deep"],
    ]
    assert _ask_module("deep.js", script, cases) == [True, False, True, True, False]
