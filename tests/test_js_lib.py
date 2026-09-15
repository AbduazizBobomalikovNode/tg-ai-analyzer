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
from pathlib import Path

import pytest

from app.services.textnorm import uz_fold

STATIC = Path(__file__).resolve().parents[1] / "src/app/web/static"
PALETTE = STATIC / "lib/palette.js"
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
        f'const P = await import({json.dumps(PALETTE.as_uri())});\n'
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
        ["ozb", "O\u02bbzbekiston yangiliklari"],   # ketma-ketlik + so'z boshi
        ["oyan", "O\u02bbzbekiston yangiliklari"],  # tarqoq ketma-ketlik
        ["zzz", "O\u02bbzbekiston"],                # mos emas
        ["ўзб", "O\u02bbzbekiston"],                # kirill so'rov, lotin nomzod
        ["", "har qanday"],                    # bo'sh so'rov — hamma mos
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


def test_every_js_file_parses() -> None:
    """`node --check` — har bir fayl (klassik ham, modul ham)."""
    assert NODE is not None
    files = sorted(STATIC.glob("*.js")) + sorted((STATIC / "lib").glob("*.js"))
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
        used |= set(re.findall(r'icons\.svg#([a-z0-9-]+)', text))
    assert used, "hech qanday ikonka ishlatilmayapti — regex eskirganmi?"
    assert not (used - have), f"sprite'da yo'q ikonkalar: {sorted(used - have)}"


def files_js(base: Path) -> list[Path]:
    return sorted(base.glob("*.js")) + sorted((base / "lib").glob("*.js"))
