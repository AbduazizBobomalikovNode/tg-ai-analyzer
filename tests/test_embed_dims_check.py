"""F3/L10 — `EMBED_DIMS` ↔ DB ustuni mosligi startupda tekshiriladi.

`EMBED_DIMS` provider'ga `output_dimensionality` bo'lib ketadi, DB ustuni esa
`Vector(embed_dim)` bilan **migratsiya vaqtida** qurilgan. Ikkalasi ajralsa
har embedding INSERT'i `expected N dimensions, not M` bilan yiqiladi — ya'ni
indekslash butunlay to'xtaydi, buni esa faqat cron loglaridan bilish mumkin
edi. Shuning uchun tekshiruv `python -m app.main check` da (xato) va
api/worker startupida (ogohlantirish).
"""

from __future__ import annotations

import inspect

import pytest

from app.config import embed_dims_mismatch, get_settings


def _reload(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()


def test_matching_dims_report_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload(monkeypatch, EMBED_DIMS="0")  # 0 = DB ustuni bilan bir xil
    try:
        s = get_settings()
        assert s.embed_dims == s.embed_dim
        assert embed_dims_mismatch(s) == ""
    finally:
        monkeypatch.delenv("EMBED_DIMS", raising=False)
        get_settings.cache_clear()


def test_mismatched_dims_report_an_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload(monkeypatch, EMBED_DIMS="1536")
    try:
        s = get_settings()
        msg = embed_dims_mismatch(s)
        assert "1536" in msg and str(s.embed_dim) in msg
        # Nima qilish kerakligi aytiladi — "migratsiya" so'zisiz xabar foydasiz.
        assert "migratsiya" in msg
    finally:
        monkeypatch.delenv("EMBED_DIMS", raising=False)
        get_settings.cache_clear()


def test_every_entrypoint_is_wired_to_the_check() -> None:
    """Uch kirish nuqtasi ham tekshiruvni chaqiradi.

    `check` uni **xato** qiladi (chiqish kodi 1), api/worker esa
    **ogohlantiradi** — jarayon ko'tarilsin, lekin jim qolmasin. Bu yerda
    faqat simlanish tekshiriladi; mantiqning o'zi yuqoridagi sof
    funksiya testlarida.
    """
    import importlib

    from app import main as M

    check_src = inspect.getsource(M._check)
    assert "embed_dims_mismatch" in check_src
    assert "check.embed_dims" in check_src and "ok = False" in check_src

    for module, attr, event in (
        ("app.web.main", "lifespan", "web.embed_dims_mismatch"),
        ("app.worker.settings", "startup", "worker.embed_dims_mismatch"),
    ):
        src = inspect.getsource(getattr(importlib.import_module(module), attr))
        assert "embed_dims_mismatch" in src
        assert f'log.warning("{event}"' in src
