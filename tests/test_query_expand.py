"""Ko'p tilli so'rov kengaytirish — parse, qopqoqlar, kesh kaliti, degradatsiya.

Qo'riqlaydi:
  * bitta arzon (`Task.ROUTE`) chaqiruv, `json_mode`, konvertsiz (savol —
    foydalanuvchining o'z matni, begona kontent emas);
  * so'rov 300 belgidan uzun bo'lsa kesiladi;
  * variantlar soni qopqoqlari (ro'yxatda 4, jami 12);
  * kesh kaliti `uz_fold` fazosida — apostrof/registr/kirill farqi bir kalit;
  * LLM xatosi yoki buzuq JSON → **kengaytirishsiz** davom etish.
"""

from __future__ import annotations

from typing import Any

from app.llm import ChatResult, LLMError, Msg, Task, Usage
from app.services import query_expand as QX

_GOOD = (
    '{"uz_lat": ["tolov", "narx"], "uz_cyr": ["тўлов"], "ru": ["оплата"], '
    '"en": ["payment"], "synonyms": ["pul"], "slang": ["baks"]}'
)


class _FakeLLM:
    def __init__(self, text: str = _GOOD, *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[tuple[Task, list[Msg], dict[str, Any]]] = []

    async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
        self.calls.append((task, messages, kw))
        if self.fail:
            raise LLMError("test", "boom")
        return ChatResult(self.text, [], "cheap", "test", Usage(10, 5))


# ─── parse ───────────────────────────────────────────────────────────────────


def test_parse_expansion_reads_all_fields() -> None:
    exp = QX.parse_expansion(_GOOD)
    assert exp.uz_lat == ["tolov", "narx"]
    assert exp.ru == ["оплата"] and exp.en == ["payment"]
    assert not exp.is_empty


def test_parse_expansion_accepts_fence_and_prose() -> None:
    exp = QX.parse_expansion(f"Sure:\n```json\n{_GOOD}\n```\ndone")
    assert exp.uz_lat == ["tolov", "narx"]


def test_parse_expansion_garbage_is_empty_not_error() -> None:
    for raw in ("", "no json", "[1,2]", '{"uz_lat": "not-a-list"}', "{oops"):
        assert QX.parse_expansion(raw).is_empty


def test_parse_expansion_drops_junk_items_and_caps_list() -> None:
    exp = QX.parse_expansion(
        '{"uz_lat": ["a", 5, null, "bir", "ikki", "uch", "tort", "besh", "bir"]}'
    )
    # "a" — 2 belgidan qisqa, 5/null — satr emas, takror tushib qoladi
    assert exp.uz_lat == ["bir", "ikki", "uch", "tort"]
    assert len(exp.uz_lat) == QX.MAX_PER_LIST


def test_total_variants_are_capped() -> None:
    many = {f: [f"{f}{i}" for i in range(4)] for f in QX._FIELDS}
    exp = QX.from_dict(many)
    assert len(exp.all_terms()) == QX.MAX_TOTAL_VARIANTS == 12


def test_script_split_for_channels() -> None:
    exp = QX.parse_expansion(_GOOD)
    assert "оплата" in exp.cyrillic_terms() and "тўлов" in exp.cyrillic_terms()
    assert "payment" in exp.latin_terms() and "оплата" not in exp.latin_terms()


# ─── kesh kaliti ─────────────────────────────────────────────────────────────


def test_cache_key_is_fold_stable() -> None:
    a = QX.cache_key("To'lov muddati")
    assert a == QX.cache_key("tolov  muddati") == QX.cache_key("TO`LOV MUDDATI")
    assert a.startswith(QX.CACHE_PREFIX)
    assert a != QX.cache_key("narx muddati")


# ─── chaqiruv ────────────────────────────────────────────────────────────────


async def test_expand_uses_cheap_task_json_mode_and_no_envelope() -> None:
    llm = _FakeLLM()
    exp = await QX.expand("to'lov haqida nima kelishgandik?", llm=llm, cache=False)  # type: ignore[arg-type]
    assert not exp.is_empty
    (task, msgs, kw) = llm.calls[0]
    assert task is Task.ROUTE and kw.get("json_mode") is True
    assert len(llm.calls) == 1  # bitta chaqiruv, har variant uchun emas
    # savol — foydalanuvchining o'z matni: konvert yo'q, prompt system'da
    assert "<untrusted_data" not in msgs[-1].content
    assert msgs[-1].content == "to'lov haqida nima kelishgandik?"


async def test_expand_clips_long_query() -> None:
    llm = _FakeLLM()
    await QX.expand("x" * 900, llm=llm, cache=False)  # type: ignore[arg-type]
    assert len(llm.calls[0][1][-1].content) == QX.MAX_QUERY_CHARS


async def test_expand_degrades_on_llm_error() -> None:
    llm = _FakeLLM(fail=True)
    exp = await QX.expand("to'lov", llm=llm, cache=False)  # type: ignore[arg-type]
    assert exp.is_empty and exp.all_terms() == []


async def test_expand_degrades_on_garbage_json() -> None:
    llm = _FakeLLM(text="I cannot help with that")
    assert (await QX.expand("to'lov", llm=llm, cache=False)).is_empty  # type: ignore[arg-type]


async def test_expand_skips_trivial_query_without_calling_llm() -> None:
    llm = _FakeLLM()
    assert (await QX.expand("hi", llm=llm, cache=False)).is_empty  # type: ignore[arg-type]
    assert llm.calls == []


async def test_expand_cache_hit_avoids_llm_call(monkeypatch) -> None:
    store: dict[str, str] = {}

    async def fake_get(key: str) -> QX.Expansion | None:
        raw = store.get(key)
        return QX.parse_expansion(raw) if raw else None

    async def fake_set(key: str, exp: QX.Expansion) -> None:
        store[key] = _GOOD

    monkeypatch.setattr(QX, "_cache_get", fake_get)
    monkeypatch.setattr(QX, "_cache_set", fake_set)
    llm = _FakeLLM()
    first = await QX.expand("to'lov muddati", llm=llm)  # type: ignore[arg-type]
    second = await QX.expand("TOLOV  MUDDATI", llm=llm)  # type: ignore[arg-type]
    assert len(llm.calls) == 1  # ikkinchisi keshdan (fold kaliti bir xil)
    assert first.uz_lat == second.uz_lat


async def test_expand_survives_broken_redis(monkeypatch) -> None:
    async def boom(*_: object, **__: object) -> None:
        raise RuntimeError("redis yo'q")

    monkeypatch.setattr(QX, "_cache_get", boom)
    monkeypatch.setattr(QX, "_cache_set", boom)
    llm = _FakeLLM()
    try:
        exp = await QX.expand("to'lov muddati", llm=llm, cache=False)  # type: ignore[arg-type]
    except RuntimeError:  # pragma: no cover — kesh o'chirilganda tegilmasligi kerak
        raise AssertionError("cache=False bo'lsa Redis'ga borilmaydi") from None
    assert not exp.is_empty
