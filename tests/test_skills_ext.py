"""`skills_ext` — katalog kengaytmasi va `send_result` (topib → yuborish taklifi).

Qo'riqlaydi:
  * `register` beshta ko'nikma (+ `cancel_reminder`) qo'shadi, yadroni almashtirmaydi;
  * hammasi `min_mode=fast`; karantin **faqat** `send_result`; to'rt yozish
    amali har rejimda tasdiq so'raydi; katalog tavsifi uz+en, misol bilan;
  * `send_result`: niyat/manzil bo'lmasa taklif yo'q; matn `facts` / joriy
    sintez / oldingi javobdan; sitata tokeni va qamrov izohi yuborilmaydi;
    taklif **yagona** yo'ldan (`write_tools.propose_or_execute`), bitta;
  * planner uchun `summary` da matn yo'q (faqat bayroq/sanoq).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.services import deep_search as DS
from app.services import output_formats as OF
from app.services import skills as SK
from app.services import skills_ext as SX
from app.services import write_tools as W
from app.services.compute import Fact

NEW = {
    "format_output",
    "create_alert",
    "set_briefing",
    "remind",
    "cancel_reminder",
    "send_result",
    "trends",
    "extract",
}
WRITES = {"create_alert", "set_briefing", "remind", "cancel_reminder", "send_result"}
#: Begona xabar matnini ko'radi — chiqishi planner'ga tushmaydi.
QUARANTINED = {"send_result", "trends", "extract"}


def test_register_adds_skills_without_touching_core() -> None:
    core = {n: s for n, s in SK.SKILLS.items() if n not in NEW}
    reg: dict[str, SK.Skill] = dict(core)
    SX.register(reg)
    assert set(reg) - set(core) == NEW
    assert all(reg[n] is core[n] for n in core)
    # yadro nomi bilan to'qnashuv bo'lsa — yadro qoladi
    clash = {"format_output": core["stats"]}
    SX.register(clash)
    assert clash["format_output"] is core["stats"]


def test_hook_registered_extra_skills() -> None:
    """Orkestrator ulagichi (`skills.register_extra`) bo'lsa — hammasi registrda."""
    if not hasattr(SK, "register_extra"):
        pytest.skip("skills.register_extra hali yo'q")
    assert NEW <= set(SK.SKILLS)
    assert "send_result" in SK.QUARANTINED_SKILLS
    text = SK.catalog_text()
    for name in NEW:
        assert f"- {name}(" in text


def test_skill_flags() -> None:
    reg = {s.name: s for s in SX.ext_skills()}
    assert set(reg) == NEW
    for name, s in reg.items():
        assert s.min_mode == "fast" and s.allowed("fast")
        assert s.quarantined is (name in QUARANTINED)
        for mode in ("fast", "deep", "exhaustive"):
            assert s.needs_confirm(mode) is (name in WRITES)
        assert s.description_uz and s.description_en
        assert "Misol" in s.description_uz and "Example" in s.description_en
        assert s.est_cost_usd() < 0.05
    assert {n for n, s in reg.items() if s.terminal} == WRITES - {"send_result"}
    # send_result zanjirda (terminal emas) — oxirgi qadam, o'zi to'xtaydi
    assert not reg["send_result"].terminal and not reg["format_output"].terminal


def test_coerce_args_follow_schema() -> None:
    s = {x.name: x for x in SX.ext_skills()}
    got = SK.coerce_args(
        s["send_result"], {"chat": "Jamoa", "text_from": "facts", "chat_id": "12", "x": 1}
    )
    assert got == {"chat": "Jamoa", "text_from": "facts", "chat_id": 12}
    got = SK.coerce_args(s["remind"], {"when": "ertaga 10", "job_ids": [3, "x", 4]})
    assert got == {"when": "ertaga 10", "job_ids": [3, 4]}


# ─── send_result ─────────────────────────────────────────────────────────────


class _Session:
    def __init__(self, prior: str | None = None) -> None:
        self.added: list[Any] = []
        self.prior = prior
        self._seq = 300

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            self._seq += 1
            obj.id = self._seq

    async def flush(self) -> None: ...

    async def execute(self, *a: Any, **k: Any) -> Any:
        row = (self.prior,) if self.prior is not None else None
        return SimpleNamespace(first=lambda: row)


class _Factory:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __call__(self) -> _Factory:
        return self

    async def __aenter__(self) -> _Session:
        return self.session

    async def __aexit__(self, *a: Any) -> None:
        return None


def _ctx(question: str, session: _Session, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "raw_question": question,
        "locale": "uz",
        "account_id": 5,
        "user_id": 1,
        "conversation_id": 44,
        "run_id": None,
        "action_ids": [],
        "selected": [],
        "chats_by_id": {12: SimpleNamespace(id=12, title="Jamoa", type="group", link_prefix="")},
        "facts": {},
        "claims": [],
        "ranked": [],
        "clusters": [],
        "history": [],
        "plan": DS.DeepPlan(),
        "tz": "Asia/Tashkent",
        "session_factory": _Factory(session),
    }
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def proposals(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def fake(session: Any, tctx: Any, *, run_id: int, tool: str, args: dict[str, Any]):
        calls.append(
            {"run_id": run_id, "tool": tool, "args": dict(args), "pinned": tctx.pinned_chat_id}
        )
        return SimpleNamespace(ok=True, text="proposed", meta={"action_id": 77, "proposed": True})

    monkeypatch.setattr(W, "propose_or_execute", fake)
    return calls


async def test_send_result_needs_intent_and_target(proposals: list[dict[str, Any]]) -> None:
    s = _Session(prior="Xulosa")
    res = await SX._send_result(_ctx("bu xulosa qanday?", s), {"chat_id": 12})
    assert res.stop and res.summary["reason"] == "no_intent"
    res = await SX._send_result(_ctx("shu xulosani guruhga tashla", s), {})
    assert res.summary["reason"] == "no_target"
    assert proposals == [] and s.added == []


async def test_send_result_prior_answer_is_cleaned(proposals: list[dict[str, Any]]) -> None:
    prior = (
        "IT ta'lim bo'yicha 3 ta e'lon bor [#c12m345], narxlar 2 mln dan #x2.\n\n"
        '```chart\n{"type":"bar"}\n```\n\n---\n**Qamrov** — chatlar: 3/10'
    )
    s = _Session(prior=prior)
    ctx = _ctx("shu xulosani 'Jamoa' guruhiga tashla", s)
    res = await SX._send_result(ctx, {"chat_id": 12, "text_from": "answer"})
    [call] = proposals
    assert call["tool"] == "send_message" and call["pinned"] is None
    text = call["args"]["text"]
    assert call["args"]["chat"] == "12"
    assert text == "IT ta'lim bo'yicha 3 ta e'lon bor, narxlar 2 mln dan."
    assert "#c12m345" not in res.answer and "Qamrov" not in text
    assert "#77" in res.answer and "Jamoa" in res.answer and "yuborilmadi" in res.answer
    assert ctx.action_ids == [77] and res.stop
    assert set(res.summary) == {"proposed", "intent", "source"}  # matn yo'q


async def test_send_result_from_facts(proposals: list[dict[str, Any]]) -> None:
    s = _Session()
    ctx = _ctx(
        "kechagi statistikani Jamoa guruhiga yubor",
        s,
        facts={
            "total": Fact("total", 120, label="messages in the window"),
            "period": Fact("period", "kecha"),
        },
        claims=[OF.wrap_block("| a | 1 |"), OF.wrap_block('```chart\n{"type":"bar"}\n```')],
    )
    await SX._send_result(ctx, {"chat": "Jamoa", "text_from": "facts"})
    [call] = proposals
    assert call["args"] == {"chat": "Jamoa", "text": "• messages in the window: 120\n\n| a | 1 |"}


async def test_send_result_synthesizes_current_evidence_once(
    proposals: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_synth(meter: Any, question: str, evidence: Any, clusters: Any, **kw: Any):
        calls.append(kw)
        return "Topildi: 2 ta e'lon #c12m9.", [], None

    monkeypatch.setattr(DS, "synthesize", fake_synth)
    monkeypatch.setattr(DS, "to_evidence", lambda *a, **k: [])
    taint_seen: list[Any] = []
    ctx = _ctx(
        "ijara haqida topib Jamoa guruhiga yubor",
        _Session(),
        ranked=[object()],
        meter=None,
        taint=SimpleNamespace(add=taint_seen.append),
        server_out=DS.ServerOutcome(),
    )
    res = await SX._send_result(ctx, {"chat_id": 12})
    assert len(calls) == 1 and "send_result" in calls[0]["mode_note"]
    [call] = proposals
    assert call["args"]["text"] == "Topildi: 2 ta e'lon."
    assert res.answer.startswith("Topildi: 2 ta e'lon.") and taint_seen == [[]]


async def test_send_result_empty_is_refused(proposals: list[dict[str, Any]]) -> None:
    res = await SX._send_result(_ctx("Jamoaga yubor", _Session()), {"chat_id": 12})
    assert res.summary["reason"] == "empty" and proposals == []


def test_clean_for_send() -> None:
    assert SX.clean_for_send("A #c1m2 va [#x3] B\n\n\n\nC") == "A va B\n\nC"
    assert len(SX.clean_for_send("x" * 10_000)) == SX.SEND_TEXT_MAX


async def test_send_result_empty_turn_result_is_not_proposed(
    proposals: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Jonli (conv 385): kecha guruhda 0 xabar — "topilmadi" matni Otabekka taklif qilinardi.
    async def fake_synth(meter: Any, question: str, evidence: Any, clusters: Any, **kw: Any):
        return "Kecha guruhda xabar topilmadi.", [], None

    monkeypatch.setattr(DS, "synthesize", fake_synth)
    monkeypatch.setattr(DS, "to_evidence", lambda *a, **k: [])
    ctx = _ctx(
        "kecha Jamoa guruhida nima bo'lganini yozib, Otabekka yubor",
        _Session(),
        facts={"chat_total": Fact("chat_total", 0, label="messages in the chat(s)")},
        claims=["chat Jamoa: 0 messages"],
        meter=None,
        taint=SimpleNamespace(add=lambda *_: None),
        server_out=DS.ServerOutcome(),
    )
    res = await SX._send_result(ctx, {"chat_id": 12})
    assert proposals == [] and res.stop
    assert res.summary == {"proposed": False, "intent": "send_result", "reason": "empty_result"}
    assert res.answer.startswith("Kecha guruhda xabar topilmadi.")
    assert "taklif qilinmadi" in res.answer


def test_turn_found_nothing() -> None:
    s = _Session()
    zero = {"chat_total": Fact("chat_total", 0)}
    assert SX.turn_found_nothing(_ctx("x", s, facts=zero))
    assert not SX.turn_found_nothing(_ctx("x", s, facts=zero, ranked=[object()]))
    assert not SX.turn_found_nothing(_ctx("x", s, facts={"total": Fact("total", 3)}))
    assert not SX.turn_found_nothing(_ctx("x", s))  # sanoq yo'q — oldingi javob manbasi


def test_analysis_skills_catalog_entries() -> None:
    """`trends` / `extract`: LLM'siz (0$), yozish emas, katalogda ikki misol bilan."""
    reg = {s.name: s for s in SX.ext_skills()}
    for name in ("trends", "extract"):
        s = reg[name]
        assert s.cost_usd == 0.0 and not s.confirm_modes and not s.terminal
        assert s.description_uz.count("→") >= 2 and s.description_en.count("→") >= 2
    assert set(reg["trends"].args) == {"chat_id", "scope", "period", "metric", "top_n", "compare"}
    got = SK.coerce_args(reg["trends"], {"metric": "reactions", "top_n": "5", "compare": True})
    assert got == {"metric": "reactions", "top_n": 5}
