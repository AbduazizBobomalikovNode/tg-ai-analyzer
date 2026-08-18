"""7-bosqich: rasm generatsiya, auto-reply, scheduling — DB/Telegram/tarmoqsiz.

Qo'riqlaydi:
  * rasm: LLM baytlari → fayl + meta, kunlik limit, egalik (`path_for_send`), sniff
  * auto-reply: trigger mos kelishi, jim soatlar, SKIP, o'z xabarini o'tkazish,
    javob **taklif** (send_message + reply_to), soatlik byudjet
  * send_message taklifi image_id bilan (caption ≤1024), preview'da image_url
  * tool registry'da generate_image / list_scheduled_messages
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from app.config import get_settings
from app.llm import ChatResult, Msg, Task, Usage
from app.services import autoreply as AR
from app.services import images as IMG
from app.services import tools as T
from app.services import write_tools as W

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200


class _Session:
    def __init__(self, objects: dict[tuple[str, Any], Any] | None = None, count: int = 0) -> None:
        self.added: list[Any] = []
        self.objects = objects or {}
        self.count = count

    def add(self, obj: Any) -> None:
        self.added.append(obj)
        if getattr(obj, "id", None) is None:
            obj.id = 100 + len(self.added)

    async def flush(self) -> None: ...

    async def get(self, model: Any, key: Any) -> Any:
        return self.objects.get((model.__name__, key))

    async def execute(self, *a: Any, **k: Any) -> Any:
        c = self.count
        return SimpleNamespace(
            scalar_one=lambda: c,
            scalar_one_or_none=lambda: None,
            first=lambda: None,
            scalars=lambda: SimpleNamespace(all=lambda: [], first=lambda: None),
            all=lambda: [],
        )


class _ImgLLM:
    def __init__(self, data: bytes = PNG) -> None:
        self.data = data
        self.prompts: list[str] = []

    async def generate_image(self, prompt: str) -> bytes:
        self.prompts.append(prompt)
        return self.data


@pytest.fixture
def data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


# ─── images ──────────────────────────────────────────────────────────────────


async def test_generate_image_writes_file_and_meta(data_dir: Any) -> None:
    llm = _ImgLLM()
    db = _Session()
    info = await IMG.generate(
        db, user_id=1, account_id=5, prompt="  a cat  ", style="flat", llm=llm
    )  # type: ignore[arg-type]
    assert info.url == f"/api/images/{info.id}" and info.mime == "image/png"
    assert (data_dir / "images" / f"{info.id}.png").read_bytes() == PNG
    assert llm.prompts == ["a cat. Style: flat"]
    (row,) = db.added
    assert row.user_id == 1 and row.prompt == "a cat" and row.size_bytes == len(PNG)


async def test_generate_image_limits(data_dir: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(IMG.ImageError, match="empty_prompt"):
        await IMG.generate(_Session(), user_id=1, account_id=None, prompt="  ", llm=_ImgLLM())  # type: ignore[arg-type]
    with pytest.raises(IMG.ImageError, match="daily_limit"):
        await IMG.generate(
            _Session(count=10_000), user_id=1, account_id=None, prompt="x", llm=_ImgLLM()
        )  # type: ignore[arg-type]
    with pytest.raises(IMG.ImageError, match="llm"):
        await IMG.generate(_Session(), user_id=1, account_id=None, prompt="x", llm=_ImgLLM(b"tiny"))  # type: ignore[arg-type]
    monkeypatch.setenv("IMAGE_GEN_ENABLED", "false")
    get_settings.cache_clear()
    with pytest.raises(IMG.ImageError, match="disabled"):
        await IMG.generate(_Session(), user_id=1, account_id=None, prompt="x", llm=_ImgLLM())  # type: ignore[arg-type]


async def test_image_ownership(data_dir: Any) -> None:
    llm = _ImgLLM()
    db = _Session()
    info = await IMG.generate(db, user_id=1, account_id=5, prompt="x", llm=llm)  # type: ignore[arg-type]
    row = db.added[0]
    db.objects[("GeneratedImage", info.id)] = row
    db.objects[("Account", 5)] = SimpleNamespace(id=5, user_id=1)
    db.objects[("Account", 6)] = SimpleNamespace(id=6, user_id=2)
    got, path = await IMG.get_owned(db, user_id=1, image_id=info.id)  # type: ignore[arg-type]
    assert got.id == info.id and path.exists()
    with pytest.raises(IMG.ImageError, match="not_found"):
        await IMG.get_owned(db, user_id=2, image_id=info.id)  # type: ignore[arg-type]
    assert (await IMG.path_for_send(db, image_id=info.id, account_id=5)).exists()  # type: ignore[arg-type]
    with pytest.raises(IMG.ImageError, match="not_found"):
        await IMG.path_for_send(db, image_id=info.id, account_id=6)  # type: ignore[arg-type]


def test_registry_has_content_tools() -> None:
    assert "generate_image" in T.READ_TOOLS and "list_scheduled_messages" in T.READ_TOOLS
    spec = next(s for s in W.WRITE_TOOL_SPECS if s.name == "send_message")
    assert (
        "image_id" in spec.parameters["properties"]
        and "schedule_at" in spec.parameters["properties"]
    )


# ─── send_message with image ─────────────────────────────────────────────────


async def test_send_proposal_with_image(monkeypatch: pytest.MonkeyPatch, data_dir: Any) -> None:
    chat = SimpleNamespace(
        id=11,
        account_id=5,
        tg_peer_id=1001,
        title="K",
        username=None,
        type="channel",
        write_mode="write_with_confirm",
        is_writable=True,
    )

    async def resolve(ctx: Any, args: Any) -> Any:
        return chat

    async def ok_path(session: Any, *, image_id: str, account_id: int) -> Any:
        assert image_id == "img-1" and account_id == 5
        return data_dir

    monkeypatch.setattr(W, "_resolve_chat", resolve)
    monkeypatch.setattr(IMG, "path_for_send", ok_path)
    ctx = T.ToolContext(session=_Session(), account_id=5, pinned_chat_id=11)  # type: ignore[arg-type]
    p = await W.build_proposal(
        ctx, "send_message", {"text": "", "image_id": "img-1"}
    )  # caption bo'sh bo'lishi mumkin
    assert (
        p.args["image_id"] == "img-1"
        and p.args["text"] == ""
        and p.preview["image_url"] == "/api/images/img-1"
    )
    with pytest.raises(W.ProposalError, match="text_too_long"):
        await W.build_proposal(ctx, "send_message", {"text": "x" * 1100, "image_id": "img-1"})


# ─── auto-reply mantiqi ──────────────────────────────────────────────────────


def _rule(**kw: Any) -> Any:
    base = dict(
        id=1,
        chat_id=11,
        account_id=5,
        enabled=True,
        trigger="questions",
        keywords="",
        instructions="qisqa javob",
        max_per_hour=5,
        quiet_from=None,
        quiet_to=None,
        last_processed_msg_id=10,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    ("trigger", "kw", "text", "mentioned", "expected"),
    [
        ("questions", "", "Narxi qancha?", False, True),
        ("questions", "", "salom hammaga", False, False),
        ("questions", "", "how does it work", False, True),
        ("mentions", "", "hey", True, True),
        ("mentions", "", "hey", False, False),
        ("keywords", "narx, yetkazib", "Yetkazib berish bormi", False, True),
        ("keywords", "narx", "salom", False, False),
        ("all", "", "salom", False, True),
        ("all", "", "   ", False, False),
    ],
)
def test_matches(trigger: str, kw: str, text: str, mentioned: bool, expected: bool) -> None:
    assert AR.matches(_rule(trigger=trigger, keywords=kw), text, mentioned=mentioned) is expected


def test_quiet_hours() -> None:
    r = _rule(quiet_from=22, quiet_to=7)
    assert AR.in_quiet_hours(r, datetime(2026, 8, 18, 23, 0, tzinfo=UTC))
    assert AR.in_quiet_hours(r, datetime(2026, 8, 18, 3, 0, tzinfo=UTC))
    assert not AR.in_quiet_hours(r, datetime(2026, 8, 18, 12, 0, tzinfo=UTC))
    assert not AR.in_quiet_hours(_rule(), datetime(2026, 8, 18, 23, 0, tzinfo=UTC))
    assert AR.in_quiet_hours(
        _rule(quiet_from=9, quiet_to=18), datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
    )


class _ReplyLLM:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[list[Msg]] = []

    async def chat(self, task: Task, messages: list[Msg], **kw: Any) -> ChatResult:
        self.calls.append(messages)
        assert task is Task.SEARCH
        return ChatResult(text=self.text, tool_calls=[], model="m", provider="p", usage=Usage(1, 1))


def _msg(i: int, text: str, sender: int | None = 42) -> Any:
    return SimpleNamespace(
        tg_msg_id=i,
        text=text,
        sender_id=sender,
        published_at=datetime(2026, 8, 18, 10, i % 60, tzinfo=UTC),
    )


async def test_draft_reply_skip_and_untrusted() -> None:
    llm = _ReplyLLM("SKIP")
    out = await AR.draft_reply(
        rule=_rule(),
        chat_title="K",
        target=_msg(12, "narxi?"),
        context=[_msg(11, "salom")],
        llm=llm,
    )  # type: ignore[arg-type]
    assert out is None
    user = llm.calls[0][-1].content
    assert (
        '<untrusted_data source="telegram" chat="K" kind="recent">' in user
        and 'kind="target"' in user
    )
    assert "qisqa javob" in user
    llm2 = _ReplyLLM("  Narxi 100 000 so'm.  ")
    assert (
        await AR.draft_reply(
            rule=_rule(), chat_title="K", target=_msg(12, "narxi?"), context=[], llm=llm2
        )
        == "Narxi 100 000 so'm."
    )  # type: ignore[arg-type]


class _ARSession(_Session):
    """process_rule uchun: get(Chat/Account) + execute() ketma-ket natijalar."""

    def __init__(self, objects: dict[Any, Any], new_msgs: list[Any]) -> None:
        super().__init__(objects)
        self.new_msgs = new_msgs
        self.calls = 0

    async def execute(self, *a: Any, **k: Any) -> Any:
        self.calls += 1
        msgs = self.new_msgs
        # 1-chaqiruv: yangi xabarlar; 2-: sent_last_hour (0); keyingilar: kontekst (bo'sh)
        if self.calls == 1:
            return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: msgs))
        if self.calls == 2:
            return SimpleNamespace(scalar_one=lambda: 0)
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))


async def test_process_rule_proposes_send_message(monkeypatch: pytest.MonkeyPatch) -> None:
    proposals: list[dict[str, Any]] = []

    async def fake_propose(
        session: Any, ctx: Any, *, run_id: int, tool: str, args: dict[str, Any]
    ) -> T.ToolResult:
        proposals.append({"tool": tool, "args": args, "run_id": run_id})
        return T.ToolResult("proposed", meta={"action_id": 1, "proposed": True})

    from app.services import write_tools as WT

    monkeypatch.setattr(WT, "propose_or_execute", fake_propose)
    chat = SimpleNamespace(
        id=11, title="K", write_mode="write_with_confirm", is_writable=True, account_id=5
    )
    acc = SimpleNamespace(id=5, user_id=1, status="active", tg_account_id=777, label="@me")
    rule = _rule()
    msgs = [
        _msg(11, "salom"),
        _msg(12, "narxi qancha?"),
        _msg(13, "men javob berdim", sender=777),
        _msg(14, "@me qachon ochiq?"),
    ]
    db = _ARSession({("Chat", 11): chat, ("Account", 5): acc}, msgs)
    rep = await AR.process_rule(
        db, rule, llm=_ReplyLLM("Javob."), now=datetime(2026, 8, 18, 12, 0, tzinfo=UTC)
    )  # type: ignore[arg-type]
    assert rep["new"] == 4 and rep["matched"] == 2 and rep["proposed"] == 2
    assert [p["args"]["reply_to"] for p in proposals] == [12, 14]
    assert all(p["tool"] == "send_message" and p["args"]["text"] == "Javob." for p in proposals)
    assert rule.last_processed_msg_id == 14
    runs = [o for o in db.added if type(o).__name__ == "AgentRun"]
    assert len(runs) == 2 and runs[0].prompt == "autoreply:#12"


async def test_process_rule_respects_read_only_and_quiet() -> None:
    chat = SimpleNamespace(id=11, title="K", write_mode="read_only", is_writable=True, account_id=5)
    acc = SimpleNamespace(id=5, user_id=1, status="active", tg_account_id=777, label="")
    db = _ARSession({("Chat", 11): chat, ("Account", 5): acc}, [_msg(12, "narxi?")])
    assert (await AR.process_rule(db, _rule()))["skipped"] == "read_only"  # type: ignore[arg-type]
    chat.write_mode = "write_with_confirm"
    r = _rule(quiet_from=22, quiet_to=7)
    assert (await AR.process_rule(db, r, now=datetime(2026, 8, 18, 23, 0, tzinfo=UTC)))[
        "skipped"
    ] == "quiet"  # type: ignore[arg-type]
