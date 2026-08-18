"""`claude_code` adapter testlari — soxta runner, subprocess yo'q.

Qo'riqlaydi:
  * argv xavfsizlik bayroqlari: --tools "" (built-in tool'lar o'chiq),
    --no-session-persistence, --setting-sources ""
  * system prompt fayl orqali, prompt stdin orqali
  * ko'p-turnli suhbat transkriptga aylanadi
  * CLI JSON'ini parse: result/usage/model, is_error → LLMError (401 retry'siz)
  * TOOLS talab qilinsa CLI chaqirilmaydi (CapabilityError)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.llm import claude_code as CC
from app.llm.base import Capability, CapabilityError, LLMError, Msg, ToolCall, ToolSpec


def _ok(result: str, **extra: Any) -> str:
    data: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": result,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 11, "output_tokens": 4, "cache_read_input_tokens": 3},
        "modelUsage": {"claude-opus-5-20260101": {"canonicalModel": "claude-opus-5"}},
        "total_cost_usd": 0.001,
    }
    data.update(extra)
    return json.dumps(data)


class _Runner:
    """Soxta CLI: chaqiruvlarni yozib oladi, oldindan berilgan javobni qaytaradi."""

    def __init__(self, *responses: tuple[int, str, str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, argv: list[str], stdin: str, timeout: float) -> tuple[int, str, str]:  # noqa: ASYNC109
        sys_file = None
        if "--system-prompt-file" in argv:
            sys_file = _read(argv[argv.index("--system-prompt-file") + 1])
        self.calls.append({"argv": argv, "stdin": stdin, "timeout": timeout, "system": sys_file})
        return self.responses.pop(0)


def _read(path: str) -> str:
    return Path(path).read_text()


def _exists(path: str) -> bool:
    return Path(path).exists()


def _provider(runner: _Runner) -> CC.ClaudeCodeProvider:
    return CC.ClaudeCodeProvider(binary="/fake/claude", runner=runner)


# ─── argv / prompt ───────────────────────────────────────────────────────────


def test_build_argv_disables_builtin_tools_and_persistence() -> None:
    argv = CC.build_argv("/bin/claude", model="claude-opus-5", system_file="/x/s.md")
    assert argv[0] == "/bin/claude" and "-p" in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--setting-sources") + 1] == ""
    assert "--no-session-persistence" in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert argv[argv.index("--model") + 1] == "claude-opus-5"
    assert argv[argv.index("--system-prompt-file") + 1] == "/x/s.md"
    # system fayl bo'lmasa bayroq ham bo'lmaydi
    assert "--system-prompt-file" not in CC.build_argv("c", model="m", system_file=None)


def test_render_prompt_single_user_is_verbatim() -> None:
    system, prompt = CC.render_prompt([Msg.system("S1"), Msg.system("S2"), Msg.user("hi")])
    assert system == "S1\n\nS2"
    assert prompt == "hi"


def test_render_prompt_multiturn_becomes_transcript() -> None:
    _, prompt = CC.render_prompt(
        [
            Msg.user("q1"),
            Msg.assistant("a1", [ToolCall("t", "search", {"q": "x"})]),
            Msg.tool_result("t", "search", "found"),
            Msg.user("q2"),
        ]
    )
    assert prompt.startswith("<conversation>")
    assert '<turn role="user">\nq1' in prompt
    assert '<turn role="assistant">\na1\n[tool_call search] {"q": "x"}' in prompt
    assert '<turn role="tool_result">\n[search] found' in prompt
    assert prompt.rstrip().endswith("Do not repeat earlier turns.")


def test_clean_env_strips_nested_session_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "tok")
    env = CC.clean_env()
    assert "CLAUDECODE" not in env and "CLAUDE_CODE_ENTRYPOINT" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok"  # noqa: S105 — CLI o'zi ishlatadi


# ─── capabilities ────────────────────────────────────────────────────────────


def test_capabilities_have_no_tools() -> None:
    caps = _provider(_Runner()).capabilities("claude-opus-5")
    assert {Capability.CHAT, Capability.JSON_MODE, Capability.LONG_CONTEXT} <= caps
    assert Capability.TOOLS not in caps
    assert Capability.EMBED not in caps and Capability.IMAGE_GEN not in caps


async def test_tools_request_rejected_before_cli() -> None:
    runner = _Runner()
    with pytest.raises(CapabilityError):
        await _provider(runner).chat(
            [Msg.user("x")],
            model="claude-opus-5",
            tools=[ToolSpec("t", "d", {"type": "object", "properties": {}})],
        )
    assert runner.calls == []


# ─── chat ────────────────────────────────────────────────────────────────────


async def test_chat_passes_system_via_file_and_prompt_via_stdin() -> None:
    runner = _Runner((0, _ok("Salom"), ""))
    res = await _provider(runner).chat(
        [Msg.system("Sen yordamchisan"), Msg.user("hi")], model="claude-opus-5"
    )
    call = runner.calls[0]
    assert call["system"] == "Sen yordamchisan"
    assert call["stdin"] == "hi"
    assert not _exists(call["argv"][call["argv"].index("--system-prompt-file") + 1])  # tozalangan
    assert res.text == "Salom"
    assert res.provider == "claude_code"
    assert res.model == "claude-opus-5"
    assert (res.usage.tokens_in, res.usage.tokens_out, res.usage.cached_tokens) == (11, 4, 3)
    assert res.finish_reason == "end_turn"


async def test_json_mode_adds_instruction_and_strips_fences() -> None:
    runner = _Runner((0, _ok('```json\n{"intent": "search"}\n```'), ""))
    res = await _provider(runner).chat([Msg.user("x")], model="claude-haiku-4-5", json_mode=True)
    assert "JSON" in (runner.calls[0]["system"] or "")
    assert res.text == '{"intent": "search"}'


async def test_not_logged_in_is_401_no_retry() -> None:
    runner = _Runner(
        (0, _ok("Not logged in · Please run /login", is_error=True, api_error_status=None), "")
    )
    with pytest.raises(LLMError, match="Not logged in") as ei:
        await _provider(runner).chat([Msg.user("x")], model="claude-opus-5")
    assert ei.value.status == 401
    assert len(runner.calls) == 1


async def test_api_error_status_is_propagated_and_retried() -> None:
    err = _ok("overloaded", is_error=True, api_error_status=529)
    runner = _Runner((0, err, ""), (0, _ok("ok"), ""))
    p = _provider(runner)
    p.max_retries = 2
    res = await p.chat([Msg.user("x")], model="claude-opus-5")
    assert res.text == "ok"
    assert len(runner.calls) == 2


async def test_non_json_output_is_wrapped() -> None:
    runner = _Runner((1, "", "boom"))
    p = _provider(runner)
    p.max_retries = 1
    with pytest.raises(LLMError, match="exit=1"):
        await p.chat([Msg.user("x")], model="claude-opus-5")


async def test_missing_binary_fails_before_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CC, "find_cli", lambda *_a, **_k: None)
    runner = _Runner()
    p = CC.ClaudeCodeProvider(runner=runner)
    assert p.binary is None
    with pytest.raises(LLMError, match="CLI topilmadi"):
        await p.chat([Msg.user("x")], model="claude-opus-5")
    assert runner.calls == []


# ─── detect ──────────────────────────────────────────────────────────────────


def test_detect_returns_none_when_binary_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(tmp_path / "nope"))
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        assert CC.detect_claude_code() is None
    finally:
        get_settings.cache_clear()
