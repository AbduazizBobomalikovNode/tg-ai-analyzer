"""Claude Code CLI adapter — `claude -p` subprocess.

Nima uchun: foydalanuvchi Claude Code'ga login qilgan (Pro/Max obuna),
`claude` CLI shu loginni Keychain / `~/.claude` dan o'zi oladi. Biz CLI'ni
headless rejimda chaqiramiz — token nusxalash, `.env` ga yozish shart emas.
Bu Anthropic'ning rasmiy yo'li (Agent SDK ham aynan shu CLI'ni ishga tushiradi).

    claude -p --output-format json --tools "" --no-session-persistence \\
           --setting-sources "" --model <model> --system-prompt-file <tmp>  < prompt

Chegaralar (rostgo'y `capabilities()`):
  * **TOOLS yo'q** — `claude -p` tool'larni o'z siklida bajaradi, bizga
    `tool_use` bloki qaytarmaydi. Bizning agent sikli (`ChatResult.tool_calls`
    → ilova bajaradi → tasdiqlash) bu bilan mos emas. `Task.TOOLS` router
    orqali fallback provider'ga tushadi. Kelajakda MCP ko'prik qo'shsa bo'ladi.
  * EMBED / IMAGE_GEN yo'q — Gemini.
  * Har so'rov alohida jarayon (~1-2 s overhead) + Claude Code'ning o'z system
    prompt overhead'i. Ko'p-turnli suhbat matn transkript sifatida beriladi.
  * Docker'da CLI (Node) va login kerak — asosan lokal/bare-metal uchun.
    Konteynerda `CLAUDE_CODE_OAUTH_TOKEN` + `claude` provider osonroq.

Xavfsizlik:
  * `--tools ""` — CLI'ning Bash/Read/Write tool'lari **o'chirilgan**: bu faqat
    LLM chaqiruvi, server fayl tizimiga tegmaydi.
  * `--setting-sources ""` + neytral cwd — foydalanuvchi/loyiha settings,
    hooks, CLAUDE.md yuklanmaydi (bizning promptdan tashqari hech narsa).
  * `--no-session-persistence` — chat kontenti diskda transkript bo'lib qolmaydi.
  * Ichki `CLAUDECODE*` env'lar tozalanadi (nested-session rad etilmasin).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.llm.base import (
    BaseProvider,
    Capability,
    ChatResult,
    LLMError,
    Msg,
    Role,
    ToolSpec,
    Usage,
)
from app.llm.claude import strip_json_fences
from app.logging import get_logger

log = get_logger(__name__)

_JSON_INSTRUCTION = (
    "Respond with a single valid JSON value only. "
    "No markdown code fences, no prose before or after the JSON."
)

# CLI ichidagi env'lar — bizning bot Claude Code ichidan ishga tushirilsa,
# `claude` "nested session" deb rad etadi. Token'ni saqlaymiz (CLI o'zi ishlatadi).
_STRIP_ENV_PREFIXES = ("CLAUDECODE", "CLAUDE_CODE_")
_KEEP_ENV = frozenset({"CLAUDE_CODE_OAUTH_TOKEN"})

# `claude auth status` uchun
_AUTH_PROBE_TIMEOUT = 20.0

# runner imzosi: (argv, stdin_text, timeout) -> (returncode, stdout, stderr)
Runner = Callable[[list[str], str, float], Awaitable[tuple[int, str, str]]]


@dataclass(frozen=True, slots=True)
class CliInfo:
    binary: str
    logged_in: bool
    auth_method: str = ""  # "claude.ai" | "console" | ...
    error: str = ""


def find_cli(settings: Settings | None = None) -> str | None:
    s = settings or get_settings()
    if s.claude_code_bin:
        return s.claude_code_bin if Path(s.claude_code_bin).exists() else None
    return shutil.which("claude")


def detect_claude_code(settings: Settings | None = None) -> CliInfo | None:
    """`claude` CLI bormi va login qilinganmi? Sinxron, tarmoqsiz (`auth status`).

    `auto` rejimda bir marta chaqiriladi (router keshlaydi).
    """
    s = settings or get_settings()
    binary = find_cli(s)
    if not binary:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 — argv ro'yxat, shell yo'q
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=_AUTH_PROBE_TIMEOUT,
            env=clean_env(),
            cwd=tempfile.gettempdir(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("llm.claude_code.probe_failed", error=str(exc))
        return CliInfo(binary, False, error=str(exc))

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        data = {}
    logged_in = bool(data.get("loggedIn"))
    return CliInfo(
        binary,
        logged_in,
        auth_method=str(data.get("authMethod") or ""),
        error="" if logged_in else (proc.stderr or proc.stdout or "").strip()[:200],
    )


def clean_env() -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k in _KEEP_ENV or not k.startswith(_STRIP_ENV_PREFIXES)
    }
    # ichki so'rovda rang/progress chiqmasin
    env.setdefault("NO_COLOR", "1")
    return env


class ClaudeCodeProvider(BaseProvider):
    name = "claude_code"

    def __init__(
        self,
        binary: str | None = None,
        *,
        timeout: float | None = None,
        runner: Runner | None = None,
    ) -> None:
        s = get_settings()
        self._binary = binary or find_cli(s)
        self._timeout = float(timeout or s.claude_code_timeout)
        # `runner` — testlarda soxta CLI ulash uchun
        self._run: Runner = runner or _run_cli

    @property
    def binary(self) -> str | None:
        return self._binary

    # ── imkoniyatlar ─────────────────────────────────────────────────────────

    def capabilities(self, model: str) -> frozenset[Capability]:
        if not model:
            return frozenset()
        # TOOLS yo'q (qarang: modul docstring), EMBED/IMAGE_GEN yo'q
        return frozenset({Capability.CHAT, Capability.JSON_MODE, Capability.LONG_CONTEXT})

    # ── chat ─────────────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Msg],
        *,
        model: str,
        tools: list[ToolSpec] | None = None,
        temperature: float = 0.7,  # CLI'da boshqarilmaydi — e'tiborsiz
        max_tokens: int | None = None,  # CLI'da boshqarilmaydi — e'tiborsiz
        json_mode: bool = False,
    ) -> ChatResult:
        needed = [Capability.CHAT]
        if tools:
            needed.append(Capability.TOOLS)  # → CapabilityError, HTTP'gacha
        if json_mode:
            needed.append(Capability.JSON_MODE)
        self.require(model, *needed)

        if not self._binary:
            raise LLMError(
                self.name,
                "`claude` CLI topilmadi. O'rnating (npm i -g @anthropic-ai/claude-code) va "
                "`claude` ichida /login qiling, yoki CLAUDE_CODE_BIN bering.",
                status=401,
            )

        system_text, prompt = render_prompt(messages)
        if json_mode:
            system_text = f"{system_text}\n\n{_JSON_INSTRUCTION}".strip()

        async def _call() -> dict[str, Any]:
            with _system_prompt_file(system_text) as sys_file:
                argv = build_argv(self._binary or "claude", model=model, system_file=sys_file)
                rc, out, err = await self._run(argv, prompt, self._timeout)
            return _decode(rc, out, err)

        data: dict[str, Any] = await self._with_retry("chat", _call)
        return parse_result(data, model=model, json_mode=json_mode)


# ─── CLI chaqiruv ────────────────────────────────────────────────────────────


def build_argv(binary: str, *, model: str, system_file: str | None) -> list[str]:
    argv = [
        binary,
        "-p",
        "--output-format",
        "json",
        "--tools",
        "",  # barcha built-in tool'lar o'chiq
        "--no-session-persistence",
        "--setting-sources",
        "",  # user/project settings, hooks, CLAUDE.md yuklanmasin
        "--model",
        model,
    ]
    if system_file:
        argv += ["--system-prompt-file", system_file]
    return argv


class _system_prompt_file:
    """System prompt'ni argv emas, fayl orqali — Linux'da bitta arg 128KB bilan chegaralangan."""

    def __init__(self, text: str) -> None:
        self._text = text
        self._path: str | None = None

    def __enter__(self) -> str | None:
        if not self._text:
            return None
        fd, path = tempfile.mkstemp(prefix="tgai-sys-", suffix=".md")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(self._text)
        self._path = path
        return path

    def __exit__(self, *_exc: object) -> None:
        if self._path:
            try:
                os.unlink(self._path)
            except OSError:
                pass


async def _run_cli(argv: list[str], stdin_text: str, timeout: float) -> tuple[int, str, str]:  # noqa: ASYNC109 — Runner imzosi
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=clean_env(),
            cwd=tempfile.gettempdir(),  # neytral cwd — loyiha CLAUDE.md'si tushmasin
        )
    except OSError as exc:
        raise LLMError("claude_code", f"CLI ishga tushmadi: {exc}", status=401) from exc

    try:
        out_b, err_b = await asyncio.wait_for(
            proc.communicate(stdin_text.encode("utf-8")), timeout=timeout
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise LLMError("claude_code", f"timeout ({timeout:.0f}s)") from None

    return proc.returncode or 0, out_b.decode("utf-8", "replace"), err_b.decode("utf-8", "replace")


def _decode(rc: int, out: str, err: str) -> dict[str, Any]:
    """CLI chiqishini JSON'ga; xatolarni LLMError'ga (status bilan, retry uchun)."""
    data: dict[str, Any] | None = None
    if out.strip():
        try:
            parsed = json.loads(out)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            data = None

    if data is None:
        msg = (err or out or "bo'sh chiqish").strip()[:300]
        raise LLMError("claude_code", f"CLI exit={rc}: {msg}", status=_status_from_text(msg))

    if data.get("is_error") or data.get("subtype", "success") != "success":
        msg = str(data.get("result") or data.get("subtype") or "noma'lum xato")[:300]
        api_status = data.get("api_error_status")
        status = api_status if isinstance(api_status, int) else _status_from_text(msg)
        raise LLMError("claude_code", msg, status=status)
    return data


def _status_from_text(msg: str) -> int | None:
    low = msg.lower()
    if "not logged in" in low or "/login" in low or "authentication" in low:
        return 401  # qayta urinish befoyda
    if "rate limit" in low or "overloaded" in low:
        return 429
    return None


# ─── konvertorlar ────────────────────────────────────────────────────────────


def render_prompt(messages: list[Msg]) -> tuple[str, str]:
    """SYSTEM → system prompt; qolgani → bitta prompt.

    Bitta user xabari bo'lsa — o'zi. Ko'p-turnli bo'lsa transkript ko'rinishida
    (CLI'ga assistant turn'larini "inject" qilib bo'lmaydi).
    """
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []

    for m in messages:
        if m.role is Role.SYSTEM:
            if m.content:
                system_parts.append(m.content)
        elif m.role is Role.TOOL:
            turns.append(("tool_result", f"[{m.name or 'tool'}] {m.content}"))
        elif m.role is Role.ASSISTANT:
            text = m.content
            for tc in m.tool_calls:
                text += f"\n[tool_call {tc.name}] {json.dumps(tc.arguments, ensure_ascii=False)}"
            if text.strip():
                turns.append(("assistant", text))
        elif m.content:
            turns.append(("user", m.content))

    system_text = "\n\n".join(system_parts)

    if not turns:
        return system_text, ""
    if len(turns) == 1 and turns[0][0] == "user":
        return system_text, turns[0][1]

    body = "\n".join(f'<turn role="{role}">\n{text}\n</turn>' for role, text in turns)
    prompt = (
        "<conversation>\n"
        f"{body}\n"
        "</conversation>\n\n"
        "Continue this conversation: reply to the last user turn only, "
        "as the assistant. Do not repeat earlier turns."
    )
    return system_text, prompt


def parse_result(data: dict[str, Any], *, model: str, json_mode: bool = False) -> ChatResult:
    text = str(data.get("result") or "")
    if json_mode:
        text = strip_json_fences(text)

    u = data.get("usage") or {}
    used_model = model
    for key, info in (data.get("modelUsage") or {}).items():
        used_model = str((info or {}).get("canonicalModel") or key)
        break

    log.debug(
        "llm.claude_code.result",
        model=used_model,
        cost_usd=data.get("total_cost_usd"),
        turns=data.get("num_turns"),
        duration_ms=data.get("duration_ms"),
    )
    return ChatResult(
        text=text,
        tool_calls=[],
        model=used_model,
        provider="claude_code",
        usage=Usage(
            tokens_in=int(u.get("input_tokens") or 0),
            tokens_out=int(u.get("output_tokens") or 0),
            cached_tokens=int(u.get("cache_read_input_tokens") or 0),
        ),
        finish_reason=str(data.get("stop_reason") or ""),
    )
