"""Yagona entrypoint: `python -m app.main {bot|check}`."""

from __future__ import annotations

import asyncio
import sys

from app.config import get_settings
from app.logging import get_logger, setup_logging


async def _check() -> int:
    """Konfiguratsiya + ulanishlarni tekshiradi. Deploy'dan keyin birinchi qadam."""
    from sqlalchemy import text

    from app.db.base import get_engine
    from app.mtproto.allowlist import ALLOWED, AUTH, DENIED

    log = get_logger("check")
    s = get_settings()
    ok = True

    assert not (ALLOWED | AUTH) & DENIED
    log.info("check.allowlist", allowed=len(ALLOWED), denied=len(DENIED), auth=len(AUTH))

    # LLM marshruti — qaysi vazifa qaysi providerga tushishini deploy paytida ko'rsatadi
    from app.llm import LLMError, Task, effective_provider, resolve
    from app.llm.claude import detect_claude_auth
    from app.llm.claude_code import detect_claude_code

    claude_auth = detect_claude_auth(s)
    claude_cli = detect_claude_code(s)
    primary = effective_provider(s.llm_provider)
    log.info(
        "check.llm_provider",
        configured=s.llm_provider,
        effective=primary,
        claude_auth=claude_auth.source if claude_auth else None,
        claude_cli=claude_cli.binary if claude_cli else None,
        claude_cli_logged_in=claude_cli.logged_in if claude_cli else None,
    )
    if primary == "claude_code" and not (claude_cli and claude_cli.logged_in):
        log.error(
            "check.llm",
            ok=False,
            error=(
                "LLM_PROVIDER=claude_code, lekin `claude` CLI topilmadi yoki login yo'q — "
                f"{claude_cli.error if claude_cli else 'PATH da claude yo`q'}"
            ),
        )
        ok = False
    if primary == "claude" and claude_auth is None:
        log.error(
            "check.llm",
            ok=False,
            error=(
                "LLM_PROVIDER=claude, lekin kredensial yo'q — ANTHROPIC_API_KEY, "
                "CLAUDE_CODE_OAUTH_TOKEN (`claude setup-token`) yoki `ant auth login` kerak"
            ),
        )
        ok = False

    for task in Task:
        try:
            provider, model = resolve(task)
            log.info("check.llm", task=str(task), provider=provider.name, model=model)
        except LLMError as exc:
            log.error("check.llm", task=str(task), ok=False, error=str(exc))
            ok = False

    if primary == "deepseek" and not s.deepseek_api_key:
        log.error(
            "check.llm", ok=False, error="LLM_PROVIDER=deepseek, lekin DEEPSEEK_API_KEY bo'sh"
        )
        ok = False
    if not s.gemini_api_key:
        log.error(
            "check.llm",
            ok=False,
            error="GEMINI_API_KEY bo'sh — embedding va rasm generatsiya ishlamaydi",
        )
        ok = False

    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("select 1"))
            ver = (
                await conn.execute(
                    text("select extversion from pg_extension where extname='vector'")
                )
            ).scalar()
        log.info("check.postgres", ok=True, pgvector=ver or "O'RNATILMAGAN")
        if not ver:
            ok = False
    except Exception as exc:
        log.error("check.postgres", ok=False, error=str(exc))
        ok = False

    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(s.redis_url)  # type: ignore[no-untyped-call]
        await client.ping()
        await client.aclose()
        log.info("check.redis", ok=True)
    except Exception as exc:
        log.error("check.redis", ok=False, error=str(exc))
        ok = False

    log.info("check.done", ok=ok, env=s.env, max_accounts=s.max_accounts)
    return 0 if ok else 1


def main() -> int:
    s = get_settings()
    setup_logging(s.log_level, json_output=s.is_prod)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "bot"

    if cmd == "bot":
        from app.bot.main import run_bot

        asyncio.run(run_bot())
        return 0
    if cmd == "check":
        return asyncio.run(_check())

    print(f"Noma'lum buyruq: {cmd}. Mavjud: bot, check", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
