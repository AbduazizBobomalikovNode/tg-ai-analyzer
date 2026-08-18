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
    from app.llm import LLMError, Task, resolve

    for task in Task:
        try:
            provider, model = resolve(task)
            log.info("check.llm", task=str(task), provider=provider.name, model=model)
        except LLMError as exc:
            log.error("check.llm", task=str(task), ok=False, error=str(exc))
            ok = False

    if s.llm_provider == "deepseek" and not s.deepseek_api_key:
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

        client = aioredis.from_url(s.redis_url)
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
