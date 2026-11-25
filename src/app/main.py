"""Yagona entrypoint: `python -m app.main {bot|check|reembed|backfill-senders}`."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

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

    # L10: `EMBED_DIMS` ↔ DB ustuni. Mos kelmasa har embedding INSERT'i
    # yiqiladi — deploy paytida bilingani ma'qul.
    from app.config import embed_dims_mismatch

    dims_error = embed_dims_mismatch(s)
    if dims_error:
        log.error("check.embed_dims", ok=False, error=dims_error)
        ok = False
    else:
        log.info("check.embed_dims", ok=True, dims=s.embed_dims, model=s.embed_model)

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


async def _reembed(argv: list[str]) -> int:
    """`python -m app.main reembed [--model X] [--account N]` — ishni navbatga qo'yadi.

    Hech narsa o'chirilmaydi: ish yangi model uchun yetishmayotgan vektorlarni
    to'ldiradi, eskilari joyida qoladi. Jarayonni worker loglari ko'rsatadi
    (`worker.reembed.progress` / `worker.reembed.done`).
    """
    from app.llm import embed_model_name
    from app.worker.queue import close_queue, enqueue
    from app.worker.tasks import job_id_reembed

    log = get_logger("reembed")
    model = ""
    account_id: int | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            model = argv[i + 1].strip()
            i += 2
        elif argv[i] == "--account" and i + 1 < len(argv):
            account_id = int(argv[i + 1])
            i += 2
        else:
            print(f"Noma'lum argument: {argv[i]}. Format: reembed [--model X] [--account N]")
            return 2
    model = model or embed_model_name()
    try:
        job = await enqueue(
            "reembed_all", account_id, model=model, job_id=job_id_reembed(account_id)
        )
    finally:
        await close_queue()
    if job is None:
        log.warning("reembed.duplicate", model=model, account_id=account_id)
        return 1
    log.info("reembed.enqueued", model=model, account_id=account_id, job=job)
    return 0


async def _backfill_senders(argv: list[str]) -> int:
    """`python -m app.main backfill-senders --account N` — nomlarni to'ldirish.

    Ish worker navbatiga tushadi (soatlik cron bilan bir xil yo'l): har chatdan
    oxirgi 200 xabar o'qiladi va `m.sender` keshidan `senders` to'ldiriladi.
    """
    from app.worker.queue import close_queue, enqueue
    from app.worker.tasks import job_id_senders

    log = get_logger("senders")
    account_id: int | None = None
    per_chat: int | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--account" and i + 1 < len(argv):
            account_id = int(argv[i + 1])
            i += 2
        elif argv[i] == "--per-chat" and i + 1 < len(argv):
            per_chat = int(argv[i + 1])
            i += 2
        else:
            print(
                f"Noma'lum argument: {argv[i]}. "
                "Format: backfill-senders --account N [--per-chat 200]"
            )
            return 2
    if account_id is None:
        print("`--account N` majburiy: backfill-senders --account N [--per-chat 200]")
        return 2
    kwargs: dict[str, Any] = {"job_id": job_id_senders(account_id)}
    if per_chat is not None:
        kwargs["per_chat"] = per_chat
    try:
        job = await enqueue("backfill_senders", account_id, **kwargs)
    finally:
        await close_queue()
    if job is None:
        log.warning("senders.duplicate", account_id=account_id)
        return 1
    log.info("senders.enqueued", account_id=account_id, job=job)
    return 0


def main() -> int:
    s = get_settings()
    setup_logging(s.log_level, json_output=s.is_prod)

    cmd = sys.argv[1] if len(sys.argv) > 1 else "bot"

    if cmd == "bot":
        from app.bot.main import run_bot

        try:
            asyncio.run(run_bot())
        except KeyboardInterrupt:
            # Auth-retry siklidagi `asyncio.sleep()` yoki polling shu yerda
            # to'xtaydi — traceback bilan emas, toza chiqish bilan.
            get_logger("bot").info("bot.stopped", reason="keyboard_interrupt")
        return 0
    if cmd == "check":
        return asyncio.run(_check())
    if cmd == "reembed":
        return asyncio.run(_reembed(sys.argv[2:]))
    if cmd == "backfill-senders":
        return asyncio.run(_backfill_senders(sys.argv[2:]))

    print(
        f"Noma'lum buyruq: {cmd}. Mavjud: bot, check, reembed, backfill-senders",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
