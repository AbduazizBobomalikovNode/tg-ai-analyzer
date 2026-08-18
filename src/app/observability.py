"""Kuzatuv (8-bosqich): Sentry (ixtiyoriy), Prometheus metrikalar, worker heartbeat.

* `init_sentry(component)` — `SENTRY_DSN` bo'lsa `sentry-sdk` (o'rnatilgan bo'lsa);
  sirlar `before_send` da maskalanadi (`app.logging._SECRET_KEYS`).
* Metrikalar — `prometheus_client` (majburiy dep, yengil). Nomlar barqaror:
    tgai_http_requests_total{method,route,status}, tgai_http_latency_seconds{route}
    tgai_llm_requests_total{provider,model,task,ok}, tgai_llm_tokens_total{provider,model,dir}
    tgai_llm_cost_usd_total{provider,model}
    tgai_tool_calls_total{tool,ok}, tgai_write_actions_total{tool,status}
    tgai_sync_messages_total, tgai_floodwait_total, tgai_snapshot_posts_total
  api: `GET /metrics` (`METRICS_TOKEN` bo'lsa `?token=`/Bearer talab qilinadi).
  worker jarayoni alohida — uning metrikalari heartbeat + DB orqali (`/api/stats`).
* Heartbeat — worker har cron/ish oxirida Redis'ga `tgai:hb:<name>` = ISO vaqt
  (TTL 3 kun); dashboard "Tizim" kartasi so'nggi ishga tushishlarni ko'rsatadi.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

# ─── metrikalar ──────────────────────────────────────────────────────────────

HTTP_REQUESTS = Counter("tgai_http_requests_total", "HTTP so'rovlar", ["method", "route", "status"])
HTTP_LATENCY = Histogram(
    "tgai_http_latency_seconds",
    "HTTP javob vaqti",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
LLM_REQUESTS = Counter(
    "tgai_llm_requests_total", "LLM chaqiruvlar", ["provider", "model", "task", "ok"]
)
LLM_TOKENS = Counter("tgai_llm_tokens_total", "LLM tokenlar", ["provider", "model", "dir"])
LLM_COST = Counter("tgai_llm_cost_usd_total", "LLM taxminiy xarajat, USD", ["provider", "model"])
LLM_LATENCY = Histogram(
    "tgai_llm_latency_seconds",
    "LLM chaqiruv vaqti",
    ["task"],
    buckets=(0.5, 1, 2, 5, 10, 20, 40, 90),
)
TOOL_CALLS = Counter("tgai_tool_calls_total", "Agent tool chaqiruvlari", ["tool", "ok"])
WRITE_ACTIONS = Counter("tgai_write_actions_total", "Yozish amallari", ["tool", "status"])
SYNC_MESSAGES = Counter("tgai_sync_messages_total", "Sinxronlangan xabarlar")
FLOODWAIT = Counter("tgai_floodwait_total", "FloodWait hodisalari")
SNAPSHOT_POSTS = Counter("tgai_snapshot_posts_total", "Snapshot olingan postlar")


def record_llm(
    *,
    provider: str,
    model: str,
    task: str,
    ok: bool,
    tokens_in: int,
    tokens_out: int,
    seconds: float,
) -> None:
    from app.llm.pricing import estimate_cost

    LLM_REQUESTS.labels(provider or "?", model or "?", task, str(ok).lower()).inc()
    if tokens_in:
        LLM_TOKENS.labels(provider or "?", model or "?", "in").inc(tokens_in)
    if tokens_out:
        LLM_TOKENS.labels(provider or "?", model or "?", "out").inc(tokens_out)
    LLM_LATENCY.labels(task).observe(seconds)
    cost = estimate_cost(provider, model, tokens_in, tokens_out)
    if cost:
        LLM_COST.labels(provider, model).inc(cost)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


# ─── Sentry (ixtiyoriy) ──────────────────────────────────────────────────────


def _scrub(event: dict[str, Any], _hint: Any) -> dict[str, Any] | None:
    from app.logging import _SECRET_KEYS

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: ("***" if str(k).lower() in _SECRET_KEYS else walk(v)) for k, v in node.items()
            }
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(event)  # type: ignore[no-any-return]


def init_sentry(component: str) -> bool:
    s = get_settings()
    if not s.sentry_dsn:
        return False
    try:
        import sentry_sdk
    except ImportError:
        log.warning("observability.sentry_missing", hint="pip install sentry-sdk")
        return False
    sentry_sdk.init(
        dsn=s.sentry_dsn,
        environment=s.env,
        release=f"tg-ai-analyzer@{s.app_version}",
        traces_sample_rate=s.sentry_traces_rate,
        send_default_pii=False,
        before_send=_scrub,
    )
    sentry_sdk.set_tag("component", component)
    log.info("observability.sentry", component=component, env=s.env)
    return True


# ─── heartbeat (worker → Redis) ──────────────────────────────────────────────

HB_PREFIX = "tgai:hb:"
HB_TTL = 3 * 24 * 3600


async def heartbeat(name: str, *, extra: dict[str, Any] | None = None) -> None:
    """Worker cron/ish oxirida chaqiriladi. Redis yo'q bo'lsa jim o'tadi."""
    try:
        import json

        import redis.asyncio as aioredis

        r = aioredis.from_url(get_settings().redis_url)  # type: ignore[no-untyped-call]
        try:
            payload = {"at": datetime.now(UTC).isoformat(), **(extra or {})}
            await r.set(HB_PREFIX + name, json.dumps(payload, default=str), ex=HB_TTL)
        finally:
            await r.aclose()
    except Exception as exc:  # kuzatuv hech qachon ishni to'xtatmasin
        log.debug("observability.heartbeat_failed", name=name, error=str(exc)[:100])


async def heartbeats() -> dict[str, Any]:
    try:
        import json

        import redis.asyncio as aioredis

        r = aioredis.from_url(get_settings().redis_url)  # type: ignore[no-untyped-call]
        try:
            keys = [k async for k in r.scan_iter(match=HB_PREFIX + "*")]
            out: dict[str, Any] = {}
            for k in keys:
                raw = await r.get(k)
                name = (k.decode() if isinstance(k, bytes) else str(k)).removeprefix(HB_PREFIX)
                out[name] = json.loads(raw) if raw else None
            return out
        finally:
            await r.aclose()
    except Exception as exc:
        return {"_error": str(exc)[:100]}


# ─── ASGI middleware (so'rov logi + metrikalar) ──────────────────────────────


def http_middleware() -> Callable[..., Awaitable[Any]]:
    async def middleware(request: Any, call_next: Callable[[Any], Awaitable[Any]]) -> Any:
        started = time.perf_counter()
        route = _route_of(request)
        status = 500
        try:
            response = await call_next(request)
            status = int(response.status_code)
            return response
        finally:
            dt = time.perf_counter() - started
            if route not in ("/health", "/metrics") and not route.startswith("/static"):
                HTTP_REQUESTS.labels(request.method, route, str(status)).inc()
                HTTP_LATENCY.labels(route).observe(dt)
                if status >= 500 or dt > 5:
                    log.warning(
                        "http.slow_or_error",
                        method=request.method,
                        route=route,
                        status=status,
                        ms=int(dt * 1000),
                    )

    return middleware


def _route_of(request: Any) -> str:
    r = request.scope.get("route")
    path = getattr(r, "path", None)
    return str(path) if path else str(request.url.path)
