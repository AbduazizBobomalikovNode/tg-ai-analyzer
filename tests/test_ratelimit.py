"""Metod sinflari bo'yicha token bucket'lar (`app/mtproto/ratelimit.py`).

Soat va tasodif — in'ektsiya qilinadi: test real vaqt kutmaydi va jitter
tufayli "ba'zan" yiqilmaydi.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.mtproto.ratelimit import (
    BUCKETS,
    DEFAULT_SPECS,
    BucketSpec,
    InMemoryBucketStore,
    MethodBuckets,
    specs_from_settings,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


def _mb(**kw: Any) -> tuple[MethodBuckets, _Clock]:
    clock = _Clock()
    wall = _Clock()
    wall.t = 1_800_000_000.0
    return MethodBuckets(clock=clock, wall_clock=wall, jitter=0.0, **kw), clock


# ─── sukut qiymatlar — arxitektura v2, 8.1-band ──────────────────────────────


def test_default_rates_match_the_spec() -> None:
    assert tuple(DEFAULT_SPECS) == BUCKETS
    assert (DEFAULT_SPECS["history"].per_minute, DEFAULT_SPECS["history"].burst) == (40, 8)
    assert (DEFAULT_SPECS["search"].per_minute, DEFAULT_SPECS["search"].burst) == (12, 4)
    resolve = DEFAULT_SPECS["resolve"]
    assert (resolve.per_minute, resolve.burst, resolve.per_day) == (4, 2, 60)
    discover = DEFAULT_SPECS["discover"]
    assert (discover.per_minute, discover.burst, discover.per_day) == (3, 1, 30)


# ─── burst / to'lish ─────────────────────────────────────────────────────────


def test_burst_then_refill() -> None:
    buckets, clock = _mb(specs={"search": BucketSpec("search", 60, 3)})
    assert [buckets.try_acquire(1, "search") for _ in range(3)] == [True, True, True]
    assert not buckets.try_acquire(1, "search")

    clock.tick(1.0)  # 60/min = soniyasiga bitta token
    assert buckets.try_acquire(1, "search")
    assert not buckets.try_acquire(1, "search")


def test_buckets_and_accounts_are_isolated() -> None:
    buckets, _ = _mb(specs={"a": BucketSpec("a", 60, 1), "b": BucketSpec("b", 60, 1)})
    assert buckets.try_acquire(1, "a")
    assert not buckets.try_acquire(1, "a")
    assert buckets.try_acquire(1, "b")  # boshqa sinf — o'z kvotasi
    assert buckets.try_acquire(2, "a")  # boshqa akkaunt — o'z kvotasi


def test_zero_rate_disables_the_class() -> None:
    buckets, _ = _mb(specs={"discover": BucketSpec("discover", 0, 1)})
    assert not buckets.try_acquire(1, "discover")
    assert buckets.retry_after(1, "discover") == 0


def test_unknown_bucket_is_a_programming_error() -> None:
    buckets, _ = _mb(specs={"search": BucketSpec("search", 1, 1)})
    with pytest.raises(KeyError):
        buckets.try_acquire(1, "nope")


# ─── kunlik chegara ──────────────────────────────────────────────────────────


def test_daily_cap_blocks_until_tomorrow() -> None:
    buckets, clock = _mb(specs={"discover": BucketSpec("discover", 600, 2, per_day=3)})
    for _ in range(3):
        assert buckets.try_acquire(1, "discover")
        clock.tick(1.0)
    assert not buckets.try_acquire(1, "discover")
    # kunlik chegarada kutish — daqiqalik to'lishdan ancha uzoq
    assert buckets.retry_after(1, "discover") > 60


def test_daily_counter_resets_on_a_new_day() -> None:
    clock, wall = _Clock(), _Clock()
    wall.t = 1_800_000_000.0
    buckets = MethodBuckets(
        {"discover": BucketSpec("discover", 600, 2, per_day=1)},
        clock=clock,
        wall_clock=wall,
        jitter=0.0,
    )
    assert buckets.try_acquire(1, "discover")
    assert not buckets.try_acquire(1, "discover")
    wall.tick(86_400)  # ertasi kun
    clock.tick(60)
    assert buckets.try_acquire(1, "discover")


# ─── retry_after / jitter ────────────────────────────────────────────────────


def test_retry_after_is_seconds_until_one_token() -> None:
    buckets, _ = _mb(specs={"search": BucketSpec("search", 12, 1)})  # 5 s/token
    assert buckets.try_acquire(1, "search")
    assert buckets.retry_after(1, "search") == 5
    assert buckets.retry_after(1, "search") >= 1  # holatni buzmaydi, hech qachon 0 emas


def test_jitter_varies_the_cost_within_20_percent() -> None:
    import random

    buckets = MethodBuckets(
        {"search": BucketSpec("search", 0, 1_000_000)},  # per_minute=0 → o'chiq
        jitter=0.2,
    )
    assert not buckets.try_acquire(1, "search")

    rng = random.Random(7)  # noqa: S311 — jitter testi, kriptografiya emas
    lively = MethodBuckets({"search": BucketSpec("search", 60, 100)}, jitter=0.2, rng=rng)
    start = lively.tokens_left(1, "search")
    for _ in range(50):
        lively.try_acquire(1, "search")
    used = start - lively.tokens_left(1, "search")
    assert 40 <= used <= 60  # 50 ta chaqiruv ±20% oralig'ida


# ─── config bilan bog'lanish ─────────────────────────────────────────────────


class _Settings:
    tg_search_rate_per_minute = 10  # eski sukut — alias ishlamaydi
    tg_rate_history_per_min = 40
    tg_rate_search_per_min = 12
    tg_rate_resolve_per_min = 4
    tg_rate_resolve_per_day = 60
    tg_rate_discover_per_min = 3
    tg_rate_discover_per_day = 30


def test_specs_from_settings_uses_the_new_fields() -> None:
    specs = specs_from_settings(_Settings())
    assert specs["history"].per_minute == 40
    assert specs["search"].per_minute == 12
    assert specs["resolve"].per_day == 60
    assert specs["discover"].per_day == 30


def test_legacy_env_var_still_drives_the_search_bucket() -> None:
    s = _Settings()
    s.tg_search_rate_per_minute = 2  # sukutdan farq qiladi → eski nom ustun
    specs = specs_from_settings(s)
    assert specs["search"].per_minute == 2
    assert specs["history"].per_minute == 40  # qolgan sinflarga ta'sir qilmaydi


def test_store_is_pluggable() -> None:
    """Redis'ga o'tish uchun faqat `BucketStore` almashtiriladi."""
    store = InMemoryBucketStore()
    buckets = MethodBuckets({"search": BucketSpec("search", 60, 1)}, store=store, jitter=0.0)
    assert buckets.try_acquire(1, "search")
    assert store.load("1:search") is not None
    assert not MethodBuckets(
        {"search": BucketSpec("search", 60, 1)}, store=store, jitter=0.0
    ).try_acquire(1, "search")  # holat umumiy saqlagichda
