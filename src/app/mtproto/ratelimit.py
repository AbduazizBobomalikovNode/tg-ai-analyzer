"""Metod sinflari bo'yicha token bucket'lar — akkauntning MTProto kvotasi.

Nega bitta emas, to'rtta: bitta `TG_SEARCH_RATE_PER_MINUTE` shifti ostida
arzon `GetHistory` ham, qimmat `SearchPosts` ham bir xil kvotani yeydi —
natijada yo tarix o'qish keraksiz to'siladi, yo global qidiruv juda erkin
qoladi. Telegram ham ularni boshqacha o'lchaydi.

| bucket     | tezlik | burst | kunlik | kimlar                                  |
|------------|--------|-------|--------|-----------------------------------------|
| `history`  | 40/min | 8     | —      | GetHistory, GetReplies, GetParticipants |
| `search`   | 12/min | 4     | —      | Search, SearchGlobal, Calendar          |
| `resolve`  | 4/min  | 2     | 60     | contacts.ResolveUsername (cache-first)  |
| `discover` | 3/min  | 1     | 30     | ChannelRecommendations, SearchPosts     |

`try_acquire()` **bloklamaydi** — bo'sh bucket modelga matn bo'lib qaytadi
(`rate_limited:<bucket>`), u esa lokal qidiruvga tushadi. Kutish yo'q, chunki
kutish = LLM sikli ichida osilib qolish.

**±20% jitter**: har chaqiruvda token narxi tasodifiy 0.8-1.2 oralig'ida
o'zgaradi. Bu bir nechta jarayon (api va worker) bir vaqtda to'lganda ularning
so'rovlari bir xil ritmda "yopishib" qolishining oldini oladi.

**Saqlash joyi ulanadigan** (`BucketStore` protokoli). Hozir —
`InMemoryBucketStore`, ya'ni hisob **jarayon bo'yicha**: `api` va `worker`
har biri o'z nusxasini yaratadi, demak amaldagi shift taxminan ikki barobar.
Bu ataylab qabul qilingan (Redis hozircha shart emas): raqamlar Telegram
chegaralaridan ancha pastda. Umumiy hisob kerak bo'lganda — `BucketStore` ni
Redis ustida amalga oshirib, `MethodBuckets(store=...)` ga bering; qolgan kod
o'zgarmaydi.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = [
    "BUCKETS",
    "DEFAULT_SPECS",
    "BucketSpec",
    "BucketState",
    "BucketStore",
    "InMemoryBucketStore",
    "MethodBuckets",
    "specs_from_settings",
]

BUCKETS: tuple[str, ...] = ("history", "search", "resolve", "discover")

# `TG_SEARCH_RATE_PER_MINUTE` ning tarixiy sukut qiymati. Foydalanuvchi uni
# o'zgartirgan bo'lsa — eski nom yangi `search` bucket'ini boshqaradi.
LEGACY_SEARCH_RATE_DEFAULT = 10


@dataclass(frozen=True, slots=True)
class BucketSpec:
    """Bitta sinf uchun tezlik. `per_minute <= 0` — sinf butunlay o'chiq."""

    name: str
    per_minute: int
    burst: int
    per_day: int = 0  # 0 = kunlik chegara yo'q


DEFAULT_SPECS: dict[str, BucketSpec] = {
    "history": BucketSpec("history", 40, 8),
    "search": BucketSpec("search", 12, 4),
    "resolve": BucketSpec("resolve", 4, 2, per_day=60),
    "discover": BucketSpec("discover", 3, 1, per_day=30),
}


@dataclass(slots=True)
class BucketState:
    """Bitta (akkaunt, bucket) juftining holati."""

    tokens: float
    updated: float  # monotonik soat
    day: int  # UTC kun raqami (epoch // 86400)
    day_used: int


class BucketStore(Protocol):
    """Holat saqlagich. Redis versiyasi shu protokolni bajarishi kifoya."""

    def load(self, key: str) -> BucketState | None: ...

    def save(self, key: str, state: BucketState) -> None: ...


class InMemoryBucketStore:
    """Jarayon xotirasi — `web/security.RateLimiter` bilan bir xil yondashuv."""

    def __init__(self) -> None:
        self._data: dict[str, BucketState] = {}

    def load(self, key: str) -> BucketState | None:
        return self._data.get(key)

    def save(self, key: str, state: BucketState) -> None:
        self._data[key] = state


def specs_from_settings(settings: Any | None = None) -> dict[str, BucketSpec]:
    """Config'dan bucket tezliklari.

    `TG_SEARCH_RATE_PER_MINUTE` — **eskirgan alias**: agar u sukut qiymatidan
    farq qilsa, `search` bucket'ini u boshqaradi (eski o'rnatmalar buzilmasin).
    `0` esa avvalgidek serverga boradigan tool'larni butunlay o'chiradi —
    buni chaqiruvchi (`tools._tg_guard`) tekshiradi.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()
    legacy = int(getattr(settings, "tg_search_rate_per_minute", LEGACY_SEARCH_RATE_DEFAULT))
    search_rate = int(
        getattr(settings, "tg_rate_search_per_min", DEFAULT_SPECS["search"].per_minute)
    )
    if legacy != LEGACY_SEARCH_RATE_DEFAULT:
        search_rate = legacy
    return {
        "history": BucketSpec(
            "history",
            int(getattr(settings, "tg_rate_history_per_min", DEFAULT_SPECS["history"].per_minute)),
            DEFAULT_SPECS["history"].burst,
        ),
        "search": BucketSpec("search", search_rate, DEFAULT_SPECS["search"].burst),
        "resolve": BucketSpec(
            "resolve",
            int(getattr(settings, "tg_rate_resolve_per_min", DEFAULT_SPECS["resolve"].per_minute)),
            DEFAULT_SPECS["resolve"].burst,
            per_day=int(
                getattr(settings, "tg_rate_resolve_per_day", DEFAULT_SPECS["resolve"].per_day)
            ),
        ),
        "discover": BucketSpec(
            "discover",
            int(
                getattr(settings, "tg_rate_discover_per_min", DEFAULT_SPECS["discover"].per_minute)
            ),
            DEFAULT_SPECS["discover"].burst,
            per_day=int(
                getattr(settings, "tg_rate_discover_per_day", DEFAULT_SPECS["discover"].per_day)
            ),
        ),
    }


class UnknownBucket(KeyError):
    """Kodda yozilgan bucket nomi noto'g'ri — dasturchi xatosi."""


class MethodBuckets:
    """Akkaunt bo'yicha to'rtta token bucket, bloklamaydigan `try_acquire`."""

    def __init__(
        self,
        specs: dict[str, BucketSpec] | None = None,
        *,
        store: BucketStore | None = None,
        jitter: float = 0.2,
        clock: Any = time.monotonic,
        wall_clock: Any = time.time,
        rng: random.Random | None = None,
    ) -> None:
        self.specs = dict(specs or DEFAULT_SPECS)
        self.store: BucketStore = store or InMemoryBucketStore()
        self.jitter = max(0.0, min(float(jitter), 0.9))
        self._clock = clock
        self._wall = wall_clock
        self._rng = rng or random.Random()  # noqa: S311 — shift jitteri, kriptografiya emas

    @classmethod
    def from_settings(cls, settings: Any | None = None, **kw: Any) -> MethodBuckets:
        return cls(specs_from_settings(settings), **kw)

    # ── ichki ────────────────────────────────────────────────────────────────

    def _spec(self, bucket: str) -> BucketSpec:
        spec = self.specs.get(bucket)
        if spec is None:
            raise UnknownBucket(bucket)
        return spec

    def _key(self, account_id: int, bucket: str) -> str:
        return f"{account_id}:{bucket}"

    def _fresh(self, spec: BucketSpec) -> BucketState:
        return BucketState(
            tokens=float(spec.burst), updated=self._clock(), day=self._today(), day_used=0
        )

    def _today(self) -> int:
        return int(self._wall() // 86_400)

    def _refilled(self, spec: BucketSpec, state: BucketState) -> BucketState:
        now = self._clock()
        elapsed = max(0.0, now - state.updated)
        state.tokens = min(float(spec.burst), state.tokens + elapsed * spec.per_minute / 60.0)
        state.updated = now
        today = self._today()
        if state.day != today:
            state.day = today
            state.day_used = 0
        return state

    def _state(self, account_id: int, bucket: str) -> tuple[BucketSpec, BucketState]:
        spec = self._spec(bucket)
        state = self.store.load(self._key(account_id, bucket)) or self._fresh(spec)
        return spec, self._refilled(spec, state)

    # ── ommaviy API ──────────────────────────────────────────────────────────

    def try_acquire(self, account_id: int, bucket: str) -> bool:
        """Bitta token oladi. Bloklamaydi; muvaffaqiyatsiz bo'lsa `False`."""
        spec, state = self._state(account_id, bucket)
        key = self._key(account_id, bucket)
        if spec.per_minute <= 0:
            self.store.save(key, state)
            return False
        if spec.per_day and state.day_used >= spec.per_day:
            self.store.save(key, state)
            return False
        # M2: jitter narxi hech qachon `burst` dan oshmasin. `burst=1` bo'lgan
        # `discover` bucket'ida 1.0-1.2 oralig'idagi narx to'la bucket'da ham
        # rad etilardi — ya'ni chaqiruvlarning ~yarmi bekorga tushib qolardi.
        cost = min(1.0 + self.jitter * (2.0 * self._rng.random() - 1.0), float(spec.burst))
        if state.tokens < cost:
            self.store.save(key, state)
            return False
        state.tokens -= cost
        state.day_used += 1
        self.store.save(key, state)
        return True

    def retry_after(self, account_id: int, bucket: str) -> int | None:
        """Necha soniyadan keyin qayta urinish mantiqiy (≥1, holatni buzmaydi).

        M4: sinf butunlay o'chiq (`per_minute <= 0`) bo'lsa — `None`, ya'ni
        "hech qachon". Ilgari `0` qaytardi va chaqiruvchi uni "darhol qayta
        urin" deb o'qishi mumkin edi (spin).
        """
        spec, state = self._state(account_id, bucket)
        self.store.save(self._key(account_id, bucket), state)
        if spec.per_minute <= 0:
            return None
        if spec.per_day and state.day_used >= spec.per_day:
            # Kunlik chegara — ertangi UTC yarim tungacha
            return max(1, int(86_400 - (self._wall() % 86_400)))
        missing = max(0.0, 1.0 - state.tokens)
        return max(1, math.ceil(missing * 60.0 / spec.per_minute))

    def tokens_left(self, account_id: int, bucket: str) -> float:
        _, state = self._state(account_id, bucket)
        self.store.save(self._key(account_id, bucket), state)
        return state.tokens
