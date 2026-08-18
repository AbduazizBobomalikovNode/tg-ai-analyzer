"""Peer deny-list — agent boshqaruv botiga hech qachon yoza olmasligi kafolati.

Nega kerak: agent boshqarilayotgan akkaunt nomidan ishlaydi. Agar u o'zini
boshqarayotgan botga xabar yozsa — bot uni yangi buyruq deb o'qiydi va
cheksiz o'zini-o'zi qo'zg'atuvchi sikl (yoki self-injection hujumi) paydo
bo'ladi. Shuning uchun bu peer'lar:

  * har `send/edit/forward/pin/reaction` oldidan bloklanadi;
  * inline chat tanlash ro'yxatidan ham filtrlanadi — user tasodifan
    tanlay olmasin.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from app.config import get_settings

TELEGRAM_SERVICE_ID = 777000  # "Telegram" servis xabarlari — login kodlari shu yerda
REPLIES_BOT_ID = 1271266957  # @replies


class PeerBlocked(PermissionError):
    def __init__(self, peer_id: int, reason: str) -> None:
        self.peer_id = peer_id
        self.reason = reason
        super().__init__(f"Peer bloklangan: {peer_id} — {reason}")


def protected_peers(extra: frozenset[int] = frozenset()) -> frozenset[int]:
    s = get_settings()
    return frozenset({s.control_bot_id, TELEGRAM_SERVICE_ID, REPLIES_BOT_ID}) | extra


def is_protected(peer_id: int, *, extra: frozenset[int] = frozenset()) -> bool:
    return abs(peer_id) in {abs(p) for p in protected_peers(extra)}


def assert_writable(peer_id: int, *, extra: frozenset[int] = frozenset()) -> None:
    """Yozishdan oldingi qattiq tekshiruv."""
    s = get_settings()
    if abs(peer_id) == abs(s.control_bot_id):
        raise PeerBlocked(peer_id, "boshqaruv boti — agent bu yerga yoza olmaydi")
    if abs(peer_id) == TELEGRAM_SERVICE_ID:
        raise PeerBlocked(peer_id, "Telegram servis akkaunti")
    if is_protected(peer_id, extra=extra):
        raise PeerBlocked(peer_id, "himoyalangan peer")


def filter_visible[T](items: Iterable[T], peer_id_of: Callable[[T], int]) -> list[T]:
    """Chat ro'yxatidan himoyalangan peer'larni olib tashlaydi.

    Inline chat tanlashda ishlatiladi — user boshqaruv botini tanlay olmasin.
    """
    return [it for it in items if not is_protected(peer_id_of(it))]
