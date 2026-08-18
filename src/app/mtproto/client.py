"""Guardrail o'rnatilgan Telethon klient.

Hech kim `TelegramClient`ni to'g'ridan-to'g'ri yaratmasin — faqat shu modul
orqali. `__call__` override qilingani uchun *hamma* TL so'rov, jumladan
Telethon'ning yuqori darajali `client.send_message()` kabi metodlari ham,
allowlist'dan o'tadi.
"""

from __future__ import annotations

from typing import Any

from telethon import TelegramClient
from telethon.sessions import StringSession

from app.config import get_settings
from app.logging import get_logger
from app.mtproto.allowlist import RpcBlocked, check_request, is_write

log = get_logger(__name__)


class GuardedTelegramClient(TelegramClient):  # type: ignore[misc]
    """Allowlist tekshiruvi bilan o'ralgan klient."""

    def __init__(self, *args: Any, account_id: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._account_id = account_id

    async def __call__(
        self, request: Any, ordered: bool = False, flood_sleep_threshold: Any = None
    ) -> Any:
        requests = request if isinstance(request, list | tuple) else [request]
        for req in requests:
            try:
                key = check_request(req)
            except RpcBlocked as exc:
                log.error(
                    "mtproto.blocked",
                    account_id=self._account_id,
                    method=exc.method,
                    reason=exc.reason,
                )
                raise
            if is_write(key):
                log.info("mtproto.write", account_id=self._account_id, method=key)
        return await super().__call__(
            request, ordered=ordered, flood_sleep_threshold=flood_sleep_threshold
        )


def build_client(
    session_string: str | None = None, *, account_id: int | None = None
) -> GuardedTelegramClient:
    """Yangi klient. `session_string=None` → yangi login uchun bo'sh session.

    device_* qiymatlari env'dan va o'zgarmas — har login'da bir xil profil
    ban riskini kamaytiradi (rejaning 4.2-bandi).
    """
    s = get_settings()
    return GuardedTelegramClient(
        StringSession(session_string) if session_string else StringSession(),
        api_id=s.tg_api_id,
        api_hash=s.tg_api_hash,
        device_model=s.tg_device_model,
        system_version=s.tg_system_version,
        app_version=s.tg_app_version,
        account_id=account_id,
        # Flood-wait'ni Telethon o'zi kutsin (60 soniyagacha), undan uzunini
        # bizning worker retry qiladi — 4.2 bandi.
        flood_sleep_threshold=60,
        connection_retries=5,
        retry_delay=2,
    )
