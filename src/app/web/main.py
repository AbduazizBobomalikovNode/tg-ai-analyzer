"""Mini App backend.

0-bosqichda faqat healthcheck. 1-bosqichda bu yerga qo'shiladi:
  * `POST /auth/qr/start`     — QR login tokeni
  * `GET  /auth/qr/poll`      — login holati
  * `POST /auth/code/send`    — Mini App fallback (telefon)
  * `POST /auth/code/verify`  — kod + 2FA

MUHIM: login kodi va 2FA paroli faqat shu HTTPS endpoint orqali qabul
qilinadi. Chat orqali hech qachon — Telegram chat'da yuborilgan kodni
bekor qiladi (rejaning 4.1-bandi).
"""

from __future__ import annotations

from fastapi import FastAPI

from app.config import get_settings
from app.logging import setup_logging

s = get_settings()
setup_logging(s.log_level, json_output=s.is_prod)

app = FastAPI(title="tg-ai-analyzer", version="0.1.0", docs_url=None if s.is_prod else "/docs")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "env": s.env}
