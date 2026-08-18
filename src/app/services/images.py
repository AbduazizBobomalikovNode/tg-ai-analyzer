"""AI rasm generatsiya (7-bosqich) — `Task.IMAGE` (Gemini) → fayl + DB meta.

Fayl `DATA_DIR/images/<uuid>.png` (compose'da `./data` volume), meta
`generated_images`. Berish faqat egasiga (`/api/images/{id}` cookie bilan).
Kunlik limit `IMAGE_MAX_PER_DAY` (foydalanuvchi bo'yicha) — xarajat qopqog'i.
Rasm Telegram'ga faqat `send_message` taklifi (`image_id`) tasdiqlanganda ketadi
(`messages.SendMedia` + `upload.SaveFilePart` — allowlist'da).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import GeneratedImage
from app.llm import LLM, LLMError
from app.logging import get_logger

log = get_logger(__name__)

MAX_PROMPT = 2000
_PNG = b"\x89PNG\r\n\x1a\n"
_JPEG = b"\xff\xd8\xff"


class ImageError(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code  # i18n: image.err.<code>
        self.detail = detail
        super().__init__(code)


@dataclass(slots=True)
class ImageInfo:
    id: str
    url: str
    mime: str
    size_bytes: int
    prompt: str
    model: str


def images_dir() -> Path:
    d = Path(get_settings().data_dir) / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def image_path(image_id: str, mime: str = "image/png") -> Path:
    ext = "jpg" if mime == "image/jpeg" else "png"
    return images_dir() / f"{image_id}.{ext}"


def _sniff(data: bytes) -> str:
    if data.startswith(_PNG):
        return "image/png"
    if data.startswith(_JPEG):
        return "image/jpeg"
    return "image/png"


def _url(image_id: str) -> str:
    return f"/api/images/{image_id}"


async def _today_count(session: AsyncSession, user_id: int) -> int:
    since = datetime.now(UTC) - timedelta(days=1)
    n = (
        await session.execute(
            select(func.count(GeneratedImage.id)).where(
                GeneratedImage.user_id == user_id, GeneratedImage.created_at >= since
            )
        )
    ).scalar_one()
    return int(n or 0)


async def generate(
    session: AsyncSession,
    *,
    user_id: int,
    account_id: int | None,
    prompt: str,
    style: str = "",
    llm: LLM | None = None,
) -> ImageInfo:
    s = get_settings()
    if not s.image_gen_enabled:
        raise ImageError("disabled")
    prompt = prompt.strip()
    if not prompt:
        raise ImageError("empty_prompt")
    if len(prompt) > MAX_PROMPT:
        raise ImageError("prompt_too_long")
    if await _today_count(session, user_id) >= s.image_max_per_day:
        raise ImageError("daily_limit")

    full = f"{prompt}. Style: {style.strip()}" if style.strip() else prompt
    try:
        data = await (llm or LLM()).generate_image(full)
    except LLMError as exc:
        raise ImageError("llm", str(exc)[:200]) from exc
    if not data or len(data) < 64:
        raise ImageError("llm", "empty image")

    mime = _sniff(data)
    image_id = str(uuid.uuid4())
    path = image_path(image_id, mime)
    path.write_bytes(data)
    row = GeneratedImage(
        id=image_id,
        user_id=user_id,
        account_id=account_id,
        prompt=prompt[:MAX_PROMPT],
        model="",  # router hal qiladi; keyingi versiyada natijadan
        mime=mime,
        size_bytes=len(data),
    )
    session.add(row)
    await session.flush()
    log.info("image.generated", image_id=image_id, user_id=user_id, bytes=len(data))
    return ImageInfo(image_id, _url(image_id), mime, len(data), prompt, row.model)


async def get_owned(
    session: AsyncSession, *, user_id: int, image_id: str
) -> tuple[GeneratedImage, Path]:
    row = await session.get(GeneratedImage, image_id)
    if row is None or row.user_id != user_id:
        raise ImageError("not_found")
    path = image_path(row.id, row.mime)
    if not path.exists():
        raise ImageError("not_found", "file missing")
    return row, path


async def path_for_send(session: AsyncSession, *, image_id: str, account_id: int) -> Path:
    """Yuborish uchun (tasdiqlangan amal): rasm shu akkaunt egasiniki bo'lishi shart."""
    from app.db.models import Account

    row = await session.get(GeneratedImage, image_id)
    if row is None:
        raise ImageError("not_found")
    acc = await session.get(Account, account_id)
    if acc is None or acc.user_id != row.user_id:
        raise ImageError("not_found")
    path = image_path(row.id, row.mime)
    if not path.exists():
        raise ImageError("not_found", "file missing")
    return path
