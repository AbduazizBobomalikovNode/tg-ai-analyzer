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

# Rasm generatsiya narxi **rasm boshiga** e'lon qilinadi; `llm/pricing.py`
# esa tokenda hisoblaydi. Gemini'ning o'z ekvivalenti: bitta rasm = 1290
# output token ($30/1M → ≈ $0.039). Taxmin — narx o'zgarsa jadval bilan birga
# shu son ham yangilanadi.
IMAGE_OUTPUT_TOKENS = 1290
# Prompt uchun taxminiy input tokenlar (4 belgi ≈ 1 token) — API bu chaqiruvda
# usage qaytarmaydi, byudjet qopqog'i uchun kam baholamaslik muhim.
_CHARS_PER_TOKEN = 4
IMAGE_CONVERSATION = "Images"


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
    provider_name, model_name = _image_model()
    row = GeneratedImage(
        id=image_id,
        user_id=user_id,
        account_id=account_id,
        prompt=prompt[:MAX_PROMPT],
        model=model_name[:64],
        mime=mime,
        size_bytes=len(data),
    )
    session.add(row)
    await session.flush()
    await _record_image_usage(
        session,
        user_id=user_id,
        account_id=account_id,
        prompt=full,
        provider=provider_name,
        model=model_name,
    )
    log.info("image.generated", image_id=image_id, user_id=user_id, bytes=len(data))
    return ImageInfo(image_id, _url(image_id), mime, len(data), prompt, row.model)


def _image_model() -> tuple[str, str]:
    """(provider, model) — chaqiruvsiz, faqat router qarori."""
    try:
        from app.llm.router import Task, resolve

        provider, model = resolve(Task.IMAGE)
        return str(provider.name), str(model)
    except Exception:  # router sozlanmagan bo'lsa xarajat yozuvi 0 bo'ladi
        return "", ""


async def _record_image_usage(
    session: AsyncSession,
    *,
    user_id: int,
    account_id: int | None,
    prompt: str,
    provider: str,
    model: str,
) -> None:
    """Rasm xarajatini kunlik byudjet yo'liga qo'shadi (briefing/judge bilan bir uslub).

    `IMAGE_MAX_PER_DAY` soni bo'yicha qopqoq edi, **pul** bo'yicha emas:
    `limits.daily_usage` faqat `conversation_messages` ni ko'radi, rasm esa u
    yerga tushmasdi. Endi har rasm egasining "Images" suhbatiga `assistant`
    qatori bo'lib yoziladi. Tokenlar **taxmin** (API usage qaytarmaydi) —
    `IMAGE_OUTPUT_TOKENS` izohiga qarang.
    """
    from sqlalchemy import select

    from app.db.models import Conversation, ConversationMessage
    from app.llm.pricing import estimate_cost

    tokens_in = max(1, len(prompt) // _CHARS_PER_TOKEN)
    tokens_out = IMAGE_OUTPUT_TOKENS
    conv = (
        await session.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id, Conversation.title == IMAGE_CONVERSATION)
            .limit(1)
        )
    ).scalar_one_or_none()
    if conv is None:
        conv = Conversation(user_id=user_id, account_id=account_id, title=IMAGE_CONVERSATION)
        session.add(conv)
        await session.flush()
    cost = estimate_cost(provider, model, tokens_in, tokens_out) or 0.0
    session.add(
        ConversationMessage(
            conversation_id=conv.id,
            role="assistant",
            content="[image]",
            model=model[:64],
            provider=provider[:32],
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            task="image",
            cost_usd=round(cost, 6) or None,
        )
    )
    await session.flush()


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
