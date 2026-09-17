"""ui/R1a: chat profili — about, avatar, belgilar, o'qilmagan, oxirgi xabar matni

Revision ID: 0017
Revises: 0016

Chat ro'yxati (redizayn, `docs/ui-research.md` §5) har qatorda avatar, bio va
o'qilmaganlar sonini ko'rsatadi. Bularning hech biri `chats` da yo'q edi —
UI ularni har ochilishda jonli MTProto bilan olishi kerak bo'lardi.

**"Media saqlanmaydi" invariantiga chegaralangan istisno.** Foydalanuvchi
aniq so'radi ("chatlarning imagelarini ham ko'rsat"), shuning uchun **faqat
profil rasmi** (kichik, `download_profile_photo`) `DATA_DIR/avatars` ga
yoziladi. Xabar mediasi (photo/video/document/voice) hech qachon yuklab
olinmaydi — bu siyosat kodda ham qo'riqlanadi: yagona yuklovchi modul
`services/avatars.py`, va `tests/test_avatars.py` boshqa modulda
`download_media` / `download_file` / `GetFileRequest` yo'qligini greplaydi.

Ustunlar:
  * `about` — `GetFull*` dan tavsif/bio (matn, NULL = hali olinmagan).
  * `is_verified` / `is_scam` / `is_fake` — Telegram belgilari (UI nishoni).
  * `unread_count`, `last_message_text` — `GetDialogs` javobidan, qo'shimcha
    RPC'siz (`services/ingestion.refresh_chats`).
  * `avatar_photo_id` — Telegram `photo.photo_id`; **o'zgarishni aniqlash**
    uchun: qiymat o'sha bo'lsa fayl qayta yuklanmaydi (RPC tejash).
  * `avatar_mime` / `avatar_updated_at` / `avatar_error` — fayl meta va
    oxirgi xatoning qisqa matni (FloodWait ham shu yerga tushadi, retry yo'q).

Fayllarning o'zi DB'da emas — `DATA_DIR/avatars/<chat_id>.<ext>`; downgrade
ularni o'chirmaydi (data volume'ga migratsiya tegmaydi).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0017"
down_revision = "0016"
branch_labels = None
depends_on = None

_COLUMNS = (
    "about",
    "is_verified",
    "is_scam",
    "is_fake",
    "unread_count",
    "last_message_text",
    "avatar_photo_id",
    "avatar_mime",
    "avatar_updated_at",
    "avatar_error",
)


def upgrade() -> None:
    op.add_column("chats", sa.Column("about", sa.Text(), nullable=True))
    op.add_column(
        "chats", sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false())
    )
    op.add_column(
        "chats", sa.Column("is_scam", sa.Boolean(), nullable=False, server_default=sa.false())
    )
    op.add_column(
        "chats", sa.Column("is_fake", sa.Boolean(), nullable=False, server_default=sa.false())
    )
    op.add_column(
        "chats", sa.Column("unread_count", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column("chats", sa.Column("last_message_text", sa.String(200), nullable=True))
    op.add_column("chats", sa.Column("avatar_photo_id", sa.BigInteger(), nullable=True))
    op.add_column("chats", sa.Column("avatar_mime", sa.String(32), nullable=True))
    op.add_column(
        "chats", sa.Column("avatar_updated_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("chats", sa.Column("avatar_error", sa.String(200), nullable=True))
    # Profil yangilash navbati: `about IS NULL` yoki avatar hali yuklanmagan
    # chatlar, eng eskisidan (`worker/tasks.refresh_chat_profiles`).
    op.create_index("ix_chats_profile_stale", "chats", ["account_id", "avatar_updated_at"])


def downgrade() -> None:
    op.drop_index("ix_chats_profile_stale", table_name="chats")
    for name in _COLUMNS:
        op.drop_column("chats", name)
