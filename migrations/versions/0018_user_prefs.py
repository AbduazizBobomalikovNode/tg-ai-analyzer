"""users.prefs — UI sozlamalari (tema, til, default rejim, pin'lar)

Revision ID: 0018
Revises: 0017

Nega JSONB va nega alohida ustunlar emas: bu yerdagi kalitlar sof
**prezentatsiya** sozlamalari. Ular ustida hech qachon `WHERE`/`JOIN`
qilinmaydi (faqat egasi o'z qatorini o'qiydi), UI o'sgani sayin esa yangi
kalit qo'shiladi — har biriga migratsiya yozish narxi qiymatidan katta.

Xavfsizlik: ustun **ixtiyoriy JSON omborxonasi emas**. Yagona yozish nuqtasi
`services/prefs.py::merge`, u faqat `ALLOWED` ro'yxatidagi kalitlarni va
faqat validatsiyadan o'tgan qiymatlarni yozadi (noma'lum kalit jimgina
tashlanadi). Shuning uchun `PUT /api/settings/prefs` orqali cheksiz katta
yoki begona struktura kirib qololmaydi.

`NOT NULL DEFAULT '{}'` — o'qish yo'llarida `None` tekshiruvi kerak bo'lmasin.
Postgres 11+ da default bilan ustun qo'shish jadvalni qayta yozmaydi
(metadata-only), shuning uchun mavjud qatorlar soni muhim emas.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "prefs",
            JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "prefs")
