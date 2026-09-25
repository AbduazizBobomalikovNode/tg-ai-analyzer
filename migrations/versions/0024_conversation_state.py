"""conversations.state: suhbatning ishchi holati (jsonb)

Revision ID: 0024
Revises: 0023

Har turn'dan keyin **kod** yozadi (`services/conv_state.py`), model emas:
aniqlangan odam/chat (`resolved_entities`), tanlangan chatlar, oxirgi vaqt
oynasi, oxirgi natijaning dalil havolalari va raqamli faktlari, javob turi,
mavzu (planner qayta yozgan savol) va foydalanuvchi tili. Planner uni
ixcham fakt bloki sifatida oladi — "u yana nima yozdi?" dagi "u" qayta
aniqlanmaydi, oyna meros qoladi.

Faqat id / sarlavha / raqam / foydalanuvchi matni — xabar matni **hech
qachon** (`_Taint.state_leak` saqlashdan oldin tekshiradi). Sukut `{}`:
eski suhbatlar bo'sh holat bilan davom etadi, backfill kerak emas.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0024"
down_revision = "0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column(
            "state",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("conversations", "state")
