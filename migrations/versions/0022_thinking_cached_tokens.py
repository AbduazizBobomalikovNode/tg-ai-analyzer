"""conversation_messages: thinking va kesh tokenlari

Revision ID: 0022
Revises: 0021

Gemini fikrlash tokenlari (`thoughts_token_count`) output narxida to'lanadi,
lekin `candidates_token_count` ga kirmaydi — ilgari umuman yozilmasdi va
xarajat pastki chegara bo'lib qolardi. Keshdan o'qilgan input tokenlari
(`cached_content_token_count`) esa faqat log'da edi. Ikkala ustun ham
nullable: eski qatorlar uchun ma'lumot yo'q edi va uni 0 deb yozish yolg'on
bo'lardi (`limits.daily_usage` `coalesce` qiladi).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_messages", sa.Column("tokens_thinking", sa.Integer(), nullable=True)
    )
    op.add_column("conversation_messages", sa.Column("tokens_cached", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("conversation_messages", "tokens_cached")
    op.drop_column("conversation_messages", "tokens_thinking")
