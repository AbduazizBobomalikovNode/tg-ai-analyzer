"""user_memories: foydalanuvchi xotirasi (alias / afzallik / fakt) + global kalit

Revision ID: 0026
Revises: 0025

Rozilik modeli (qaror): xotira **sukut bo'yicha yoqiq**, lekin to'liq
ko'rinadi — har saqlangan yozuv Sozlamalar → Xotira'da (kind, qiymat, manba
suhbat, sana) o'chirish / o'chirib qo'yish tugmasi bilan, `users.memory_enabled`
esa global kalit (false → hech narsa ishlatilmaydi va saqlanmaydi).

Yagona yozuvchi — `services/user_memory.py`. Faqat foydalanuvchining **o'z**
turn'idan olinadi (xabar matnidan hech qachon); telefon, kod, parol
saqlanmaydi (`user_memory.is_sensitive`). Dedup — kodda, fold qilingan kalit
bo'yicha; `uq_user_memory_key` esa (user, kind, lower(key)) bo'yicha oxirgi
to'siq. Foydalanuvchiga ≤100 yozuv (`MAX_ITEMS`, kodda).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0026"
down_revision = "0025"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_memories",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("key", sa.String(120), nullable=False),
        sa.Column("value", sa.String(300), nullable=False),
        sa.Column(
            "source_conversation_id",
            sa.Integer(),
            sa.ForeignKey("conversations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.CheckConstraint("kind IN ('fact', 'preference', 'alias')", name="ck_user_memory_kind"),
    )
    op.create_index("ix_user_memories_user", "user_memories", ["user_id"])
    op.create_index(
        "uq_user_memory_key",
        "user_memories",
        ["user_id", "kind", sa.text("lower(key)")],
        unique=True,
    )
    op.add_column(
        "users",
        sa.Column("memory_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    op.drop_column("users", "memory_enabled")
    op.drop_index("uq_user_memory_key", table_name="user_memories")
    op.drop_index("ix_user_memories_user", table_name="user_memories")
    op.drop_table("user_memories")
