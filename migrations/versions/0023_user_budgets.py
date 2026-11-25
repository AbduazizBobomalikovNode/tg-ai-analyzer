"""user_budgets: foydalanuvchi bo'yicha kunlik byudjet va vaqtincha qo'shimcha

Revision ID: 0023
Revises: 0022

Kunlik LLM byudjeti ilgari faqat global (`LLM_DAILY_TOKEN_BUDGET` /
`LLM_DAILY_COST_BUDGET_USD`) edi. Endi foydalanuvchi o'z limitini
Sozlamalar sahifasidan o'zgartira oladi va "bugun uchun qo'shimcha"
(boost) qo'sha oladi. Qator **foydalanuvchi** bo'yicha — sarf ham
foydalanuvchi bo'yicha sanaladi (`limits.daily_usage`), bir nechta
akkaunt bitta limitni bo'lishadi.

`token_budget` / `cost_budget_usd` NULL = global standart. Boost faqat
`boost_until > now()` bo'lsa qo'shiladi — muddati o'tgan qator o'zi
kuchini yo'qotadi, tozalash cron'i kerak emas.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_budgets",
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("token_budget", sa.BigInteger(), nullable=True),
        sa.Column("cost_budget_usd", sa.Numeric(12, 4), nullable=True),
        sa.Column("boost_tokens", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("boost_usd", sa.Numeric(12, 4), nullable=False, server_default="0"),
        sa.Column("boost_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("token_budget IS NULL OR token_budget >= 0", name="ck_ub_tokens"),
        sa.CheckConstraint("cost_budget_usd IS NULL OR cost_budget_usd >= 0", name="ck_ub_usd"),
        sa.CheckConstraint("boost_tokens >= 0 AND boost_usd >= 0", name="ck_ub_boost"),
    )


def downgrade() -> None:
    op.drop_table("user_budgets")
