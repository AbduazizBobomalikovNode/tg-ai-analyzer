"""phase 3/B: shaxsiy briefing (L3) — akkaunt sozlamalari + `briefings` jadvali

Revision ID: 0010
Revises: 0009

Nima uchun `accounts`, `users` emas:

  Briefing mazmuni **akkaunt bo'yicha** hosil bo'ladi — `retrieval_nodes`
  (level 2, kun tugunlari) va `chats` ikkalasi ham `account_id` ga bog'langan,
  `briefing_chats` esa aynan o'sha chat id'lari ro'yxati. Bir user bir necha
  Telegram akkaunt ulashi mumkin (`uq_account_per_user`), va ularning kun
  tartibi ham, jim soatlari ham bir xil bo'lishi shart emas.

  Boshqaruv boti esa **user** ga yozadi: `accounts.user_id → users.tg_user_id`.
  Ya'ni sozlama akkauntda, manzil user'da — invariant o'zgarmaydi: bot faqat
  o'z egasining shaxsiy chatiga push qiladi, ixtiyoriy peer'ga emas.

Nozik joylar:

  * `tz` — IANA nomi (`Asia/Tashkent`), offset emas: DST va qonun
    o'zgarishlarini `zoneinfo` hal qiladi.
  * `quiet_from`/`quiet_to` — `Time` (timezone'siz), **lokal** vaqt `tz` da.
    Yarim tun orqali o'ralishi mumkin (`22:00 → 07:00`) — mantiq
    `services/briefing.in_quiet_hours` da.
  * `briefings.day` — `Date`, akkauntning lokal kuni. `uq_briefing_account_day`
    bir kunga bitta qator kafolatlaydi (cron idempotent).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "accounts",
        sa.Column("tz", sa.String(64), nullable=False, server_default="Asia/Tashkent"),
    )
    op.add_column("accounts", sa.Column("quiet_from", sa.Time(), nullable=True))
    op.add_column("accounts", sa.Column("quiet_to", sa.Time(), nullable=True))
    op.add_column(
        "accounts",
        sa.Column("briefing_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "accounts",
        sa.Column("briefing_hour", sa.SmallInteger(), nullable=False, server_default="8"),
    )
    # null = barcha chatlar; aks holda `chats.id` ro'yxati
    op.add_column("accounts", sa.Column("briefing_chats", postgresql.JSONB, nullable=True))

    op.create_table(
        "briefings",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer,
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("day", sa.Date, nullable=False),
        sa.Column("text", sa.Text, nullable=False, server_default=""),
        sa.Column("structured", postgresql.JSONB, nullable=True),
        sa.Column("model", sa.String(64), nullable=False, server_default=""),
        sa.Column("tokens_in", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tokens_out", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_error", sa.String(500), nullable=True),
        sa.UniqueConstraint("account_id", "day", name="uq_briefing_account_day"),
    )
    op.create_index("ix_briefings_account_day", "briefings", ["account_id", "day"])


def downgrade() -> None:
    op.drop_index("ix_briefings_account_day", table_name="briefings")
    op.drop_table("briefings")
    for col in (
        "briefing_chats",
        "briefing_hour",
        "briefing_enabled",
        "quiet_to",
        "quiet_from",
        "tz",
    ):
        op.drop_column("accounts", col)
