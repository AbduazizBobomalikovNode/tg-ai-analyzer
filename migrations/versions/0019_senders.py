"""senders — jo'natuvchining ko'rinadigan nomi (`user:<id>` o'rniga)

Revision ID: 0019
Revises: 0018

Nega jadval kerak: `messages.sender_id` xom Telegram id, tool natijasi esa
modelga `user:124972562` bo'lib tushardi. Model javobda aynan shu qatorni
takrorlardi — foydalanuvchi uchun ma'nosiz, judge uchun esa "grounded emas"ga
o'xshaydi.

Ism **qo'shimcha RPC'siz** olinadi: `GetHistory` javobi bilan birga Telethon
`m.sender` keshini to'ldiradi, ingestion o'sha keshdan yozadi. `get_sender()` /
`users.GetUsers` chaqirilmaydi — allowlist chegarasi joyida qoladi.

PK `(account_id, tg_user_id)`: bir odam ikki akkauntda boshqacha (kontakt)
nom bilan ko'rinishi mumkin va bir akkauntning ko'rgani ikkinchisiga
oqib o'tmasligi kerak.

**Telefon raqami uchun ustun yo'q va qo'shilmaydi** — jadval qiymati LLM
promptiga tushadi (`tools._line`, `search_messages(from_user=…)`), raqam esa
u yerda kerak emas. `tests/test_tools_senders.py` shuni qo'riqlaydi.

Indekslar `search_messages(from_user=…)` uchun: nom/username bo'yicha
`ILIKE` prefiltri akkaunt ichida yuradi.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "senders",
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("tg_user_id", sa.BigInteger(), primary_key=True),
        sa.Column("display_name", sa.String(96), nullable=False, server_default=""),
        sa.Column("username", sa.String(64), nullable=True),
        sa.Column("is_bot", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_senders_account_username", "senders", ["account_id", "username"])
    op.create_index("ix_senders_account_name", "senders", ["account_id", "display_name"])


def downgrade() -> None:
    op.drop_index("ix_senders_account_name", table_name="senders")
    op.drop_index("ix_senders_account_username", table_name="senders")
    op.drop_table("senders")
