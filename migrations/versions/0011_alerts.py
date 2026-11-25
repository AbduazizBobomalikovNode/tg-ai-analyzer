"""phase 3/C: alertlar — `alert_rules` + `alert_events`

Revision ID: 0011
Revises: 0010

Nima uchun ikki jadval:

  `alert_rules` — foydalanuvchi qo'ygan kuzatuv (kalit so'z yoki ma'no).
  `alert_events` — **nima topilgani**. Ikkinchisi bo'lmasa "bu xabar
  haqida allaqachon xabar berilganmi" degan savolga javob yo'q, va har
  skanerdan keyin bir xil xabar qayta-qayta push bo'lardi.

  `uq_alert_event (rule_id, message_id)` — dedup **bazada**, kodda emas:
  skaner bir vaqtda ikki marta yugursa ham (worker qayta ishga tushdi,
  sync oxiridagi hook bilan cron ustma-ust tushdi) ikkinchi INSERT
  `ON CONFLICT DO NOTHING` bilan jimgina yo'qoladi.

Nozik joylar:

  * ``chat_id`` **nullable** — `NULL` = akkauntning barcha chatlari.
    Alohida "hamma" bayrog'i emas, chunki u holda ikki manba haqiqat
    bo'lardi (bayroq va id).
  * ``kind`` — native PG enum (`alert_kind`). Ikkita qiymat, o'zgarmaydi;
    `varchar` + CHECK bilan bir xil kuch, lekin autogenerate diffi tinch.
  * ``embedding halfvec(768)`` — **keshlangan** qoida vektori. Semantik
    qoida matni o'zgarmaguncha qayta embed qilinmaydi (har 5 daqiqalik
    skaner uchun bitta embedding chaqiruvi = bekorga xarajat).
    `pattern` yangilansa router `embedding` ni `NULL` ga tushiradi.
  * ``last_checked_msg_id`` — `messages.id` (ichki, monoton), `tg_msg_id`
    emas: qoida bir nechta chatni qamraydi, `tg_msg_id` esa chat ichida
    lokal. Kursor global bo'lishi shart.
  * HNSW indeks **yo'q**: `alert_rules` da bir necha o'nlab qator bo'ladi,
    ular bo'ylab qidirilmaydi — vektor faqat kesh sifatida o'qiladi.
"""

from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.config import get_settings

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None

EMBED_DIM = get_settings().embed_dim

# `create_type=False` — turni CREATE TYPE bilan o'zimiz yaratamiz, aks holda
# `create_table` uni ikkinchi marta yaratishga urinadi.
ALERT_KIND = postgresql.ENUM("keyword", "semantic", name="alert_kind", create_type=False)


def upgrade() -> None:
    op.execute("CREATE TYPE alert_kind AS ENUM ('keyword', 'semantic')")

    op.create_table(
        "alert_rules",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer,
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # NULL = akkauntning barcha chatlari
        sa.Column(
            "chat_id", sa.Integer, sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=True
        ),
        sa.Column("kind", ALERT_KIND, nullable=False, server_default="keyword"),
        sa.Column("label", sa.String(64), nullable=False, server_default=""),
        # keyword: vergul bilan ajratilgan so'zlar (fold fazosida moslanadi)
        # semantic: tabiiy tilda tavsif
        sa.Column("pattern", sa.Text, nullable=False, server_default=""),
        sa.Column("embedding", pgvector.sqlalchemy.HALFVEC(EMBED_DIM), nullable=True),
        sa.Column("threshold", sa.Float, nullable=False, server_default="0.80"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("cooldown_minutes", sa.Integer, nullable=False, server_default="30"),
        sa.Column("last_checked_msg_id", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.create_index("ix_alert_rules_account_enabled", "alert_rules", ["account_id", "enabled"])

    op.create_table(
        "alert_events",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "rule_id",
            sa.Integer,
            sa.ForeignKey("alert_rules.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "message_id",
            sa.BigInteger,
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chat_id", sa.Integer, sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("tg_msg_id", sa.BigInteger, nullable=False),
        sa.Column("score", sa.Float, nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_error", sa.String(500), nullable=True),
        sa.UniqueConstraint("rule_id", "message_id", name="uq_alert_event"),
    )
    op.execute("CREATE INDEX ix_alert_events_rule_time ON alert_events (rule_id, created_at DESC)")


def downgrade() -> None:
    op.drop_index("ix_alert_events_rule_time", table_name="alert_events")
    op.drop_table("alert_events")
    op.drop_index("ix_alert_rules_account_enabled", table_name="alert_rules")
    op.drop_table("alert_rules")
    op.execute("DROP TYPE alert_kind")
