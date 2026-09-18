"""relevance_cache + scan_runs — korpus miqyosidagi kaskad (deep search v3)

Revision ID: 0020
Revises: 0019

Ikkita jadval, ikkita alohida sabab.

`relevance_cache` — **narx qopqog'i**. Kaskadning `filter` bosqichi har
nomzodni Flash-Lite'ga ko'rsatadi; bir savolni ikki marta so'rash (yoki
"chuqurroq qidir" bilan qaytish) o'sha xabarlarni yana modelga yuborardi.
Natija o'zgarmaydi, token esa behuda ketadi — bu `seen_calls` bilan aynan
bir mantiq. Kalit `q_hash = md5(uz_fold(standalone_question))`: savolning
imlo/apostrof/kirill farqi kesh promahiga aylanmasin.

**Xabar matni saqlanmaydi** — faqat `score`. Kesh qatori LLM promptiga
qaytib tushmaydi; u faqat "bu juftlikni yana so'ramaymiz" degan belgi.
Shuning uchun unikal cheklov `(q_hash, chat_id, tg_msg_id)` da: `chat_id`
allaqachon akkauntga bog'langan, ya'ni akkauntlararo oqish yo'q.

`scan_runs` — **halollik hisobi**. `agent_runs` token va narxni yozadi,
qamrovni emas: nechta bucket rejalashtirilgan, nechtasi skanlangan, nega
to'xtagan. Javob ostidagi qamrov bloki shu raqamlardan quriladi, shuning
uchun ular jimgina yo'qolmasligi kerak. Kod bu jadvalga faqat INSERT
qiladi (UPDATE yo'q), lekin `agent_actions` dagidek trigger qo'yilmagan:
bu audit emas, o'lchov.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "relevance_cache",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("q_hash", sa.String(32), nullable=False),
        sa.Column(
            "chat_id", sa.Integer(), sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("tg_msg_id", sa.BigInteger(), nullable=False),
        sa.Column("score", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("model", sa.String(64), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("q_hash", "chat_id", "tg_msg_id", name="uq_relcache_q_chat_msg"),
    )
    # Tozalash (`DELETE ... WHERE account_id = ... AND created_at < ...`) uchun.
    op.create_index(
        "ix_relcache_account_created",
        "relevance_cache",
        ["account_id", sa.text("created_at DESC")],
    )

    op.create_table(
        "scan_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "account_id",
            sa.Integer(),
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "conversation_id",
            sa.Integer(),
            sa.ForeignKey("conversations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("mode", sa.String(16), nullable=False, server_default="deep"),
        sa.Column("q_hash", sa.String(32), nullable=False, server_default=""),
        sa.Column("buckets_planned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("buckets_scanned", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("candidates", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("filtered", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("kept", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("evidence", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("claims", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cache_hits", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("est_cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("stopped_by", sa.String(16), nullable=False, server_default="complete"),
        sa.Column("truncated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("elapsed_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_scan_runs_account_created", "scan_runs", ["account_id", sa.text("created_at DESC")]
    )


def downgrade() -> None:
    op.drop_index("ix_scan_runs_account_created", table_name="scan_runs")
    op.drop_table("scan_runs")
    op.drop_index("ix_relcache_account_created", table_name="relevance_cache")
    op.drop_table("relevance_cache")
