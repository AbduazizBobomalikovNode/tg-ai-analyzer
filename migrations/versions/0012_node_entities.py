"""phase 4/A: entity indeksi — `node_entities`

Revision ID: 0012
Revises: 0011

Nima uchun alohida jadval:

  `@nurbek`, `#deploy`, `example.com` — bular matn emas, **kalit**. Ularni
  `tsv` ichidan qidirish (a) `@` va `#` ni tashlab yuboradi, (b) reyting
  beradi, aniq tenglik esa reytingga muhtoj emas. Alohida jadval + b-tree
  indeks bilan "shu username qayerda tilga olingan" savoli bitta indeks
  o'qishga tushadi.

Nozik joylar:

  * ``kind`` — **matn + CHECK**, native PG enum emas: ro'yxat hali
    o'sishi mumkin (forum topic, geo), native enum'ga qiymat qo'shish esa
    migratsiya bilan `ALTER TYPE` talab qiladi. `CHECK` shu kuchni beradi
    va `phone_like` ni **bazada** to'sib turadi — telefon raqami indeksga
    tushmasligi kod xushmuomalaligiga emas, DB cheklovi bilan qo'riqlanadi.
  * ``message_id`` va ``node_id`` — ikkalasi ham nullable. Entity xabar
    bilan birga yoziladi, epizod esa keyinroq quriladi
    (`episodes.build_episodes` bitta `UPDATE` bilan `node_id` ni
    to'ldiradi). Epizod qayta qurilsa `ON DELETE SET NULL` — entity
    qatori o'chib ketmaydi, faqat bog'lanish uziladi.
  * ``uq_node_entity_value (message_id, kind, value)`` — idempotentlik
    bazada: qayta sync `ON CONFLICT DO NOTHING` ga tushadi va allaqachon
    to'ldirilgan `node_id` yo'qolmaydi.
  * ``ix_node_entities_account_kind_value`` — asosiy qidiruv yo'li
    (`search.entity_search`: akkaunt + tur + qiymat tengligi).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

#: `services/entities.KINDS` bilan **bir xil** ro'yxat. `phone_like` yo'q.
KINDS: tuple[str, ...] = (
    "username",
    "hashtag",
    "cashtag",
    "url_domain",
    "tme_link",
    "mention_name",
)


def upgrade() -> None:
    op.create_table(
        "node_entities",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer,
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chat_id", sa.Integer, sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "message_id",
            sa.BigInteger,
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "node_id",
            sa.BigInteger,
            sa.ForeignKey("retrieval_nodes.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("value", sa.Text, nullable=False),
        sa.Column("value_raw", sa.Text, nullable=False, server_default=""),
        sa.Column("seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("message_id", "kind", "value", name="uq_node_entity_value"),
        sa.CheckConstraint(
            "kind IN (" + ", ".join(f"'{k}'" for k in KINDS) + ")",
            name="ck_node_entity_kind",
        ),
    )
    op.execute(
        "CREATE INDEX ix_node_entities_account_kind_value ON node_entities "
        "(account_id, kind, value)"
    )
    op.execute("CREATE INDEX ix_node_entities_chat_seen ON node_entities (chat_id, seen_at DESC)")
    op.execute("CREATE INDEX ix_node_entities_node ON node_entities (node_id)")


def downgrade() -> None:
    op.drop_table("node_entities")
