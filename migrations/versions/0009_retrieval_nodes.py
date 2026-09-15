"""phase 2/A: ierarxik retrieval indeksi — `retrieval_nodes` + `halfvec` embeddinglar

Revision ID: 0009
Revises: 0008

Nima uchun:

  Xabar darajasidagi qidiruv "ha, roziman" tipidagi qatorlarda ishlamaydi —
  kontekst yo'q. Shuning uchun ustiga **ierarxiya** quriladi: epizod (level 1,
  LLM'siz segmentatsiya), kun (level 2, `chat_digests` ko'zgusi), hafta
  (level 3, faqat kun tugunlaridan), oy (level 4, keyinroq). Har tugun o'z
  `ctx_prefix` i bilan embed qilinadi (arxitektura v2, 5.3-band).

  **Level 0 ko'chirilmaydi**: xabarlar `messages` da, vektorlari
  `message_embeddings` da qoladi. 4 million dublikat qator yaratilmaydi.

Nozik joylar:

  * ``tsv`` generated ustuni ``ctx_prefix || ' ' || text`` ustida quriladi —
    chat nomi va so'zlovchilar ham leksik kanalga tushadi. ``uz_fold``
    ``STRICT`` (0008), shuning uchun ``coalesce`` ikkala tomonda ham shart.
  * ``content_hash`` — ``varchar(32)`` (md5 hex), `chat_digests.src_fingerprint`
    bilan bir xil shakl. Qayta qurish kerakmi degan savolga bitta taqqoslash
    javob beradi.
  * ``simhash`` — **belgili** bigint. `textnorm.simhash64` 0..2^64 oralig'ida
    ishorasiz son qaytaradi, Postgres `bigint` esa belgili: yozishdan oldin
    `nodes.to_signed64` bilan siljitiladi.
  * ``vector halfvec(768)`` — pgvector ≥0.7 (bu yerda tekshirilgani 0.8.6,
    `pgvector/pgvector:pg16`). Xotira ~2x, aniqlik yo'qotishi o'lchovda
    sezilmadi (v2, 3.3-band).
"""

from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.config import get_settings

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

EMBED_DIM = get_settings().embed_dim

#: `tsv` generated ustunining ifodasi — `db/models.py::RetrievalNode.tsv` bilan
#: **bayt-ma-bayt** bir xil bo'lishi shart, aks holda alembic autogenerate har
#: safar bo'sh diff o'rniga ALTER taklif qiladi.
TSV_EXPR = "to_tsvector('simple', uz_fold(coalesce(ctx_prefix, '') || ' ' || coalesce(text, '')))"


def upgrade() -> None:
    op.create_table(
        "retrieval_nodes",
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
        # 0 msg (saqlanmaydi) | 1 epizod | 2 kun | 3 hafta | 4 oy
        sa.Column("level", sa.SmallInteger, nullable=False),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("msg_lo", sa.BigInteger, nullable=False),
        sa.Column("msg_hi", sa.BigInteger, nullable=False),
        sa.Column("msg_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "speakers",
            postgresql.ARRAY(sa.BigInteger),
            nullable=False,
            server_default=sa.text("'{}'::bigint[]"),
        ),
        sa.Column("text", sa.Text, nullable=False, server_default=""),
        sa.Column("ctx_prefix", sa.Text, nullable=False, server_default=""),
        sa.Column("content_hash", sa.String(32), nullable=False),
        sa.Column("simhash", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("structured", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("chat_id", "level", "starts_at", name="uq_node_chat_level_start"),
    )
    op.execute(
        "ALTER TABLE retrieval_nodes ADD COLUMN tsv tsvector "
        f"GENERATED ALWAYS AS ({TSV_EXPR}) STORED"
    )
    op.execute(
        "CREATE INDEX ix_nodes_account_level_time ON retrieval_nodes "
        "(account_id, level, starts_at DESC)"
    )
    op.execute(
        "CREATE INDEX ix_nodes_chat_level_time ON retrieval_nodes (chat_id, level, starts_at)"
    )
    op.execute("CREATE INDEX ix_nodes_tsv ON retrieval_nodes USING GIN (tsv)")
    op.execute("CREATE INDEX ix_nodes_simhash ON retrieval_nodes (simhash)")

    op.create_table(
        "retrieval_node_embeddings",
        sa.Column(
            "node_id",
            sa.BigInteger,
            sa.ForeignKey("retrieval_nodes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("vector", pgvector.sqlalchemy.HALFVEC(EMBED_DIM), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
    )
    op.execute(
        "CREATE INDEX ix_node_emb_hnsw ON retrieval_node_embeddings "
        "USING hnsw (vector halfvec_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    op.execute("CREATE INDEX ix_node_emb_model ON retrieval_node_embeddings (model)")


def downgrade() -> None:
    op.drop_table("retrieval_node_embeddings")
    op.drop_table("retrieval_nodes")
