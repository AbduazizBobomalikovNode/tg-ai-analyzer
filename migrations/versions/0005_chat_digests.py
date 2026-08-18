"""chat_digests: daily digest cache

Revision ID: 0005
Revises: 0004
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_digests",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "chat_id", sa.Integer, sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("day", sa.DateTime(timezone=True), nullable=False),
        sa.Column("digest", sa.Text, nullable=False, server_default=""),
        sa.Column("msg_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("model", sa.String(64), nullable=False, server_default=""),
        sa.Column("tokens_in", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tokens_out", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("chat_id", "day", name="uq_digest_chat_day"),
    )
    op.create_index("ix_chat_digests_chat_id", "chat_digests", ["chat_id"])
    op.create_index("ix_chat_digests_day", "chat_digests", ["day"])


def downgrade() -> None:
    op.drop_index("ix_chat_digests_day", table_name="chat_digests")
    op.drop_index("ix_chat_digests_chat_id", table_name="chat_digests")
    op.drop_table("chat_digests")
