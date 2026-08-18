"""conversation_messages: quality + cost columns (dashboard)

Revision ID: 0004
Revises: 0003
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

COLS = (
    sa.Column("task", sa.String(16), nullable=False, server_default=""),
    sa.Column("latency_ms", sa.Integer),
    sa.Column("cost_usd", sa.Float),
    sa.Column("rating", sa.Integer),
    sa.Column("rating_comment", sa.String(512)),
    sa.Column("rated_at", sa.DateTime(timezone=True)),
    sa.Column("auto_relevance", sa.Integer),
    sa.Column("auto_usefulness", sa.Integer),
    sa.Column("auto_grounded", sa.Boolean),
    sa.Column("auto_note", sa.String(256)),
)


def upgrade() -> None:
    for col in COLS:
        op.add_column("conversation_messages", col)
    op.create_index(
        "ix_conversation_messages_created_at", "conversation_messages", ["created_at"]
    )


def downgrade() -> None:
    op.drop_index("ix_conversation_messages_created_at", table_name="conversation_messages")
    for col in reversed(COLS):
        op.drop_column("conversation_messages", col.name)
