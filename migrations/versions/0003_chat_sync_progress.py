"""chats: sync progress columns

Revision ID: 0003
Revises: 0002
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chats", sa.Column("total_estimate", sa.Integer))
    op.add_column("chats", sa.Column("sync_error", sa.String(256)))
    op.add_column("chats", sa.Column("last_message_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_column("chats", "last_message_at")
    op.drop_column("chats", "sync_error")
    op.drop_column("chats", "total_estimate")
