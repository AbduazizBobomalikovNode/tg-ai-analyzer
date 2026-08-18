"""stage 7: generated_images, auto_reply_rules

Revision ID: 0006
Revises: 0005
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "generated_images",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("account_id", sa.Integer, sa.ForeignKey("accounts.id", ondelete="SET NULL")),
        sa.Column("prompt", sa.Text, nullable=False, server_default=""),
        sa.Column("model", sa.String(64), nullable=False, server_default=""),
        sa.Column("mime", sa.String(32), nullable=False, server_default="image/png"),
        sa.Column("size_bytes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_generated_images_user_id", "generated_images", ["user_id"])

    op.create_table(
        "auto_reply_rules",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("account_id", sa.Integer, sa.ForeignKey("accounts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chat_id", sa.Integer, sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("trigger", sa.String(16), nullable=False, server_default="questions"),
        sa.Column("keywords", sa.Text, nullable=False, server_default=""),
        sa.Column("instructions", sa.Text, nullable=False, server_default=""),
        sa.Column("max_per_hour", sa.Integer, nullable=False, server_default="5"),
        sa.Column("quiet_from", sa.Integer),
        sa.Column("quiet_to", sa.Integer),
        sa.Column("last_processed_msg_id", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("chat_id", name="uq_autoreply_chat"),
    )
    op.create_index("ix_auto_reply_rules_account_id", "auto_reply_rules", ["account_id"])


def downgrade() -> None:
    op.drop_index("ix_auto_reply_rules_account_id", table_name="auto_reply_rules")
    op.drop_table("auto_reply_rules")
    op.drop_index("ix_generated_images_user_id", table_name="generated_images")
    op.drop_table("generated_images")
