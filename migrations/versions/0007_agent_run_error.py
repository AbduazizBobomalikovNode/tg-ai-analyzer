"""phase 0: agent_runs.error

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("error", sa.String(500), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "error")
