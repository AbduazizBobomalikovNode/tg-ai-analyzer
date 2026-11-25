"""scheduled_jobs: eslatma yetkazish maydonlari (attempts / delivered_at / error)

Revision ID: 0025
Revises: 0024

`scheduled_jobs` 0001 dan beri bor, lekin bo'sh edi. Endi `type="reminder"`
qatorlarini `services/scheduled.py` yozadi (faqat `agent_actions` tasdig'idan
keyin) va `worker.tasks.deliver_reminders` egasining boshqaruv botiga push
qiladi. Yetkazish briefing/alert bilan bir xil naqshda: push'dan **oldin**
shartli UPDATE bilan band qilish (`delivered_at`, `attempts + 1`), xato
bo'lsa bo'shatish yoki `BRIEFING`-uslubida voz kechish — shu uchta ustun.

Qisman indeks: cron har daqiqada faqat `pending` va muddati kelganlarini
so'raydi; tarixdagi `sent`/`cancelled` qatorlar indeksga kirmaydi.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0025"
down_revision = "0024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "scheduled_jobs",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "scheduled_jobs", sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("scheduled_jobs", sa.Column("error", sa.String(500), nullable=True))
    op.create_index(
        "ix_scheduled_jobs_due",
        "scheduled_jobs",
        ["type", "run_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index("ix_scheduled_jobs_due", table_name="scheduled_jobs")
    op.drop_column("scheduled_jobs", "error")
    op.drop_column("scheduled_jobs", "delivered_at")
    op.drop_column("scheduled_jobs", "attempts")
