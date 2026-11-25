"""scan_runs: bosqich bo'yicha to'xtash sabablari

Revision ID: 0021
Revises: 0020

`stopped_by` bitta ustun edi va uni **oxirgi** bosqich yozardi. Amalda
`filter_batched` hovuzni tugatib "complete" deb yozar, pull byudjeti qo'ygan
"wall" esa yo'qolardi — auditda "reja to'liq skanlandi" turar, holbuki
bucket'larning yarmi umuman qaralmagan edi (jonli yugurishlarda ushlandi).

Endi har bosqich o'z sababini yozadi, umumiysi esa `scale_search.combine_stops`
bilan hisoblanadi (wall/ceiling — yield/complete dan ustun). Eski qatorlar
bo'sh satr bilan qoladi: ular uchun bosqich ma'lumoti yo'q edi va uni
o'ylab topish yolg'on bo'lardi.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0021"
down_revision = "0020"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "scan_runs",
        sa.Column("pull_stopped_by", sa.String(16), nullable=False, server_default=""),
    )
    op.add_column(
        "scan_runs",
        sa.Column("filter_stopped_by", sa.String(16), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("scan_runs", "filter_stopped_by")
    op.drop_column("scan_runs", "pull_stopped_by")
