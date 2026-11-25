"""F2: yetkazish urinishlari + `messages.out` / `messages.created_at`

Revision ID: 0014
Revises: 0013

To'rtta ustun, to'rtta aniq muammo uchun:

  * ``alert_events.attempts`` / ``briefings.attempts`` — push abadiy
    qayta urinilmasin. Ilgari `delivered_at IS NULL` qolgan hodisa har
    yugurishda qayta yuborilardi: bot bloklangan bo'lsa navbat o'smasa
    ham, har 5 daqiqada bir xil xato takrorlanardi va **boshqa**
    qoidalarning hodisalari `MAX_EVENTS_PER_RUN` oynasida ularning
    ortida qolib ketardi. `ALERT_MAX_DELIVERY_ATTEMPTS` dan keyin
    hodisa `delivery_error` bilan yopiladi (`delivered_at = now`),
    ya'ni "yetkazildi" emas, "voz kechildi".

  * ``messages.out`` — xabarni **biz** yozganmizmi (Telethon `m.out`).
    Shaxsiy chatda `from_id` bo'sh keladi, shuning uchun
    "egasining xabari emas" filtri `sender_id` ga tayanib ishlamasdi:
    o'z yozgan xabaringiz o'z alertingizni uyg'otardi (va cooldown'ni
    yeb, haqiqiy xabarni yashirardi).

  * ``messages.created_at`` — qatorning **bazaga tushgan** payti
    (`published_at` Telegram vaqti, u ketma-ketlikni bermaydi).
    Alert kursori `max(messages.id)` gacha siljiydi; READ COMMITTED da
    esa hozir ochiq tranzaksiya kichikroq id bilan keyinroq ko'rinishi
    mumkin — bunday qator kursordan ortda qolib, hech qachon
    skanerlanmasdi. Kursor endi faqat "cho'kkan" (≥5 s oldin yozilgan)
    qatorlargacha siljiydi.

`ADD COLUMN ... DEFAULT` PG11+ da jadvalni qayta yozmaydi (`now()` —
stable), shuning uchun `messages` dagi ikki ustun katta jadvalda ham
tez qo'shiladi.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "alert_events",
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column(
        "briefings",
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
    )
    op.add_column(
        "messages",
        sa.Column("out", sa.Boolean, nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "messages",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    # Kursor klampi: "qamrov ichida, cho'kkan, eng katta id" so'rovi.
    op.create_index("ix_messages_created_at", "messages", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_messages_created_at", table_name="messages")
    op.drop_column("messages", "created_at")
    op.drop_column("messages", "out")
    op.drop_column("briefings", "attempts")
    op.drop_column("alert_events", "attempts")
