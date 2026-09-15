"""phase 4/C: `embed_batches` — Gemini Batch API jobining holati

Revision ID: 0013
Revises: 0012

Nega jadval kerak:

  `batches.create_embeddings` darhol qaytadi, vektorlar esa soatlar keyin
  tayyor bo'ladi va ularni **boshqa jarayon** (keyingi cron yugurishi)
  o'qiydi. Job nomi va elementlar ro'yxati faqat xotirada tursa, worker
  qayta ishga tushishi bilan yo'qolardi: job serverda bajarilib, hisob
  to'lanib, natija hech qachon o'qilmay qolardi.

Nozik joylar:

  * ``item_ids jsonb`` — **tartiblangan** id massivi. Tartib yagona bog'lovchi:
    Gemini javoblarni "in the same order as the input requests" qaytaradi,
    ya'ni `vectors[i]` ↔ `item_ids[i]`. Shuning uchun poll paytida uzunlik
    **qat'iy** tekshiriladi (0-bosqich qoidasi) va mos kelmasa hech narsa
    yozilmaydi — aks holda vektor boshqa xabarga tushib, indeks jimgina
    buzilardi.

    `ARRAY(bigint)` emas, chunki bu ustun bo'ylab **konteynment** so'raladi
    ("bu xabar allaqachon uchayotgan jobdami?") va `jsonb_path_ops` GIN
    indeksi aynan shu savolga ishlaydi. Massivga ham GIN qo'yish mumkin
    edi, lekin `@>` semantikasi jsonb'da ixchamroq va ustun boshqa hech
    qayerda o'qilmaydi.

  * ``job_name`` unique — bir job ikki qator bo'lib tushmasin (submit
    takrorlansa yoki cron ustma-ust yugursa).

  * ``model`` — API **qaytargan** nom, so'ralgani emas (`pricing.py`
    qoidasi: alias qaytsa narx jadvalida o'sha nom bo'lishi kerak).

  * ``attempts`` — `max_attempts` dan keyin `failed_permanent`. Cheksiz
    qayta yuborish = cheksiz to'lov; qopqoqsiz "o'zi tuzaladi" siyosati
    bu yerda narx xatosi.

  * ``account_id`` nullable: `nodes` jobi bir nechta akkauntning
    tugunlarini birlashtirishi mumkin, `messages` jobi esa bitta
    akkauntga tegishli. `ON DELETE SET NULL` — akkaunt o'chsa job tarixi
    qoladi (xarajat hisoboti uchun), lekin o'sha job'ning elementlari
    baribir kaskad bilan yo'qolgan bo'ladi.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "embed_batches",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("provider", sa.String(32), nullable=False, server_default="gemini"),
        sa.Column("job_name", sa.String(255), nullable=False),
        # 'messages' | 'nodes'
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column(
            "account_id",
            sa.Integer,
            sa.ForeignKey("accounts.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "item_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("model", sa.String(64), nullable=False, server_default=""),
        sa.Column("task_type", sa.String(32), nullable=False, server_default="RETRIEVAL_DOCUMENT"),
        sa.Column("status", sa.String(24), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "submitted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.String(500), nullable=True),
        sa.Column("cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.UniqueConstraint("job_name", name="uq_embed_batches_job"),
        sa.CheckConstraint("kind in ('messages', 'nodes')", name="ck_embed_batches_kind"),
    )
    op.create_index("ix_embed_batches_status", "embed_batches", ["status", "kind"])
    op.execute(
        "CREATE INDEX ix_embed_batches_items ON embed_batches USING gin (item_ids jsonb_path_ops)"
    )


def downgrade() -> None:
    op.drop_index("ix_embed_batches_items", table_name="embed_batches")
    op.drop_index("ix_embed_batches_status", table_name="embed_batches")
    op.drop_table("embed_batches")
