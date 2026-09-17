"""phase 1/A1: retrieval poydevori — uz_fold SQL, generated column'lar, yangi ustunlar

Revision ID: 0008
Revises: 0007

Nima qilinadi:

  1. ``uz_norm(text)`` / ``uz_fold(text)`` — ``IMMUTABLE STRICT PARALLEL SAFE``
     SQL funksiyalari. Ular ``services/textnorm.py`` dagi Python versiyalari
     bilan **bayt-ma-bayt** bir xil natija berishi shart; buni
     ``tests/test_db_integration.py::test_uz_fold_sql_python_parity`` CI'da
     Postgres service bilan tekshiradi.
  2. ``messages`` — ``lang``/``lang_conf`` (ingest'da lingua bilan),
     forward manbasi (``fwd_from_peer_id``/``fwd_from_msg_id``/``fwd_date``),
     ``top_msg_id`` (forum topic / thread), ``deleted_at`` (soft-delete,
     2-bosqich updates listener uchun).
  3. ``messages`` — generated stored ustunlar ``text_fold``, ``tsv_uz``,
     ``tsv_ru`` va ular ustidagi GIN indekslar. Eski ifodali indekslar
     (``ix_messages_fts``, ``ix_messages_trgm``, 0001-migratsiya)
     **o'chiriladi**: ular endi takror va so'rovdagi ifoda bilan aynan
     mos kelmasligi mumkin edi (sekin skan sababi).
  4. ``chats.link_prefix`` — sitata havolasini runtime'da yig'ish o'rniga
     sync vaqtida bir marta hisoblanadi.
  5. ``chat_digests`` — ``src_fingerprint`` (``msg_count`` invalidatsiyasi
     tahrir va o'chirishni sezmaydi), ``level``, ``period_start``,
     ``structured``, ``faithfulness``.

Izoh — ``lower()`` va lokal: ``uz_norm`` Postgres ``lower()`` ga tayanadi.
UTF-8 lokalda (``en_US.utf8`` va h.k.) u Python ``str.lower()`` bilan bir
xil ishlaydi. Baza ``C``/``POSIX`` lokalda yaratilgan bo'lsa ``lower()``
faqat ASCII'ni pasaytiradi va kirill matn fold'dan o'tmay qoladi.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

# ─── uz_fold jadvallari (textnorm.py bilan sinxron) ──────────────────────────

# Ko'p belgili almashtirishlar — `replace()` zanjiri bilan, `translate()` dan
# OLDIN (translate faqat 1:1 bilan ishlaydi).
_MULTI: tuple[tuple[str, str], ...] = (
    ("ё", "yo"),
    ("ю", "yu"),
    ("я", "ya"),
    ("ц", "ts"),
    ("ч", "ch"),
    ("ш", "sh"),
    ("щ", "sh"),
)

# 1:1 + oxiridagi `ъ`/`ь` (nishoni yo'q → `translate()` ularni o'chiradi).
_TR_FROM = "абвгдежзийклмнопрстуфхыэғқҳўъь"
_TR_TO = "abvgdejziyklmnoprstufxiegqho"

# Apostrof sinfi — `textnorm.APOSTROPHES` bilan bir xil tartibda.
_APOSTROPHES = "'‘’‛ʻʼʹ´`ʽ"


def _sql_literal(value: str) -> str:
    """Python satrini SQL satr literaliga aylantiradi (faqat ``'`` qochiriladi)."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _uz_fold_body() -> str:
    expr = "uz_norm(t)"
    for src, dst in _MULTI:
        expr = f"replace({expr}, {_sql_literal(src)}, {_sql_literal(dst)})"
    return f"translate({expr}, {_sql_literal(_TR_FROM)}, {_sql_literal(_TR_TO)})"


def _create_functions() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION uz_norm(t text) RETURNS text AS $$
            SELECT regexp_replace(lower(normalize(t, NFC)),
                                  {_sql_literal("[" + _APOSTROPHES + "]")}, '', 'g')
        $$ LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE;
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION uz_fold(t text) RETURNS text AS $$
            SELECT {_uz_fold_body()}
        $$ LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE;
        """
    )


def upgrade() -> None:
    _create_functions()

    # ── messages: oddiy ustunlar ──
    op.add_column("messages", sa.Column("lang", sa.String(8), nullable=True))
    op.add_column("messages", sa.Column("lang_conf", sa.Float, nullable=True))
    op.add_column("messages", sa.Column("fwd_from_peer_id", sa.BigInteger, nullable=True))
    op.add_column("messages", sa.Column("fwd_from_msg_id", sa.BigInteger, nullable=True))
    op.add_column("messages", sa.Column("fwd_date", sa.DateTime(timezone=True), nullable=True))
    op.add_column("messages", sa.Column("top_msg_id", sa.BigInteger, nullable=True))
    op.add_column("messages", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    # ── messages: generated stored ustunlar ──
    op.execute(
        "ALTER TABLE messages ADD COLUMN text_fold text "
        "GENERATED ALWAYS AS (uz_fold(coalesce(text, ''))) STORED"
    )
    op.execute(
        "ALTER TABLE messages ADD COLUMN tsv_uz tsvector "
        "GENERATED ALWAYS AS (to_tsvector('simple', uz_fold(coalesce(text, '')))) STORED"
    )
    op.execute(
        "ALTER TABLE messages ADD COLUMN tsv_ru tsvector "
        "GENERATED ALWAYS AS (to_tsvector('russian', coalesce(text, ''))) STORED"
    )

    # Eski ifodali indekslar — yangi ustunlar bilan takrorlanadi.
    op.execute("DROP INDEX IF EXISTS ix_messages_fts")
    op.execute("DROP INDEX IF EXISTS ix_messages_trgm")

    op.execute("CREATE INDEX ix_messages_tsv_uz ON messages USING GIN (tsv_uz)")
    op.execute("CREATE INDEX ix_messages_tsv_ru ON messages USING GIN (tsv_ru)")
    op.execute("CREATE INDEX ix_messages_fold_trgm ON messages USING GIN (text_fold gin_trgm_ops)")
    op.execute(
        "CREATE INDEX ix_messages_top_msg ON messages (chat_id, top_msg_id) "
        "WHERE top_msg_id IS NOT NULL"
    )

    # ── chats ──
    op.add_column("chats", sa.Column("link_prefix", sa.String(128), nullable=True))

    # ── chat_digests ──
    op.add_column("chat_digests", sa.Column("src_fingerprint", sa.String(32), nullable=True))
    op.add_column(
        "chat_digests",
        sa.Column("level", sa.SmallInteger, nullable=False, server_default="1"),
    )
    op.add_column("chat_digests", sa.Column("period_start", sa.Date, nullable=True))
    op.add_column("chat_digests", sa.Column("structured", postgresql.JSONB, nullable=True))
    op.add_column("chat_digests", sa.Column("faithfulness", sa.Float, nullable=True))


def downgrade() -> None:
    op.drop_column("chat_digests", "faithfulness")
    op.drop_column("chat_digests", "structured")
    op.drop_column("chat_digests", "period_start")
    op.drop_column("chat_digests", "level")
    op.drop_column("chat_digests", "src_fingerprint")

    op.drop_column("chats", "link_prefix")

    op.execute("DROP INDEX IF EXISTS ix_messages_top_msg")
    op.execute("DROP INDEX IF EXISTS ix_messages_fold_trgm")
    op.execute("DROP INDEX IF EXISTS ix_messages_tsv_ru")
    op.execute("DROP INDEX IF EXISTS ix_messages_tsv_uz")

    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS tsv_ru")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS tsv_uz")
    op.execute("ALTER TABLE messages DROP COLUMN IF EXISTS text_fold")

    # 0001 dagi ifodali indekslarni tiklaymiz.
    op.execute(
        "CREATE INDEX ix_messages_fts ON messages "
        "USING GIN (to_tsvector('simple', coalesce(text, '')))"
    )
    op.execute("CREATE INDEX ix_messages_trgm ON messages USING GIN (text gin_trgm_ops)")

    for col in (
        "deleted_at",
        "top_msg_id",
        "fwd_date",
        "fwd_from_msg_id",
        "fwd_from_peer_id",
        "lang_conf",
        "lang",
    ):
        op.drop_column("messages", col)

    op.execute("DROP FUNCTION IF EXISTS uz_fold(text)")
    op.execute("DROP FUNCTION IF EXISTS uz_norm(text)")
