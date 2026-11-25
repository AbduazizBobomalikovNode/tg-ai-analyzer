"""embedding jadvallari: PK → `(id, model)` — ikki model yonma-yon tursin

Revision ID: 0016
Revises: 0015

Muammo (F3/H1): `message_embeddings` da birlamchi kalit **faqat**
``message_id`` edi (`0001`), `retrieval_node_embeddings` da faqat ``node_id``
(`0009`). Shu bilan birga yozuvchilar (`worker/tasks.embed_messages`,
`worker/tasks.reembed_all`, `services/nodes.embed_nodes`,
`services/embed_batches`) "**joriy model** bo'yicha vektori yo'q" qatorlarni
tanlaydi. `EMBED_MODEL` almashtirilishi bilan ikkalasi to'qnashadi:

  * tanlov eski modelli qatorni "yetishmayapti" deb qaytaradi;
  * INSERT esa o'sha ``message_id`` ustidan `UniqueViolation` beradi;
  * cron har yugurishda o'sha qatorlarda qulab tushadi, `search._embed_model`
    esa yangi modelni filtrlaydi — vektor qidiruv **bo'sh** qoladi.

Ya'ni `docs/llm.md` dagi "qatorlar yonma-yon yoziladi" va'dasi sxema
darajasida **mumkin emas** edi. Bu migratsiya uni mumkin qiladi.

Nozik joylar:

  * ``model`` NOT NULL bo'ladi. Migratsiyada `get_settings()` o'qilmaydi
    (env migratsiya vaqtida boshqa bo'lishi mumkin), shuning uchun bo'sh/NULL
    qatorlar loyihaning yagona tarixiy embedding modeli — ``gemini-embedding-001``
    — deb belgilanadi (`config.Settings.gemini_model_embed` defaulti).
  * **Alohida `message_id` / `node_id` indeksi qo'shilmaydi**: yangi PK
    btree'sida bu ustun **yetakchi**, ya'ni FK ``ON DELETE CASCADE`` uchun
    kerakli ``WHERE message_id = ?`` skani o'sha indeksdan o'qiydi. Ikkinchi
    indeks 4 million qatorli jadvalda faqat bo'sh joy va yozuv narxi bo'lardi.
  * HNSW indekslari (``ix_embeddings_hnsw``, ``ix_node_emb_hnsw``) vektor
    ustuni ustida — PK constraint'iga bog'liq emas va **qayta qurilmaydi**
    (`tests/test_db_integration.py::test_embedding_pk_allows_two_models_side_by_side`
    buni DB'dan tekshiradi).
  * Downgrade **dedup qiladi**: bitta id uchun bir nechta model qatori bo'lsa
    eng yangisi (``created_at``, teng bo'lsa ``ctid``) qoladi — aks holda
    bir ustunli PK'ni tiklab bo'lmaydi. Ya'ni downgrade **ma'lumot yo'qotadi**,
    bu ataylab: teskari yo'lda boshqa tanlov yo'q.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None

#: Loyihaning yagona tarixiy embedding modeli (`config.Settings.gemini_model_embed`).
#: Migratsiyada `get_settings()` chaqirilmaydi — env bu yerda boshqa bo'lishi mumkin.
LEGACY_EMBED_MODEL = "gemini-embedding-001"

_TABLES = (("message_embeddings", "message_id"), ("retrieval_node_embeddings", "node_id"))


def upgrade() -> None:
    for table, key in _TABLES:
        op.execute(
            sa.text(
                f"UPDATE {table} SET model = :m "  # noqa: S608 — jadval nomi konstanta
                "WHERE model IS NULL OR btrim(model) = ''"
            ).bindparams(m=LEGACY_EMBED_MODEL)
        )
        op.alter_column(table, "model", existing_type=sa.String(64), nullable=False)
        op.drop_constraint(f"{table}_pkey", table, type_="primary")
        op.create_primary_key(f"{table}_pkey", table, [key, "model"])


def downgrade() -> None:
    for table, key in _TABLES:
        # Bir id uchun bir nechta model qatori — eng yangisi qoladi.
        op.execute(
            f"DELETE FROM {table} t USING ("  # noqa: S608 — jadval nomi konstanta
            f"  SELECT ctid FROM ("
            f"    SELECT ctid, row_number() OVER ("
            f"      PARTITION BY {key} ORDER BY created_at DESC NULLS LAST, ctid DESC"
            f"    ) AS rn FROM {table}"
            f"  ) ranked WHERE rn > 1"
            f") dead WHERE t.ctid = dead.ctid"
        )
        op.drop_constraint(f"{table}_pkey", table, type_="primary")
        op.create_primary_key(f"{table}_pkey", table, [key])
