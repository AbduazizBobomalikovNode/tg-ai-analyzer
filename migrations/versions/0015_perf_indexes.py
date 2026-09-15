"""perf: `chat_id` bilan birlashtirilgan GIN indekslar + "tirik xabar" qopqog'i

Revision ID: 0015
Revises: 0014

Uch o'zgarish, uchalasi ham **o'lchovdan** keyin (300k xabarli sintetik
korpus, `evals/perf/`; raqamlar `docs/architecture-v2.md` §11.1 da):

  1. ``ix_messages_chat_trgm`` — ``GIN (chat_id, text_fold gin_trgm_ops)``,
     ``btree_gin`` kengaytmasi bilan. Eski ``ix_messages_fold_trgm`` faqat
     ``text_fold`` ustida edi, shuning uchun "bitta chat ichida trigram"
     so'rovi indeksdan **umuman foydalanmasdi**: planner ``chat_id``
     indeksini tanlab, chatdagi har qatorda ``strict_word_similarity`` ni
     hisoblardi (35k xabarli chatda 4 so'z uchun ≈4.4 s). Birlashtirilgan
     indeksda ikkala shart bitta skanga tushadi.

     Eski indeks **o'chiriladi**: birlashtirilgan indeks uni to'liq
     qoplaydi (GIN ko'p ustunli indeksda yetakchi ustun shart emas —
     ``text_fold %>> 'x'`` yolg'iz ham shu indeksdan o'qiydi) va o'lchovda
     **kichikroq** chiqdi: 49 MB (birlashtirilgan) vs 76 MB (eski).

  2. ``ix_messages_chat_msgid_alive`` — ``(chat_id, tg_msg_id)``
     ``WHERE deleted_at IS NULL``. `uq_msg_per_chat` bir xil ustunlarda,
     lekin u **unique** va o'chirilgan qatorlarni ham saqlaydi; deyarli
     barcha o'qish yo'llari (`node_hits_to_messages` oralig'i, `search`
     dumi, `window_context`, epizod qurish) `deleted_at IS NULL` bilan
     keladi. Qisman indeks kichikroq va planner uchun aniqroq.

  3. ``ix_messages_chat_created_id`` — ``(chat_id, created_at) INCLUDE (id)``.
     Alert kursori (`alerts._scope_max_id`) "qamrov ichida, cho'kkan, eng
     katta id" so'rovini har 5 daqiqada bajaradi; ``ix_messages_created_at``
     yolg'iz `chat_id` bo'yicha qirqa olmasdi.

Nega `CONCURRENTLY` emas: alembic migratsiyasi tranzaksiyada yuguradi va
`CREATE INDEX CONCURRENTLY` u yerda ishlamaydi. Katta bazada bu indekslarni
qo'lda, xizmatni to'xtatmasdan qurish mumkin (SQL shu yerda ko'rinib turibdi),
keyin migratsiyani `IF NOT EXISTS` tufayli jimgina o'tkazish mumkin.
"""

from __future__ import annotations

from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # `btree_gin` — GIN indeksida `chat_id` kabi oddiy skalar ustunni
    # tsvector/trigram bilan bir qatorga qo'yish uchun (contrib, PG bilan keladi).
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gin")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_messages_chat_trgm ON messages "
        "USING GIN (chat_id, text_fold gin_trgm_ops)"
    )
    op.execute("DROP INDEX IF EXISTS ix_messages_fold_trgm")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_messages_chat_msgid_alive ON messages "
        "(chat_id, tg_msg_id) WHERE deleted_at IS NULL"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_messages_chat_created_id ON messages "
        "(chat_id, created_at) INCLUDE (id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_messages_chat_created_id")
    op.execute("DROP INDEX IF EXISTS ix_messages_chat_msgid_alive")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_messages_fold_trgm ON messages "
        "USING GIN (text_fold gin_trgm_ops)"
    )
    op.execute("DROP INDEX IF EXISTS ix_messages_chat_trgm")
