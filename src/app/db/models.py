"""ORM modellar.

Dizayn qarorlari:
  * Media saqlanmaydi — faqat `media_type` belgisi (talab bo'yicha to'liq ignore).
  * `MessageMetricSnapshot` — views/reactions vaqt qatori. Telegram tarixiy
    qiymat bermaydi, shuning uchun 1-kundan snapshot yig'ish shart, aks holda
    "bu hafta qancha ko'rish qo'shildi" savoliga hech qachon javob bo'lmaydi.
  * `AgentAction` — immutable audit log. Har bir MTProto write shu yerga tushadi.
"""

from __future__ import annotations

from datetime import date, datetime, time
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import HALFVEC, Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db.base import Base


class AccountStatus(StrEnum):
    PENDING = "pending"  # QR/kod kutilmoqda
    ACTIVE = "active"
    NEEDS_2FA = "needs_2fa"
    REVOKED = "revoked"  # user logout qildi yoki session o'lgan
    BANNED = "banned"


class ChatType(StrEnum):
    PRIVATE = "private"
    GROUP = "group"
    SUPERGROUP = "supergroup"
    CHANNEL = "channel"


class SyncState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ActionStatus(StrEnum):
    PROPOSED = "proposed"  # agent taklif qildi, user tasdiqlamagan
    CONFIRMED = "confirmed"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"
    BLOCKED = "blocked"  # guardrail to'sdi


def _now() -> Any:
    return func.now()


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    tg_user_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(64))
    locale: Mapped[str] = mapped_column(String(8), default="uz")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())

    # --- ui/R1b (migratsiya 0018) ---
    # UI sozlamalari — bitta JSONB. Nega alohida ustunlar emas: bular sof
    # **prezentatsiya** sozlamalari (tema, zichlik, default rejim, pin'lar),
    # ular ustida so'rov/JOIN qilinmaydi va UI o'sgani sayin yangi kalit
    # qo'shiladi — har biri uchun migratsiya yozish narxi qiymatidan katta.
    # Yagona yozish nuqtasi `services/prefs.py::merge` (faqat tanilgan,
    # validatsiyadan o'tgan kalitlar yoziladi — ixtiyoriy JSON emas).
    # `users.locale` ustuni o'z joyida qoladi (bot til tanlovi); web
    # `prefs["locale"]` to'ldirilgan bo'lsa **u** ustun turadi (pages.py).
    prefs: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb"), nullable=False
    )

    accounts: Mapped[list[Account]] = relationship(back_populates="user")


class Account(Base):
    """Ulangan Telegram akkaunt (MTProto session)."""

    __tablename__ = "accounts"
    __table_args__ = (UniqueConstraint("user_id", "tg_account_id", name="uq_account_per_user"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    tg_account_id: Mapped[int | None] = mapped_column(BigInteger, index=True)

    label: Mapped[str] = mapped_column(String(64), default="")
    phone_hash: Mapped[str | None] = mapped_column(String(64))  # sha256, raw raqam saqlanmaydi
    status: Mapped[str] = mapped_column(String(16), default=AccountStatus.PENDING)

    # Envelope encryption — app.crypto.envelope
    session_ciphertext: Mapped[bytes | None] = mapped_column(LargeBinary)
    session_wrapped_dek: Mapped[bytes | None] = mapped_column(LargeBinary)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # ── 3-bosqich/B (migratsiya 0010): shaxsiy briefing va jim soatlar ──
    # Sozlama **akkauntda**, chunki briefing mazmuni akkauntning kun
    # tugunlaridan (`retrieval_nodes`, level 2) va uning chatlaridan quriladi.
    # Yetkazish manzili esa `user.tg_user_id` — bot faqat egasiga yozadi.
    tz: Mapped[str] = mapped_column(
        String(64), default="Asia/Tashkent", server_default="Asia/Tashkent"
    )
    # Lokal (`tz` dagi) vaqt; yarim tun orqali o'ralishi mumkin (22:00 → 07:00).
    quiet_from: Mapped[time | None] = mapped_column(Time)
    quiet_to: Mapped[time | None] = mapped_column(Time)
    briefing_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false")
    )
    briefing_hour: Mapped[int] = mapped_column(SmallInteger, default=8, server_default="8")
    # null = barcha chatlar; aks holda `chats.id` ro'yxati
    briefing_chats: Mapped[list[int] | None] = mapped_column(JSONB)

    user: Mapped[User] = relationship(back_populates="accounts")
    chats: Mapped[list[Chat]] = relationship(back_populates="account")


class Chat(Base):
    __tablename__ = "chats"
    __table_args__ = (
        UniqueConstraint("account_id", "tg_peer_id", name="uq_chat_per_account"),
        # 0017 (ui/R1a) — profil yangilash navbati: eng eski `avatar_updated_at`
        Index("ix_chats_profile_stale", "account_id", "avatar_updated_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    tg_peer_id: Mapped[int] = mapped_column(BigInteger)
    access_hash: Mapped[int | None] = mapped_column(BigInteger)

    type: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(256), default="")
    username: Mapped[str | None] = mapped_column(String(64), index=True)

    is_writable: Mapped[bool] = mapped_column(Boolean, default=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)  # stats API uchun
    participants_count: Mapped[int | None] = mapped_column(Integer)

    # Agent uchun rejim — chat darajasida, default read_only (zero-risk)
    write_mode: Mapped[str] = mapped_column(String(24), default="read_only")

    sync_state: Mapped[str] = mapped_column(String(16), default=SyncState.IDLE)
    synced_min_id: Mapped[int | None] = mapped_column(BigInteger)  # eng eski olingan
    synced_max_id: Mapped[int | None] = mapped_column(BigInteger)  # eng yangi olingan
    synced_total: Mapped[int] = mapped_column(Integer, default=0)
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 2-bosqich: progress UI va xatolar uchun
    total_estimate: Mapped[int | None] = mapped_column(Integer)  # Telegram `count`
    sync_error: Mapped[str | None] = mapped_column(String(256))
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # 1-bosqich: sitata havolasi (`<prefix><tg_msg_id>`) sync vaqtida hisoblanadi —
    # runtime'da peer turini qayta tekshirishga hojat qolmaydi.
    link_prefix: Mapped[str | None] = mapped_column(String(128))

    # ── ui/R1a (migratsiya 0017): chat profili ──
    # `about` — `GetFull*` dan tavsif/bio; NULL = hali olinmagan (navbat mezoni).
    about: Mapped[str | None] = mapped_column(Text)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    is_scam: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    is_fake: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    # `GetDialogs` javobidan — qo'shimcha RPC yo'q (`ingestion.refresh_chats`).
    unread_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_message_text: Mapped[str | None] = mapped_column(String(200))
    # Profil rasmi — "media saqlanmaydi" invariantiga chegaralangan istisno
    # (migratsiya 0017 izohi). Fayl `DATA_DIR/avatars/<chat_id>.<ext>`,
    # DB'da faqat meta. `avatar_photo_id` — o'zgarishni aniqlash kaliti.
    avatar_photo_id: Mapped[int | None] = mapped_column(BigInteger)
    avatar_mime: Mapped[str | None] = mapped_column(String(32))
    avatar_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    avatar_error: Mapped[str | None] = mapped_column(String(200))

    account: Mapped[Account] = relationship(back_populates="chats")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("chat_id", "tg_msg_id", name="uq_msg_per_chat"),
        Index("ix_messages_chat_published", "chat_id", "published_at"),
        # 0008 — generated ustunlar ustidagi indekslar
        Index("ix_messages_tsv_uz", "tsv_uz", postgresql_using="gin"),
        Index("ix_messages_tsv_ru", "tsv_ru", postgresql_using="gin"),
        # 0015 — `chat_id` bilan birlashtirilgan trigram indeksi (btree_gin).
        # Yolg'iz `text_fold` ustidagi eski indeks (`ix_messages_fold_trgm`)
        # "bitta chat ichida trigram" so'rovida ishlatilmasdi; bu indeks
        # ikkala shartni bitta skanga qo'yadi va o'lchovda kichikroq chiqdi.
        Index(
            "ix_messages_chat_trgm",
            "chat_id",
            "text_fold",
            postgresql_using="gin",
            postgresql_ops={"text_fold": "gin_trgm_ops"},
        ),
        Index(
            "ix_messages_top_msg",
            "chat_id",
            "top_msg_id",
            postgresql_where=text("top_msg_id IS NOT NULL"),
        ),
        # 0015 — o'qish yo'llarining deyarli hammasi `deleted_at IS NULL` bilan.
        Index(
            "ix_messages_chat_msgid_alive",
            "chat_id",
            "tg_msg_id",
            postgresql_where=text("deleted_at IS NULL"),
        ),
        # 0015 — alert kursori (`alerts._scope_max_id`).
        Index(
            "ix_messages_chat_created_id",
            "chat_id",
            "created_at",
            postgresql_include=["id"],
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    tg_msg_id: Mapped[int] = mapped_column(BigInteger)

    sender_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    edited_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    text: Mapped[str] = mapped_column(Text, default="")
    # Media yuklanmaydi — faqat turi bilinsin ("photo"/"video"/... yoki None)
    media_type: Mapped[str | None] = mapped_column(String(24))

    reply_to_msg_id: Mapped[int | None] = mapped_column(BigInteger)
    fwd_from_id: Mapped[int | None] = mapped_column(BigInteger)
    grouped_id: Mapped[int | None] = mapped_column(BigInteger)  # albom
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False)

    # Oxirgi ko'rilgan qiymatlar (tez filtr uchun). Vaqt qatori — snapshot jadvalda.
    views: Mapped[int | None] = mapped_column(Integer)
    forwards: Mapped[int | None] = mapped_column(Integer)
    reactions_total: Mapped[int | None] = mapped_column(Integer)
    replies_count: Mapped[int | None] = mapped_column(Integer)

    raw: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    # ── 1-bosqich (migratsiya 0008) ──
    # Til — ingest'da `lingua` bilan aniqlanadi (uz | ru | en | other).
    lang: Mapped[str | None] = mapped_column(String(8))
    lang_conf: Mapped[float | None] = mapped_column(Float)
    # Forward manbasi: `fwd_from_id` (eski) faqat muallif; bular — manba chat/post.
    fwd_from_peer_id: Mapped[int | None] = mapped_column(BigInteger)
    fwd_from_msg_id: Mapped[int | None] = mapped_column(BigInteger)
    fwd_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Forum topic / thread ildizi (`reply_to.reply_to_top_id`).
    top_msg_id: Mapped[int | None] = mapped_column(BigInteger)
    # Soft-delete — 2-bosqich updates listener to'ldiradi.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # ── migratsiya 0014 ──
    # Xabarni **biz** yozganmiz (Telethon `m.out`). Shaxsiy chatda `from_id`
    # bo'sh keladi, shuning uchun "o'zimizniki emas" filtri `sender_id` ga
    # tayanib ishlamaydi — alertlar aynan shu ustunga qaraydi.
    # (`text()` bu yerda chaqirilmaydi — sinf tanasida `text` ustun nomi bilan
    #  soyalangan; satr sifatida berilgan server_default ham bir xil ishlaydi.)
    out: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    # Qator **bazaga** tushgan payt (`published_at` — Telegram vaqti).
    # Alert kursori faqat "cho'kkan" qatorlargacha siljiydi: READ COMMITTED
    # da kichikroq id keyinroq ko'rinishi mumkin.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())

    # ── generated stored ustunlar: DB hisoblaydi, ORM hech qachon yozmaydi ──
    # `deferred=True` — og'ir ustunlar oddiy SELECT'da tortilmasin.
    text_fold: Mapped[str | None] = mapped_column(
        Text, Computed("uz_fold(coalesce(text, ''))", persisted=True), deferred=True
    )
    tsv_uz: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('simple', uz_fold(coalesce(text, '')))", persisted=True),
        deferred=True,
        nullable=True,
    )
    tsv_ru: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('russian', coalesce(text, ''))", persisted=True),
        deferred=True,
        nullable=True,
    )


class MessageEmbedding(Base):
    """Xabar vektori — PK **`(message_id, model)`** (migratsiya 0016).

    Model ustuni kalitning bir qismi: `EMBED_MODEL` almashtirilganda yangi
    model qatorlari eskilari bilan **yonma-yon** yoziladi (`reembed_all` hech
    narsa o'chirmaydi), qidiruv esa `search._embed_model` bilan joriy modelni
    filtrlaydi. Bir ustunli PK'da bu mumkin emas edi — yozuvchilar
    `UniqueViolation` da qulardi.
    """

    __tablename__ = "message_embeddings"

    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(64), primary_key=True)
    vector: Mapped[Any] = mapped_column(Vector(get_settings().embed_dim))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class RetrievalNode(Base):
    """Ierarxik retrieval tuguni (2-bosqich, arxitektura v2, 3.3/5-bo'lim).

    Darajalar: **0 xabar (bu jadvalda saqlanmaydi)**, 1 epizod, 2 kun,
    3 hafta, 4 oy. Level 0 ataylab yo'q — xabarlar `messages` da, vektorlari
    `message_embeddings` da; dublikat million qator yaratilmaydi.

    `text` — `digests._line` formatidagi qatorlar; `ctx_prefix` — metadatadan
    bepul olingan kontekst sarlavhasi (chat, sana, so'zlovchilar, nimaga
    javob). Embed qilinadigan matn — `ctx_prefix + "\\n" + text`.

    `content_hash` (md5 hex) manba o'zgarganini sezadi: epizodda xabarlar
    ro'yxati, hafta tugunida esa bolalarning hash'lari ustidan md5.
    `simhash` — `textnorm.simhash64`, **belgili** bigint sifatida saqlanadi
    (`services/nodes.to_signed64`), chunki Postgres `bigint` belgili.
    """

    __tablename__ = "retrieval_nodes"
    __table_args__ = (
        UniqueConstraint("chat_id", "level", "starts_at", name="uq_node_chat_level_start"),
        Index("ix_nodes_account_level_time", "account_id", "level", text("starts_at DESC")),
        Index("ix_nodes_chat_level_time", "chat_id", "level", "starts_at"),
        Index("ix_nodes_tsv", "tsv", postgresql_using="gin"),
        Index("ix_nodes_simhash", "simhash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))
    level: Mapped[int] = mapped_column(SmallInteger)

    starts_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ends_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    msg_lo: Mapped[int] = mapped_column(BigInteger)  # tg_msg_id, inclusive
    msg_hi: Mapped[int] = mapped_column(BigInteger)  # tg_msg_id, inclusive
    msg_count: Mapped[int] = mapped_column(Integer, default=0)
    speakers: Mapped[list[int]] = mapped_column(
        ARRAY(BigInteger), default=list, server_default=text("'{}'::bigint[]")
    )

    text: Mapped[str] = mapped_column(Text, default="")
    ctx_prefix: Mapped[str] = mapped_column(Text, default="")
    content_hash: Mapped[str] = mapped_column(String(32))  # md5 hex
    simhash: Mapped[int] = mapped_column(BigInteger, default=0)  # belgili 64-bit
    structured: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=_now(), onupdate=_now()
    )

    # generated stored — DB hisoblaydi, ORM hech qachon yozmaydi
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('simple', uz_fold(coalesce(ctx_prefix, '') || ' ' || coalesce(text, '')))",
            persisted=True,
        ),
        deferred=True,
        nullable=True,
    )


class RetrievalNodeEmbedding(Base):
    """Tugun vektori — `halfvec(768)` (xotira ~2x, aniqlik farqi sezilmas).

    `model` ustuni bosqichma-bosqich re-embed uchun: yangi model qatorlari
    eskilari bilan yonma-yon turadi, qidiruv joriy modelni filtrlaydi.
    Shuning uchun u PK'ning bir qismi (migratsiya 0016) — `MessageEmbedding`
    bilan bir xil sabab.
    """

    __tablename__ = "retrieval_node_embeddings"
    __table_args__ = (Index("ix_node_emb_model", "model"),)

    node_id: Mapped[int] = mapped_column(
        ForeignKey("retrieval_nodes.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(64), primary_key=True)
    vector: Mapped[Any] = mapped_column(HALFVEC(get_settings().embed_dim))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class MessageMetricSnapshot(Base):
    """views/reactions vaqt qatori — delta hisoblash uchun yagona manba."""

    __tablename__ = "message_metric_snapshots"
    __table_args__ = (Index("ix_snap_msg_time", "message_id", "captured_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())

    views: Mapped[int | None] = mapped_column(Integer)
    forwards: Mapped[int | None] = mapped_column(Integer)
    replies_count: Mapped[int | None] = mapped_column(Integer)
    reactions_total: Mapped[int | None] = mapped_column(Integer)
    reactions: Mapped[dict[str, Any] | None] = mapped_column(JSONB)  # {"👍": 42, ...}


class ChatDailyRollup(Base):
    """Kunlik agregat — "joriy oy/yil" so'rovlarini millisekundda javoblash uchun."""

    __tablename__ = "chat_daily_rollups"
    __table_args__ = (UniqueConstraint("chat_id", "day", name="uq_rollup_chat_day"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    day: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    msg_count: Mapped[int] = mapped_column(Integer, default=0)
    views_sum: Mapped[int] = mapped_column(BigInteger, default=0)  # o'sha kuni chop etilganlar
    views_delta: Mapped[int] = mapped_column(BigInteger, default=0)  # o'sha kuni qo'shilgan
    reactions_sum: Mapped[int] = mapped_column(BigInteger, default=0)
    forwards_sum: Mapped[int] = mapped_column(BigInteger, default=0)
    participants_count: Mapped[int | None] = mapped_column(Integer)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    account_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id", ondelete="SET NULL"))
    chat_id: Mapped[int | None] = mapped_column(ForeignKey("chats.id", ondelete="SET NULL"))

    prompt: Mapped[str] = mapped_column(Text)
    model: Mapped[str] = mapped_column(String(64), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    # LLM uzilsa (masalan tool'siz provider) — run yarim bo'sh qolmaydi
    error: Mapped[str | None] = mapped_column(String(500))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class AgentAction(Base):
    """Append-only audit. Har bir tool chaqiruvi, ayniqsa write, shu yerda."""

    __tablename__ = "agent_actions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)

    tool: Mapped[str] = mapped_column(String(64))
    args: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(16), default=ActionStatus.PROPOSED)
    block_reason: Mapped[str | None] = mapped_column(String(128))

    target_peer_id: Mapped[int | None] = mapped_column(BigInteger)
    result_msg_id: Mapped[int | None] = mapped_column(BigInteger)
    error: Mapped[str | None] = mapped_column(Text)

    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class Conversation(Base):
    """Web UI AI chat suhbati (agent_runs — audit; bu — UI tarixi)."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    account_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id", ondelete="SET NULL"))
    title: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=_now(), onupdate=_now()
    )

    messages: Mapped[list[ConversationMessage]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.id",
    )


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text, default="")
    # javob qaysi model bilan, qancha token — UI'da ko'rsatish uchun
    model: Mapped[str] = mapped_column(String(64), default="")
    provider: Mapped[str] = mapped_column(String(32), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    # kontekst: qaysi chat, nechta xabar, strategiya — kontent emas (u qayta olinadi)
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())

    # ── sifat / xarajat (dashboard) ──
    task: Mapped[str] = mapped_column(String(16), default="")  # search | deep
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Float)  # taxminiy, pricing.py
    # foydalanuvchi bahosi: 1 = 👍, -1 = 👎
    rating: Mapped[int | None] = mapped_column(Integer)
    rating_comment: Mapped[str | None] = mapped_column(String(512))
    rated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # avtomatik baho (LLM-judge, arzon model): 1..5
    auto_relevance: Mapped[int | None] = mapped_column(Integer)
    auto_usefulness: Mapped[int | None] = mapped_column(Integer)
    auto_grounded: Mapped[bool | None] = mapped_column(Boolean)
    auto_note: Mapped[str | None] = mapped_column(String(256))

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class ChatDigest(Base):
    """Kunlik digest keshi — katta vaqt oynasi savollari uchun map-reduce takrorlanmasin.

    Bir marta arzon model bilan hisoblanadi (`services/digests.py`), keyin har savolda
    tayyor matn ishlatiladi. Xabarlar o'zgarsa qayta hisoblanadi — invalidatsiya
    `src_fingerprint` bo'yicha (`md5` ro'yxati): `msg_count` tahrir va o'chirishni
    sezmaydi, fingerprint ikkalasini ham qamrab oladi.
    """

    __tablename__ = "chat_digests"
    __table_args__ = (UniqueConstraint("chat_id", "day", name="uq_digest_chat_day"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"), index=True)
    day: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)  # UTC 00:00
    digest: Mapped[str] = mapped_column(Text, default="")
    msg_count: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str] = mapped_column(String(64), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())

    # ── 1-bosqich (migratsiya 0008) ──
    # md5 over "tg_msg_id:md5(text)" ro'yxati — tahrir/o'chirishni ham sezadi.
    src_fingerprint: Mapped[str | None] = mapped_column(String(32))
    level: Mapped[int] = mapped_column(SmallInteger, default=1, server_default="1")  # 1 kun
    period_start: Mapped[date | None] = mapped_column(Date)
    structured: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    faithfulness: Mapped[float | None] = mapped_column(Float)


class GeneratedImage(Base):
    """AI generatsiya qilgan rasm (7-bosqich). Fayl `DATA_DIR/images/<id>.png`, DB'da meta."""

    __tablename__ = "generated_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)  # uuid4
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    account_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id", ondelete="SET NULL"))
    prompt: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    mime: Mapped[str] = mapped_column(String(32), default="image/png")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class AutoReplyRule(Base):
    """Per-chat auto-reply qoidasi (7-bosqich).

    Worker har 5 daqiqada yangi kiruvchi xabarlarni tekshiradi, mos kelsa LLM bilan
    javob yozadi va **`send_message` taklifi** yaratadi (`agent_actions.proposed`).
    Chat `autonomous` bo'lsa darhol yuboriladi. Delete/edit hech qachon.
    """

    __tablename__ = "auto_reply_rules"
    __table_args__ = (UniqueConstraint("chat_id", name="uq_autoreply_chat"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), index=True
    )
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))
    enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    # trigger: all | mentions | keywords | questions
    trigger: Mapped[str] = mapped_column(String(16), default="questions")
    keywords: Mapped[str] = mapped_column(Text, default="")  # vergul bilan
    instructions: Mapped[str] = mapped_column(Text, default="")  # javob uslubi/mazmuni
    max_per_hour: Mapped[int] = mapped_column(Integer, default=5)
    quiet_from: Mapped[int | None] = mapped_column(Integer)  # soat (UTC), masalan 22
    quiet_to: Mapped[int | None] = mapped_column(Integer)  # soat (UTC), masalan 7
    last_processed_msg_id: Mapped[int] = mapped_column(BigInteger, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=_now(), onupdate=_now()
    )


class ScheduledJob(Base):
    __tablename__ = "scheduled_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    chat_id: Mapped[int | None] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))

    type: Mapped[str] = mapped_column(String(32))  # scheduled_post | auto_reply | digest
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    run_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # Telegram server tomonda rejalashtirilgan bo'lsa — bizning worker kerak emas
    tg_scheduled_msg_id: Mapped[int | None] = mapped_column(BigInteger)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class Briefing(Base):
    """Kunlik shaxsiy briefing (L3, 3-bosqich/B — arxitektura v2, 9-bo'lim).

    Bir akkauntning bir **lokal** kuni uchun bitta qator
    (`uq_briefing_account_day`) — cron idempotent bo'lishi shu kalitga
    tayanadi. `text` — foydalanuvchi ko'radigan markdown; `structured` —
    mexanik tekshiruvdan o'tgan bloklar (tldr, chatlar, qarorlar, ochiq
    savollar, kim faol bo'lgani).

    Yetkazish maydonlari (`delivered_at` / `delivery_error`) ataylab shu
    yerda: qayta yuborishning oldini olish uchun "qurilgan" va "yetkazilgan"
    bitta qatorda ko'rinib turishi kerak.
    """

    __tablename__ = "briefings"
    __table_args__ = (
        UniqueConstraint("account_id", "day", name="uq_briefing_account_day"),
        Index("ix_briefings_account_day", "account_id", "day"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    day: Mapped[date] = mapped_column(Date)

    text: Mapped[str] = mapped_column(Text, default="")
    structured: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    model: Mapped[str] = mapped_column(String(64), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_error: Mapped[str | None] = mapped_column(String(500))
    # Push urinishlari (0014) — `delivered_at` shartli UPDATE bilan "band
    # qilinadi", shuning uchun ikki cron tiki bitta briefingni ikki marta
    # yubora olmaydi; urinish soni esa qachon voz kechishni belgilaydi.
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")


# ─── 3-bosqich/C: alertlar (migratsiya 0011) ─────────────────────────────────


class AlertKind(StrEnum):
    KEYWORD = "keyword"  # vergul bilan ajratilgan so'zlar, fold fazosida
    SEMANTIC = "semantic"  # tabiiy tildagi tavsif → embedding → cosine


#: `alert_rules.kind` uchun native PG enum. Nomi migratsiya 0011 bilan bir xil;
#: `create_type=False` — turni migratsiya yaratadi, ORM emas.
ALERT_KIND_ENUM = SAEnum(AlertKind.KEYWORD.value, AlertKind.SEMANTIC.value, name="alert_kind")


class AlertRule(Base):
    """Kuzatuv qoidasi: yangi xabar mos kelsa — boshqaruv botiga push.

    `chat_id is None` → akkauntning **barcha** chatlari. Kursor
    `last_checked_msg_id` — `messages.id` (global monoton), `tg_msg_id` emas.
    """

    __tablename__ = "alert_rules"
    __table_args__ = (Index("ix_alert_rules_account_enabled", "account_id", "enabled"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    chat_id: Mapped[int | None] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))

    kind: Mapped[str] = mapped_column(ALERT_KIND_ENUM, default=AlertKind.KEYWORD)
    label: Mapped[str] = mapped_column(String(64), default="")
    pattern: Mapped[str] = mapped_column(Text, default="")
    # Semantik qoidaning keshlangan vektori — `pattern` o'zgarsa NULL ga tushadi.
    embedding: Mapped[Any] = mapped_column(HALFVEC(get_settings().embed_dim), nullable=True)
    threshold: Mapped[float] = mapped_column(Float, default=0.80, server_default="0.80")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    cooldown_minutes: Mapped[int] = mapped_column(Integer, default=30, server_default="30")
    last_checked_msg_id: Mapped[int] = mapped_column(BigInteger, default=0, server_default="0")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=_now(), onupdate=_now()
    )


class AlertEvent(Base):
    """Bitta mos kelgan xabar. `uq_alert_event` — dedup bazada, kodda emas."""

    __tablename__ = "alert_events"
    __table_args__ = (
        UniqueConstraint("rule_id", "message_id", name="uq_alert_event"),
        Index("ix_alert_events_rule_time", "rule_id", text("created_at DESC")),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    rule_id: Mapped[int] = mapped_column(ForeignKey("alert_rules.id", ondelete="CASCADE"))
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id", ondelete="CASCADE"))
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))
    tg_msg_id: Mapped[int] = mapped_column(BigInteger)
    score: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    delivery_error: Mapped[str | None] = mapped_column(String(500))
    # Push urinishlari (0014). `ALERT_MAX_DELIVERY_ATTEMPTS` dan keyin hodisa
    # `delivered_at` bilan yopiladi — "yetkazildi" emas, "voz kechildi".
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")


# ─── 4-bosqich/A: entity indeksi (migratsiya 0012) ───────────────────────────


class EntityKind(StrEnum):
    """`node_entities.kind` — matn sifatida saqlanadi (DB'da `CHECK`, native enum emas).

    `phone_like` **yo'q va bo'lmaydi**: telefon raqami indeksga ham,
    promptga ham tushmaydi (`services/entities.py` sarlavhasiga qarang).
    """

    USERNAME = "username"
    HASHTAG = "hashtag"
    CASHTAG = "cashtag"
    URL_DOMAIN = "url_domain"
    TME_LINK = "tme_link"
    MENTION_NAME = "mention_name"


class NodeEntity(Base):
    """Xabar ichidagi identifikator: `@username`, `#hashtag`, domen, `t.me` havolasi.

    `message_id` — manba xabar (soft-delete'da qator qoladi, qidiruv
    `messages.deleted_at IS NULL` bilan filtrlanadi). `node_id` — xabar
    qaysi epizodga tushgani; `episodes.build_episodes` bitta `UPDATE` bilan
    to'ldiradi, shuning uchun u **nullable** (entity xabar bilan birga,
    epizoddan oldin yoziladi).

    `uq_node_entity_value (message_id, kind, value)` — idempotentlik bazada:
    qayta sync `ON CONFLICT DO NOTHING` ga tushadi va `node_id` yo'qolmaydi.
    """

    __tablename__ = "node_entities"
    __table_args__ = (
        UniqueConstraint("message_id", "kind", "value", name="uq_node_entity_value"),
        Index("ix_node_entities_account_kind_value", "account_id", "kind", "value"),
        Index("ix_node_entities_chat_seen", "chat_id", text("seen_at DESC")),
        Index("ix_node_entities_node", "node_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))
    message_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("messages.id", ondelete="CASCADE")
    )
    node_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("retrieval_nodes.id", ondelete="SET NULL")
    )

    kind: Mapped[str] = mapped_column(String(16))
    value: Mapped[str] = mapped_column(Text)
    value_raw: Mapped[str] = mapped_column(Text, default="")
    seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ─── 4-bosqich/C: batch embedding ────────────────────────────────────────────


class EmbedBatchStatus(StrEnum):
    PENDING = "pending"  # submit qilindi, hali boshlanmadi
    RUNNING = "running"
    SUCCEEDED = "succeeded"  # vektorlar yozildi
    FAILED = "failed"  # qayta urinish mumkin (`attempts` oshadi)
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    FAILED_PERMANENT = "failed_permanent"  # `max_attempts` tugadi — boshqa yuborilmaydi


#: Elementlari hali "band" hisoblanadigan holatlar — yangi jobga tushmaydi.
EMBED_BATCH_INFLIGHT = (EmbedBatchStatus.PENDING, EmbedBatchStatus.RUNNING)


class EmbedBatch(Base):
    """Gemini Batch API'ga yuborilgan bitta embedding jobi.

    Nega jadval kerak: `batches.create_embeddings` darhol qaytadi, natija esa
    soatlar keyin, **boshqa jarayonda** olinadi. Job nomi va **tartiblangan**
    element id'lari xotirada saqlansa, worker qayta ishga tushishi bilan
    yo'qolardi va to'langan job hech qachon o'qilmay qolardi.

    `item_ids` — ataylab JSONB massiv, `ARRAY(BigInteger)` emas: bu qatorlar
    bo'ylab qidirilmaydi, faqat **tartib** muhim, va `@>` konteynment bilan
    "bu element allaqachon jobdami?" savoliga GIN indeks orqali javob beriladi
    (`ix_embed_batches_items`, `jsonb_path_ops`).

    `model` — API **qaytargan** nom (`pricing.py` qoidasi), so'ralgani emas.
    """

    __tablename__ = "embed_batches"
    __table_args__ = (
        Index("ix_embed_batches_status", "status", "kind"),
        Index(
            "ix_embed_batches_items",
            "item_ids",
            postgresql_using="gin",
            postgresql_ops={"item_ids": "jsonb_path_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), default="gemini")
    job_name: Mapped[str] = mapped_column(String(255), unique=True)
    kind: Mapped[str] = mapped_column(String(16))  # 'messages' | 'nodes'
    # `nodes` tugunlari bir nechta akkauntga tegishli bo'lishi mumkin → NULL
    account_id: Mapped[int | None] = mapped_column(ForeignKey("accounts.id", ondelete="SET NULL"))
    item_ids: Mapped[list[int]] = mapped_column(JSONB, default=list)
    model: Mapped[str] = mapped_column(String(64), default="")
    task_type: Mapped[str] = mapped_column(String(32), default="RETRIEVAL_DOCUMENT")
    status: Mapped[str] = mapped_column(String(24), default=EmbedBatchStatus.PENDING)
    attempts: Mapped[int] = mapped_column(Integer, default=1, server_default="1")

    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(String(500))
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")


# ─── jo'natuvchi nomlari (0019) ──────────────────────────────────────────────


class Sender(Base):
    """Bitta akkaunt ko'rgan Telegram foydalanuvchisining **ko'rinadigan** nomi.

    Nega alohida jadval: `messages.sender_id` xom raqam, tool natijasi esa
    modelga `user:124972562` bo'lib tushardi — model uni javobda shundayligicha
    takrorlardi va foydalanuvchi kimligini bilmasdi. Ism `GetHistory` javobidagi
    `m.sender` keshidan olinadi, ya'ni **qo'shimcha RPC yo'q** va allowlist
    buzilmaydi (`get_sender()` chaqirilmaydi).

    PK `(account_id, tg_user_id)`: bir xil odam ikki akkauntda boshqa nom bilan
    ko'rinishi mumkin (kontakt nomi), va bir akkauntning ko'rgani boshqasiga
    oqib o'tmasligi kerak.

    **Telefon raqami bu yerda yo'q va hech qachon bo'lmaydi** — jadval qiymati
    LLM promptiga tushadi, raqam esa u yerda kerak emas (`list_members` bilan
    bir mezon). Yangi ustun qo'shishdan oldin: "buni modelga berish shartmi?"
    """

    __tablename__ = "senders"
    __table_args__ = (
        Index("ix_senders_account_username", "account_id", "username"),
        Index("ix_senders_account_name", "account_id", "display_name"),
    )

    account_id: Mapped[int] = mapped_column(
        ForeignKey("accounts.id", ondelete="CASCADE"), primary_key=True
    )
    tg_user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    display_name: Mapped[str] = mapped_column(String(96), default="", server_default="")
    username: Mapped[str | None] = mapped_column(String(64))
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


# ─── korpus miqyosidagi skan (0020) ──────────────────────────────────────────


class RelevanceCache(Base):
    """Bitta savol x bitta xabar uchun filtr bahosi — skan keshi (0020).

    Nega kerak: korpus miqyosidagi kaskad `filter_batched` da har nomzodni
    Flash-Lite'ga ko'rsatadi. Foydalanuvchi bir savolni ikki marta so'rasa
    (yoki "chuqurroq qidir" bilan qaytsa) o'sha xabarlar yana modelga
    ketardi — natija o'zgarmaydi, token esa behuda ketadi (`seen_calls`
    bilan bir mantiq).

    Kalit `q_hash` = `md5(uz_fold(standalone_question))`: savolning imlo/
    apostrof/kirill farqi kesh promahiga aylanmasin. Unikal cheklov
    `(q_hash, chat_id, tg_msg_id)` — `chat_id` allaqachon akkauntga
    bog'langan, shuning uchun akkauntlararo oqish yo'q; `account_id` esa
    tozalash (`DELETE ... WHERE account_id`) va audit uchun.

    **Xabar matni bu yerda saqlanmaydi** — faqat baho. Kesh qatori
    modelga qaytib tushmaydi, u faqat "bu xabarni yana so'ramaymiz"
    degan belgidir.
    """

    __tablename__ = "relevance_cache"
    __table_args__ = (
        UniqueConstraint("q_hash", "chat_id", "tg_msg_id", name="uq_relcache_q_chat_msg"),
        Index("ix_relcache_account_created", "account_id", text("created_at DESC")),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    q_hash: Mapped[str] = mapped_column(String(32))  # md5 hex
    chat_id: Mapped[int] = mapped_column(ForeignKey("chats.id", ondelete="CASCADE"))
    tg_msg_id: Mapped[int] = mapped_column(BigInteger)
    score: Mapped[int] = mapped_column(SmallInteger, default=0)
    model: Mapped[str] = mapped_column(String(64), default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())


class ScanRun(Base):
    """Bitta korpus skanining audit qatori (0020) — halollik hisobi.

    `agent_runs` token/narxni yozadi, bu jadval esa **qamrovni**: nechta
    bucket rejalashtirilgan va nechtasi haqiqatan skanlangan, nechta
    nomzod filtrdan o'tgan, nega to'xtagan (`yield|ceiling|saturation|
    wall|complete`). Javob ostidagi qamrov hisoboti shu raqamlardan
    quriladi — "hammasini ko'rdim" degan gap chiqmasin.

    Jadval **faqat qo'shiladi** (`agent_actions` kabi trigger yo'q, lekin
    kod hech qachon UPDATE qilmaydi): har yugurish o'z qatorini yozadi.
    """

    __tablename__ = "scan_runs"
    __table_args__ = (Index("ix_scan_runs_account_created", "account_id", text("created_at DESC")),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id", ondelete="CASCADE"))
    # `Integer` ataylab aniq: `conversations.id` ham `Integer`, `Mapped[int | None]`
    # esa `Base` ning annotation map'i bo'yicha `BigInteger` ga tushib ketardi
    # (migratsiya bilan drift → `alembic check` qizarardi).
    conversation_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("conversations.id", ondelete="SET NULL")
    )
    mode: Mapped[str] = mapped_column(String(16), default="deep")
    q_hash: Mapped[str] = mapped_column(String(32), default="", server_default="")

    buckets_planned: Mapped[int] = mapped_column(Integer, default=0)
    buckets_scanned: Mapped[int] = mapped_column(Integer, default=0)
    candidates: Mapped[int] = mapped_column(Integer, default=0)
    filtered: Mapped[int] = mapped_column(Integer, default=0)
    kept: Mapped[int] = mapped_column(Integer, default=0)
    evidence: Mapped[int] = mapped_column(Integer, default=0)
    claims: Mapped[int] = mapped_column(Integer, default=0)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)

    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    est_cost_usd: Mapped[float] = mapped_column(Float, default=0.0, server_default="0")
    stopped_by: Mapped[str] = mapped_column(String(16), default="complete")
    # Bosqich sabablari alohida (0021): `filter` hovuzni tugatib "complete"
    # deganda ham `pull` qo'ygan "wall" auditda ko'rinib tursin.
    pull_stopped_by: Mapped[str] = mapped_column(String(16), default="", server_default="")
    filter_stopped_by: Mapped[str] = mapped_column(String(16), default="", server_default="")
    truncated: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    elapsed_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=_now())
