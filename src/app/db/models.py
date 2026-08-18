"""ORM modellar.

Dizayn qarorlari:
  * Media saqlanmaydi — faqat `media_type` belgisi (talab bo'yicha to'liq ignore).
  * `MessageMetricSnapshot` — views/reactions vaqt qatori. Telegram tarixiy
    qiymat bermaydi, shuning uchun 1-kundan snapshot yig'ish shart, aks holda
    "bu hafta qancha ko'rish qo'shildi" savoliga hech qachon javob bo'lmaydi.
  * `AgentAction` — immutable audit log. Har bir MTProto write shu yerga tushadi.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
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

    user: Mapped[User] = relationship(back_populates="accounts")
    chats: Mapped[list[Chat]] = relationship(back_populates="account")


class Chat(Base):
    __tablename__ = "chats"
    __table_args__ = (UniqueConstraint("account_id", "tg_peer_id", name="uq_chat_per_account"),)

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

    account: Mapped[Account] = relationship(back_populates="chats")


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("chat_id", "tg_msg_id", name="uq_msg_per_chat"),
        Index("ix_messages_chat_published", "chat_id", "published_at"),
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


class MessageEmbedding(Base):
    __tablename__ = "message_embeddings"

    message_id: Mapped[int] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    model: Mapped[str] = mapped_column(String(64))
    vector: Mapped[Any] = mapped_column(Vector(get_settings().embed_dim))
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
    tayyor matn ishlatiladi. Xabarlar o'zgarsa (yangi sync) — `msg_count` farq qilsa qayta.
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
