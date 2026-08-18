"""initial schema

Revision ID: 0001
Revises:
"""

from __future__ import annotations

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.config import get_settings

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBED_DIM = get_settings().embed_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("tg_user_id", sa.BigInteger, nullable=False, unique=True),
        sa.Column("username", sa.String(64)),
        sa.Column("locale", sa.String(8), nullable=False, server_default="uz"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_tg_user_id", "users", ["tg_user_id"])

    op.create_table(
        "accounts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("tg_account_id", sa.BigInteger),
        sa.Column("label", sa.String(64), nullable=False, server_default=""),
        sa.Column("phone_hash", sa.String(64)),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("session_ciphertext", sa.LargeBinary),
        sa.Column("session_wrapped_dek", sa.LargeBinary),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_seen_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("user_id", "tg_account_id", name="uq_account_per_user"),
    )
    op.create_index("ix_accounts_user_id", "accounts", ["user_id"])
    op.create_index("ix_accounts_tg_account_id", "accounts", ["tg_account_id"])

    op.create_table(
        "chats",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer,
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tg_peer_id", sa.BigInteger, nullable=False),
        sa.Column("access_hash", sa.BigInteger),
        sa.Column("type", sa.String(16), nullable=False),
        sa.Column("title", sa.String(256), nullable=False, server_default=""),
        sa.Column("username", sa.String(64)),
        sa.Column("is_writable", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("is_admin", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("participants_count", sa.Integer),
        sa.Column("write_mode", sa.String(24), nullable=False, server_default="read_only"),
        sa.Column("sync_state", sa.String(16), nullable=False, server_default="idle"),
        sa.Column("synced_min_id", sa.BigInteger),
        sa.Column("synced_max_id", sa.BigInteger),
        sa.Column("synced_total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_sync_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("account_id", "tg_peer_id", name="uq_chat_per_account"),
    )
    op.create_index("ix_chats_account_id", "chats", ["account_id"])
    op.create_index("ix_chats_username", "chats", ["username"])

    op.create_table(
        "messages",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "chat_id", sa.Integer, sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("tg_msg_id", sa.BigInteger, nullable=False),
        sa.Column("sender_id", sa.BigInteger),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("edited_at", sa.DateTime(timezone=True)),
        sa.Column("text", sa.Text, nullable=False, server_default=""),
        sa.Column("media_type", sa.String(24)),
        sa.Column("reply_to_msg_id", sa.BigInteger),
        sa.Column("fwd_from_id", sa.BigInteger),
        sa.Column("grouped_id", sa.BigInteger),
        sa.Column("is_pinned", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("views", sa.Integer),
        sa.Column("forwards", sa.Integer),
        sa.Column("reactions_total", sa.Integer),
        sa.Column("replies_count", sa.Integer),
        sa.Column("raw", postgresql.JSONB),
        sa.UniqueConstraint("chat_id", "tg_msg_id", name="uq_msg_per_chat"),
    )
    op.create_index("ix_messages_chat_id", "messages", ["chat_id"])
    op.create_index("ix_messages_sender_id", "messages", ["sender_id"])
    op.create_index("ix_messages_published_at", "messages", ["published_at"])
    op.create_index("ix_messages_chat_published", "messages", ["chat_id", "published_at"])

    # Hybrid search, 1-yarim: full-text.
    # 'simple' config ataylab — PG'da o'zbek tili lug'ati yo'q, stemming
    # o'rniga trigram bilan qoplaymiz.
    op.execute(
        "CREATE INDEX ix_messages_fts ON messages "
        "USING GIN (to_tsvector('simple', coalesce(text, '')))"
    )
    op.execute("CREATE INDEX ix_messages_trgm ON messages USING GIN (text gin_trgm_ops)")

    op.create_table(
        "message_embeddings",
        sa.Column(
            "message_id",
            sa.BigInteger,
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("vector", pgvector.sqlalchemy.Vector(EMBED_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # Hybrid search, 2-yarim: semantik. Cosine masofa.
    op.execute(
        "CREATE INDEX ix_embeddings_hnsw ON message_embeddings "
        "USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "message_metric_snapshots",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "message_id",
            sa.BigInteger,
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("views", sa.Integer),
        sa.Column("forwards", sa.Integer),
        sa.Column("replies_count", sa.Integer),
        sa.Column("reactions_total", sa.Integer),
        sa.Column("reactions", postgresql.JSONB),
    )
    op.create_index("ix_snap_msg_time", "message_metric_snapshots", ["message_id", "captured_at"])

    op.create_table(
        "chat_daily_rollups",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "chat_id", sa.Integer, sa.ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("day", sa.DateTime(timezone=True), nullable=False),
        sa.Column("msg_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("views_sum", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("views_delta", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("reactions_sum", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("forwards_sum", sa.BigInteger, nullable=False, server_default="0"),
        sa.Column("participants_count", sa.Integer),
        sa.UniqueConstraint("chat_id", "day", name="uq_rollup_chat_day"),
    )
    op.create_index("ix_chat_daily_rollups_chat_id", "chat_daily_rollups", ["chat_id"])
    op.create_index("ix_chat_daily_rollups_day", "chat_daily_rollups", ["day"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("account_id", sa.Integer, sa.ForeignKey("accounts.id", ondelete="SET NULL")),
        sa.Column("chat_id", sa.Integer, sa.ForeignKey("chats.id", ondelete="SET NULL")),
        sa.Column("prompt", sa.Text, nullable=False),
        sa.Column("model", sa.String(64), nullable=False, server_default=""),
        sa.Column("tokens_in", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tokens_out", sa.Integer, nullable=False, server_default="0"),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_runs_user_id", "agent_runs", ["user_id"])

    op.create_table(
        "agent_actions",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "run_id",
            sa.BigInteger,
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("tool", sa.String(64), nullable=False),
        sa.Column("args", postgresql.JSONB),
        sa.Column("status", sa.String(16), nullable=False, server_default="proposed"),
        sa.Column("block_reason", sa.String(128)),
        sa.Column("target_peer_id", sa.BigInteger),
        sa.Column("result_msg_id", sa.BigInteger),
        sa.Column("error", sa.Text),
        sa.Column("confirmed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_actions_run_id", "agent_actions", ["run_id"])

    # Audit append-only: UPDATE/DELETE ni DB darajasida to'sadi.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION agent_actions_append_only() RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'agent_actions append-only: DELETE taqiqlangan';
            END IF;
            -- status/confirmed_at/result_msg_id/error dan boshqasini o'zgartirib bo'lmaydi
            IF NEW.run_id IS DISTINCT FROM OLD.run_id
               OR NEW.tool IS DISTINCT FROM OLD.tool
               OR NEW.args IS DISTINCT FROM OLD.args
               OR NEW.target_peer_id IS DISTINCT FROM OLD.target_peer_id
               OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
                RAISE EXCEPTION
                    'agent_actions append-only: audit maydonlarini o''zgartirib bo''lmaydi';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        "CREATE TRIGGER trg_agent_actions_append_only "
        "BEFORE UPDATE OR DELETE ON agent_actions "
        "FOR EACH ROW EXECUTE FUNCTION agent_actions_append_only()"
    )

    op.create_table(
        "scheduled_jobs",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "account_id",
            sa.Integer,
            sa.ForeignKey("accounts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chat_id", sa.Integer, sa.ForeignKey("chats.id", ondelete="CASCADE")),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("payload", postgresql.JSONB),
        sa.Column("run_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("tg_scheduled_msg_id", sa.BigInteger),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_scheduled_jobs_run_at", "scheduled_jobs", ["run_at"])


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_agent_actions_append_only ON agent_actions")
    op.execute("DROP FUNCTION IF EXISTS agent_actions_append_only()")
    for table in (
        "scheduled_jobs",
        "agent_actions",
        "agent_runs",
        "chat_daily_rollups",
        "message_metric_snapshots",
        "message_embeddings",
        "messages",
        "chats",
        "accounts",
        "users",
    ):
        op.drop_table(table)
