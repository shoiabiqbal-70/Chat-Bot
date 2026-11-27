"""Add user accounts, sessions, and per-user document ownership.

Revision ID: 8b0d71a4c2f0
Revises: 390707ba537e
"""

from alembic import op
import sqlalchemy as sa


revision = "8b0d71a4c2f0"
down_revision = "390707ba537e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create account tables and scope future document hashes by owner."""
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(256), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_table(
        "user_sessions",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])
    op.add_column("documents", sa.Column("owner_id", sa.UUID(), nullable=True))
    op.create_foreign_key("fk_documents_owner_id_users", "documents", "users", ["owner_id"], ["id"], ondelete="CASCADE")
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])
    op.drop_constraint("uq_documents_sha256", "documents", type_="unique")
    # Earlier documents have a null owner; new accounts can upload equal files.
    op.create_unique_constraint("uq_documents_owner_sha256", "documents", ["owner_id", "sha256"])


def downgrade() -> None:
    """Remove accounts and restore the original global hash constraint."""
    op.drop_constraint("uq_documents_owner_sha256", "documents", type_="unique")
    op.create_unique_constraint("uq_documents_sha256", "documents", ["sha256"])
    op.drop_index("ix_documents_owner_id", table_name="documents")
    op.drop_constraint("fk_documents_owner_id_users", "documents", type_="foreignkey")
    op.drop_column("documents", "owner_id")
    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
    op.drop_table("user_sessions")
    op.drop_table("users")
