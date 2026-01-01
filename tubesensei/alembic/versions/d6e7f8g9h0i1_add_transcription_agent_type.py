"""Add transcription value to agenttype enum

Revision ID: d6e7f8g9h0i1
Revises: c5d6e7f8g9h0
Create Date: 2026-01-01 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd6e7f8g9h0i1'
down_revision: Union[str, Sequence[str], None] = 'c5d6e7f8g9h0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add 'transcription' value to the agenttype enum."""
    op.execute("ALTER TYPE agenttype ADD VALUE IF NOT EXISTS 'transcription'")


def downgrade() -> None:
    """
    Note: PostgreSQL doesn't support removing enum values directly.
    To truly downgrade, you would need to:
    1. Create a new enum type without the value
    2. Migrate the column to the new type
    3. Drop the old enum type

    Since this is complex and 'transcription' being present doesn't break anything,
    we leave this as a no-op.
    """
    pass
