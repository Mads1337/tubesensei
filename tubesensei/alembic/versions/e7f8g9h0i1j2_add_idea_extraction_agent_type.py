"""Add idea_extraction value to agenttype enum

Revision ID: e7f8g9h0i1j2
Revises: d6e7f8g9h0i1
Create Date: 2026-02-23 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7f8g9h0i1j2'
down_revision: Union[str, Sequence[str], None] = 'd6e7f8g9h0i1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add 'IDEA_EXTRACTION' value to the agenttype enum (uppercase to match existing values)."""
    op.execute("ALTER TYPE agenttype ADD VALUE IF NOT EXISTS 'IDEA_EXTRACTION'")


def downgrade() -> None:
    """
    Note: PostgreSQL doesn't support removing enum values directly.
    Leaving as no-op since the value being present doesn't break anything.
    """
    pass
