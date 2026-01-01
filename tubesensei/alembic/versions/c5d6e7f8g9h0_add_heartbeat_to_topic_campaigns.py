"""Add last_heartbeat_at to topic_campaigns

Revision ID: c5d6e7f8g9h0
Revises: b4c5d6e7f8g9
Create Date: 2026-01-01 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c5d6e7f8g9h0'
down_revision: Union[str, Sequence[str], None] = 'b4c5d6e7f8g9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add last_heartbeat_at column to topic_campaigns table."""
    op.add_column(
        'topic_campaigns',
        sa.Column(
            'last_heartbeat_at',
            sa.DateTime(timezone=True),
            nullable=True,
            comment='Last heartbeat from running worker task'
        )
    )


def downgrade() -> None:
    """Remove last_heartbeat_at column from topic_campaigns table."""
    op.drop_column('topic_campaigns', 'last_heartbeat_at')
