"""Add celery_task_id to topic_campaigns

Revision ID: b4c5d6e7f8g9
Revises: a3b4c5d6e7f8
Create Date: 2025-12-31 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b4c5d6e7f8g9'
down_revision: Union[str, Sequence[str], None] = 'a3b4c5d6e7f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add celery_task_id column to topic_campaigns table."""
    op.add_column(
        'topic_campaigns',
        sa.Column(
            'celery_task_id',
            sa.String(length=255),
            nullable=True,
            comment='ID of the currently running Celery task'
        )
    )
    op.create_index(
        'idx_campaign_celery_task_id',
        'topic_campaigns',
        ['celery_task_id'],
        unique=False
    )


def downgrade() -> None:
    """Remove celery_task_id column from topic_campaigns table."""
    op.drop_index('idx_campaign_celery_task_id', table_name='topic_campaigns')
    op.drop_column('topic_campaigns', 'celery_task_id')
