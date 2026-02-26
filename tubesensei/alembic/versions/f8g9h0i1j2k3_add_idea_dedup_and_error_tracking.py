"""Add idea deduplication and extraction error tracking

Revision ID: f8g9h0i1j2k3
Revises: e7f8g9h0i1j2
Create Date: 2026-02-26 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f8g9h0i1j2k3'
down_revision: Union[str, Sequence[str], None] = 'e7f8g9h0i1j2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Add content_hash to ideas for deduplication
    idea_cols = [c['name'] for c in inspector.get_columns('ideas')]
    if 'content_hash' not in idea_cols:
        op.add_column('ideas', sa.Column('content_hash', sa.String(64), nullable=True))
    indexes = [idx['name'] for idx in inspector.get_indexes('ideas')]
    if 'idx_idea_content_hash' not in indexes:
        op.create_index('idx_idea_content_hash', 'ideas', ['content_hash'])

    # Add error tracking columns to campaign_videos
    cv_cols = [c['name'] for c in inspector.get_columns('campaign_videos')]
    if 'idea_extraction_retry_count' not in cv_cols:
        op.add_column('campaign_videos', sa.Column(
            'idea_extraction_retry_count', sa.Integer(), nullable=False, server_default='0'
        ))
    if 'idea_extraction_last_error' not in cv_cols:
        op.add_column('campaign_videos', sa.Column(
            'idea_extraction_last_error', sa.Text(), nullable=True
        ))


def downgrade() -> None:
    op.drop_column('campaign_videos', 'idea_extraction_last_error')
    op.drop_column('campaign_videos', 'idea_extraction_retry_count')
    op.drop_index('idx_idea_content_hash', table_name='ideas')
    op.drop_column('ideas', 'content_hash')
