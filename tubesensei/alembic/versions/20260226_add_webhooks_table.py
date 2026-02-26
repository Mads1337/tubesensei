"""Add webhooks table

Revision ID: add_webhooks_table
Revises: h0i1j2k3l4m5
Create Date: 2026-02-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_webhooks_table'
down_revision: Union[str, Sequence[str], None] = 'h0i1j2k3l4m5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if 'webhooks' not in inspector.get_table_names():
        op.create_table(
            'webhooks',
            sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('gen_random_uuid()')),
            sa.Column('name', sa.String(255), nullable=False),
            sa.Column('url', sa.String(500), nullable=False),
            sa.Column('secret', sa.String(255), nullable=True),
            sa.Column('events', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
            sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
            sa.Column('last_triggered_at', sa.DateTime(timezone=True), nullable=True),
            sa.Column('failure_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
            sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
            sa.PrimaryKeyConstraint('id'),
        )

    indexes = [idx['name'] for idx in inspector.get_indexes('webhooks')] if 'webhooks' in inspector.get_table_names() else []
    if 'idx_webhook_active_events' not in indexes:
        op.create_index('idx_webhook_active_events', 'webhooks', ['is_active'])


def downgrade() -> None:
    op.drop_index('idx_webhook_active_events', table_name='webhooks')
    op.drop_table('webhooks')
