"""Add investigation_agents table

Revision ID: g9h0i1j2k3l4
Revises: f8g9h0i1j2k3
Create Date: 2026-02-26 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'g9h0i1j2k3l4'
down_revision: Union[str, Sequence[str], None] = 'f8g9h0i1j2k3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - add investigation_agents table."""
    op.create_table(
        'investigation_agents',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('user_prompt_template', sa.Text(), nullable=False),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.PrimaryKeyConstraint('id')
    )

    op.create_index('idx_investigation_agent_is_active', 'investigation_agents', ['is_active'], unique=False)


def downgrade() -> None:
    """Downgrade schema - remove investigation_agents table."""
    op.drop_index('idx_investigation_agent_is_active', table_name='investigation_agents')
    op.drop_table('investigation_agents')
