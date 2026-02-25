"""Add investigation_runs table

Revision ID: h0i1j2k3l4m5
Revises: g9h0i1j2k3l4
Create Date: 2026-02-26 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'h0i1j2k3l4m5'
down_revision: Union[str, Sequence[str], None] = 'g9h0i1j2k3l4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create investigationrunstatus enum
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE investigationrunstatus AS ENUM ('pending', 'running', 'completed', 'failed');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create investigation_runs table
    op.create_table(
        'investigation_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('agent_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('idea_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            'status',
            sa.Enum('pending', 'running', 'completed', 'failed', name='investigationrunstatus'),
            nullable=False,
            server_default='pending',
        ),
        sa.Column('result', sa.Text(), nullable=True),
        sa.Column('result_structured', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['agent_id'], ['investigation_agents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['idea_id'], ['ideas.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes
    op.create_index('ix_investigation_runs_agent_id', 'investigation_runs', ['agent_id'], unique=False)
    op.create_index('ix_investigation_runs_idea_id', 'investigation_runs', ['idea_id'], unique=False)
    op.create_index('ix_investigation_runs_status', 'investigation_runs', ['status'], unique=False)
    op.create_index('idx_investigation_run_agent_idea', 'investigation_runs', ['agent_id', 'idea_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_investigation_run_agent_idea', table_name='investigation_runs')
    op.drop_index('ix_investigation_runs_status', table_name='investigation_runs')
    op.drop_index('ix_investigation_runs_idea_id', table_name='investigation_runs')
    op.drop_index('ix_investigation_runs_agent_id', table_name='investigation_runs')
    op.drop_table('investigation_runs')
    sa.Enum(name='investigationrunstatus').drop(op.get_bind(), checkfirst=True)
