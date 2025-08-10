"""Add Idea model and update Channel fields

Revision ID: 107993acac09
Revises: f91834886dd0
Create Date: 2025-08-10 22:47:46.030770

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '107993acac09'
down_revision: Union[str, Sequence[str], None] = 'f91834886dd0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create idea_status enum using raw SQL with check
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE ideastatus AS ENUM ('extracted', 'reviewed', 'selected', 'rejected', 'in_progress', 'implemented');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    
    # Create idea_priority enum using raw SQL with check
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE ideapriority AS ENUM ('low', 'medium', 'high', 'critical');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)
    
    # Create ideas table
    op.create_table('ideas',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('status', postgresql.ENUM('extracted', 'reviewed', 'selected', 'rejected', 'in_progress', 'implemented', name='ideastatus', create_type=False), nullable=False),
        sa.Column('priority', postgresql.ENUM('low', 'medium', 'high', 'critical', name='ideapriority', create_type=False), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('complexity_score', sa.Integer(), nullable=True),
        sa.Column('market_size_estimate', sa.String(length=50), nullable=True),
        sa.Column('target_audience', sa.String(length=200), nullable=True),
        sa.Column('implementation_time_estimate', sa.String(length=50), nullable=True),
        sa.Column('source_timestamp', sa.Integer(), nullable=True),
        sa.Column('source_context', sa.Text(), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('technologies', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('competitive_advantage', sa.Text(), nullable=True),
        sa.Column('potential_challenges', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('monetization_strategies', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('related_ideas', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column('extraction_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('review_notes', sa.Text(), nullable=True),
        sa.Column('reviewed_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('selected_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('selected_by', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('export_count', sa.Integer(), nullable=False),
        sa.Column('last_exported_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['reviewed_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['selected_by'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for ideas table
    op.create_index('idx_idea_category_status', 'ideas', ['category', 'status'], unique=False)
    op.create_index('idx_idea_priority_status', 'ideas', ['priority', 'status'], unique=False)
    op.create_index('idx_idea_reviewed', 'ideas', ['reviewed_at', 'reviewed_by'], unique=False)
    op.create_index('idx_idea_status_confidence', 'ideas', ['status', 'confidence_score'], unique=False)
    op.create_index('idx_idea_video_status', 'ideas', ['video_id', 'status'], unique=False)
    op.create_index(None, 'ideas', ['category'], unique=False)
    op.create_index(None, 'ideas', ['id'], unique=False)
    op.create_index(None, 'ideas', ['priority'], unique=False)
    op.create_index(None, 'ideas', ['status'], unique=False)
    op.create_index(None, 'ideas', ['video_id'], unique=False)
    
    # Check if channel_name column exists and rename it to name
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('channels')]
    
    if 'channel_name' in columns and 'name' not in columns:
        # Rename channel_name to name
        op.alter_column('channels', 'channel_name', new_column_name='name')
    
    # Add channel_url if it doesn't exist
    if 'channel_url' not in columns:
        op.add_column('channels', sa.Column('channel_url', sa.String(length=500), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # Check if columns exist before modifying
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('channels')]
    
    # Rename name back to channel_name if needed
    if 'name' in columns and 'channel_name' not in columns:
        op.alter_column('channels', 'name', new_column_name='channel_name')
    
    # Drop channel_url if it exists
    if 'channel_url' in columns:
        op.drop_column('channels', 'channel_url')
    
    # Check if ideas table exists before dropping
    tables = inspector.get_table_names()
    if 'ideas' in tables:
        # Drop ideas table and its indexes
        op.drop_index('idx_idea_video_status', table_name='ideas')
        op.drop_index('idx_idea_status_confidence', table_name='ideas')
        op.drop_index('idx_idea_reviewed', table_name='ideas')
        op.drop_index('idx_idea_priority_status', table_name='ideas')
        op.drop_index('idx_idea_category_status', table_name='ideas')
        op.drop_table('ideas')
    
    # Drop enums with checkfirst
    sa.Enum(name='ideapriority').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='ideastatus').drop(op.get_bind(), checkfirst=True)
