"""Add topic campaigns for topic-based video discovery

Revision ID: a3b4c5d6e7f8
Revises: 107993acac09
Create Date: 2025-12-31 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a3b4c5d6e7f8'
down_revision: Union[str, Sequence[str], None] = '107993acac09'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - add topic campaign tables."""

    # Create campaignstatus enum
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE campaignstatus AS ENUM ('draft', 'running', 'paused', 'completed', 'failed', 'cancelled');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create discoverysource enum
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE discoverysource AS ENUM ('search', 'channel_expansion', 'similar_videos');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create agenttype enum
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE agenttype AS ENUM ('coordinator', 'search', 'channel_expansion', 'topic_filter', 'similar_videos');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create agentrunstatus enum
    op.execute("""
        DO $$ BEGIN
            CREATE TYPE agentrunstatus AS ENUM ('pending', 'running', 'completed', 'failed', 'cancelled');
        EXCEPTION
            WHEN duplicate_object THEN null;
        END $$;
    """)

    # Create topic_campaigns table
    op.create_table('topic_campaigns',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('topic', sa.String(length=1000), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', postgresql.ENUM('draft', 'running', 'paused', 'completed', 'failed', 'cancelled', name='campaignstatus', create_type=False), nullable=False),
        sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('total_videos_discovered', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_videos_relevant', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_videos_filtered', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_channels_explored', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_transcripts_extracted', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('paused_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('estimated_completion_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('progress_percent', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('api_calls_made', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('llm_calls_made', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('checkpoint_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('last_checkpoint_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('execution_time_seconds', sa.Float(), nullable=True),
        sa.Column('campaign_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('statistics', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for topic_campaigns
    op.create_index('idx_campaign_status', 'topic_campaigns', ['status'], unique=False)
    op.create_index('idx_campaign_topic', 'topic_campaigns', ['topic'], unique=False)
    op.create_index('idx_campaign_started', 'topic_campaigns', ['started_at'], unique=False)
    op.create_index('idx_campaign_status_created', 'topic_campaigns', ['status', 'created_at'], unique=False)
    op.create_index(None, 'topic_campaigns', ['id'], unique=False)

    # Create agent_runs table (before campaign_videos since it references agent_runs)
    op.create_table('agent_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('agent_type', postgresql.ENUM('coordinator', 'search', 'channel_expansion', 'topic_filter', 'similar_videos', name='agenttype', create_type=False), nullable=False),
        sa.Column('status', postgresql.ENUM('pending', 'running', 'completed', 'failed', 'cancelled', name='agentrunstatus', create_type=False), nullable=False),
        sa.Column('parent_run_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('execution_time_seconds', sa.Float(), nullable=True),
        sa.Column('input_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('output_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('items_processed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('items_produced', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('api_calls_made', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('llm_calls_made', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('estimated_cost_usd', sa.Float(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('progress_percent', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('current_item', sa.String(length=500), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('errors', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_retries', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('rate_limited', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('rate_limit_wait_seconds', sa.Float(), nullable=True),
        sa.Column('agent_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('checkpoint_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['campaign_id'], ['topic_campaigns.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['parent_run_id'], ['agent_runs.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for agent_runs
    op.create_index('idx_ar_campaign_type', 'agent_runs', ['campaign_id', 'agent_type'], unique=False)
    op.create_index('idx_ar_campaign_status', 'agent_runs', ['campaign_id', 'status'], unique=False)
    op.create_index('idx_ar_started_at', 'agent_runs', ['started_at'], unique=False)
    op.create_index('idx_ar_parent', 'agent_runs', ['parent_run_id'], unique=False)
    op.create_index(None, 'agent_runs', ['id'], unique=False)
    op.create_index(None, 'agent_runs', ['campaign_id'], unique=False)
    op.create_index(None, 'agent_runs', ['agent_type'], unique=False)
    op.create_index(None, 'agent_runs', ['status'], unique=False)

    # Create campaign_videos table
    op.create_table('campaign_videos',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('discovery_source', postgresql.ENUM('search', 'channel_expansion', 'similar_videos', name='discoverysource', create_type=False), nullable=False),
        sa.Column('discovered_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('agent_run_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('source_video_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('source_channel_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('is_topic_relevant', sa.Boolean(), nullable=True),
        sa.Column('relevance_score', sa.Float(), nullable=True),
        sa.Column('filter_reasoning', sa.Text(), nullable=True),
        sa.Column('matched_keywords', postgresql.JSONB(astext_type=sa.Text()), nullable=True, server_default='[]'),
        sa.Column('topic_alignment', sa.String(length=50), nullable=True),
        sa.Column('filtered_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('transcript_extracted', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('transcript_extracted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('ideas_extracted', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('ideas_extracted_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('discovery_depth', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('discovery_order', sa.Integer(), nullable=True),
        sa.Column('discovery_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.ForeignKeyConstraint(['campaign_id'], ['topic_campaigns.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['agent_run_id'], ['agent_runs.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['source_video_id'], ['videos.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['source_channel_id'], ['channels.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('campaign_id', 'video_id', name='uq_campaign_video')
    )

    # Create indexes for campaign_videos
    op.create_index('idx_cv_campaign_relevant', 'campaign_videos', ['campaign_id', 'is_topic_relevant'], unique=False)
    op.create_index('idx_cv_campaign_source', 'campaign_videos', ['campaign_id', 'discovery_source'], unique=False)
    op.create_index('idx_cv_relevance_score', 'campaign_videos', ['relevance_score'], unique=False)
    op.create_index('idx_cv_discovered_at', 'campaign_videos', ['discovered_at'], unique=False)
    op.create_index(None, 'campaign_videos', ['id'], unique=False)
    op.create_index(None, 'campaign_videos', ['campaign_id'], unique=False)
    op.create_index(None, 'campaign_videos', ['video_id'], unique=False)
    op.create_index(None, 'campaign_videos', ['is_topic_relevant'], unique=False)
    op.create_index(None, 'campaign_videos', ['discovery_source'], unique=False)

    # Create campaign_channels table
    op.create_table('campaign_channels',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('campaign_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('channel_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('discovery_source', postgresql.ENUM('search', 'channel_expansion', 'similar_videos', name='discoverysource', create_type=False), nullable=False),
        sa.Column('discovered_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('now()')),
        sa.Column('source_video_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('was_expanded', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('expanded_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('videos_discovered', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('videos_relevant', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('videos_filtered_out', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('videos_pending_filter', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('videos_limit', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('limit_reached', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('expansion_priority', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('channel_subscriber_count', sa.Integer(), nullable=True),
        sa.Column('channel_video_count', sa.Integer(), nullable=True),
        sa.Column('expansion_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('expansion_error', sa.Text(), nullable=True),
        sa.Column('expansion_retries', sa.Integer(), nullable=False, server_default='0'),
        sa.ForeignKeyConstraint(['campaign_id'], ['topic_campaigns.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['source_video_id'], ['videos.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('campaign_id', 'channel_id', name='uq_campaign_channel')
    )

    # Create indexes for campaign_channels
    op.create_index('idx_cc_campaign_expanded', 'campaign_channels', ['campaign_id', 'was_expanded'], unique=False)
    op.create_index('idx_cc_campaign_priority', 'campaign_channels', ['campaign_id', 'expansion_priority'], unique=False)
    op.create_index('idx_cc_discovered_at', 'campaign_channels', ['discovered_at'], unique=False)
    op.create_index(None, 'campaign_channels', ['id'], unique=False)
    op.create_index(None, 'campaign_channels', ['campaign_id'], unique=False)
    op.create_index(None, 'campaign_channels', ['channel_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema - remove topic campaign tables."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()

    # Drop tables in reverse order of creation (respecting foreign keys)
    if 'campaign_channels' in tables:
        op.drop_index('idx_cc_discovered_at', table_name='campaign_channels')
        op.drop_index('idx_cc_campaign_priority', table_name='campaign_channels')
        op.drop_index('idx_cc_campaign_expanded', table_name='campaign_channels')
        op.drop_table('campaign_channels')

    if 'campaign_videos' in tables:
        op.drop_index('idx_cv_discovered_at', table_name='campaign_videos')
        op.drop_index('idx_cv_relevance_score', table_name='campaign_videos')
        op.drop_index('idx_cv_campaign_source', table_name='campaign_videos')
        op.drop_index('idx_cv_campaign_relevant', table_name='campaign_videos')
        op.drop_table('campaign_videos')

    if 'agent_runs' in tables:
        op.drop_index('idx_ar_parent', table_name='agent_runs')
        op.drop_index('idx_ar_started_at', table_name='agent_runs')
        op.drop_index('idx_ar_campaign_status', table_name='agent_runs')
        op.drop_index('idx_ar_campaign_type', table_name='agent_runs')
        op.drop_table('agent_runs')

    if 'topic_campaigns' in tables:
        op.drop_index('idx_campaign_status_created', table_name='topic_campaigns')
        op.drop_index('idx_campaign_started', table_name='topic_campaigns')
        op.drop_index('idx_campaign_topic', table_name='topic_campaigns')
        op.drop_index('idx_campaign_status', table_name='topic_campaigns')
        op.drop_table('topic_campaigns')

    # Drop enums with checkfirst
    sa.Enum(name='agentrunstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='agenttype').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='discoverysource').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='campaignstatus').drop(op.get_bind(), checkfirst=True)
