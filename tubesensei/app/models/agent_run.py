"""
AgentRun model - Tracks individual agent executions within a campaign.

Each discovery agent (search, channel expansion, topic filter, similar videos)
creates an AgentRun record when it executes, enabling detailed monitoring
and debugging of the discovery pipeline.
"""
from sqlalchemy import Column, String, Integer, DateTime, Enum as SQLEnum, Index, Float, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.orm import relationship
import enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

from app.models.base import BaseModel


class AgentType(enum.Enum):
    """Types of discovery agents."""
    COORDINATOR = "coordinator"           # Orchestrates the entire campaign
    SEARCH = "search"                     # YouTube search for initial videos
    CHANNEL_EXPANSION = "channel_expansion"  # Gets all videos from a channel
    TOPIC_FILTER = "topic_filter"         # AI-based relevance filtering
    SIMILAR_VIDEOS = "similar_videos"     # Related videos discovery


class AgentRunStatus(enum.Enum):
    """Status of an agent run."""
    PENDING = "pending"       # Queued but not started
    RUNNING = "running"       # Currently executing
    COMPLETED = "completed"   # Finished successfully
    FAILED = "failed"         # Failed with error
    CANCELLED = "cancelled"   # Cancelled by user/coordinator


class AgentRun(BaseModel):
    """
    Tracks execution of a single agent within a campaign.

    Provides detailed metrics and logging for each agent invocation,
    enabling monitoring, debugging, and performance analysis.
    """
    __tablename__ = "agent_runs"

    # Campaign reference
    campaign_id = Column(
        UUID(as_uuid=True),
        ForeignKey("topic_campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Agent identification
    agent_type = Column(
        SQLEnum(AgentType),
        nullable=False,
        index=True
    )

    status = Column(
        SQLEnum(AgentRunStatus),
        nullable=False,
        default=AgentRunStatus.PENDING,
        index=True
    )

    # Agent hierarchy (for sub-agents spawned by coordinator)
    parent_run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="SET NULL"),
        nullable=True,
        comment="Parent agent that spawned this run"
    )

    # Timing
    started_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    completed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )

    # Execution time in seconds
    execution_time_seconds = Column(
        Float,
        nullable=True
    )

    # Input/Output data
    input_data = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Input parameters provided to the agent"
    )

    output_data = Column(
        JSONB,
        nullable=True,
        comment="Results produced by the agent"
    )

    # Metrics
    items_processed = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of input items processed"
    )

    items_produced = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of output items produced"
    )

    # API usage
    api_calls_made = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of YouTube API calls made"
    )

    llm_calls_made = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of LLM API calls made"
    )

    # Cost tracking (for LLM agents)
    estimated_cost_usd = Column(
        Float,
        nullable=True,
        comment="Estimated cost in USD for LLM calls"
    )

    tokens_used = Column(
        Integer,
        nullable=True,
        comment="Total tokens used for LLM calls"
    )

    # Progress tracking (for long-running agents)
    progress_percent = Column(
        Float,
        nullable=False,
        default=0.0
    )

    current_item = Column(
        String(500),
        nullable=True,
        comment="Current item being processed (for progress display)"
    )

    # Error tracking
    error_message = Column(
        String,
        nullable=True
    )

    error_details = Column(
        JSONB,
        nullable=True,
        comment="Detailed error information including stack trace"
    )

    errors = Column(
        JSONB,
        nullable=False,
        default=list,
        comment="List of non-fatal errors encountered during execution"
    )

    # Retry information
    retry_count = Column(
        Integer,
        nullable=False,
        default=0
    )

    max_retries = Column(
        Integer,
        nullable=False,
        default=3
    )

    # Rate limit tracking
    rate_limited = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether the agent was rate limited during execution"
    )

    rate_limit_wait_seconds = Column(
        Float,
        nullable=True,
        comment="Total time spent waiting due to rate limits"
    )

    # Agent-specific metadata
    agent_metadata = Column(
        JSONB,
        nullable=False,
        default=dict,
        comment="Agent-specific configuration and state"
    )

    # Checkpoint for resumable agents
    checkpoint_data = Column(
        JSONB,
        nullable=True,
        comment="Checkpoint for resuming interrupted runs"
    )

    # Relationships
    campaign = relationship(
        "TopicCampaign",
        back_populates="agent_runs"
    )

    parent_run = relationship(
        "AgentRun",
        remote_side="AgentRun.id",
        backref="child_runs"
    )

    # Indexes
    __table_args__ = (
        Index("idx_ar_campaign_type", "campaign_id", "agent_type"),
        Index("idx_ar_campaign_status", "campaign_id", "status"),
        Index("idx_ar_started_at", "started_at"),
        Index("idx_ar_parent", "parent_run_id"),
    )

    def __repr__(self) -> str:
        return f"<AgentRun(id={self.id}, type={self.agent_type.value}, status={self.status.value})>"

    # Properties
    @property
    def is_active(self) -> bool:
        """Check if agent is currently running."""
        return self.status in [AgentRunStatus.PENDING, AgentRunStatus.RUNNING]

    @property
    def is_complete(self) -> bool:
        """Check if agent has finished (any terminal state)."""
        return self.status in [
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED
        ]

    @property
    def is_successful(self) -> bool:
        """Check if agent completed successfully."""
        return self.status == AgentRunStatus.COMPLETED

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration."""
        if self.started_at:
            end_time = self.completed_at or datetime.now(timezone.utc)
            return (end_time - self.started_at).total_seconds()
        return None

    @property
    def throughput_per_minute(self) -> Optional[float]:
        """Calculate items processed per minute."""
        duration = self.duration_seconds
        if duration and duration > 0 and self.items_processed > 0:
            return (self.items_processed / duration) * 60
        return None

    @property
    def success_rate(self) -> float:
        """Calculate success rate based on items processed vs produced."""
        if self.items_processed > 0:
            return (self.items_produced / self.items_processed) * 100
        return 0.0

    @property
    def has_errors(self) -> bool:
        """Check if there were any errors."""
        return bool(self.error_message) or (self.errors and len(self.errors) > 0)

    @property
    def can_retry(self) -> bool:
        """Check if agent can be retried."""
        return self.status == AgentRunStatus.FAILED and self.retry_count < self.max_retries

    # State transitions
    def start(self) -> None:
        """Mark agent as running."""
        self.status = AgentRunStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def complete(self, output_data: Optional[Dict[str, Any]] = None) -> None:
        """Mark agent as completed successfully."""
        self.status = AgentRunStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.progress_percent = 100.0
        if output_data:
            self.output_data = output_data
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def fail(self, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        """Mark agent as failed."""
        self.status = AgentRunStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error_message
        if error_details:
            self.error_details = error_details
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    def cancel(self) -> None:
        """Mark agent as cancelled."""
        self.status = AgentRunStatus.CANCELLED
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()

    # Progress updates
    def update_progress(self, percent: float, current_item: Optional[str] = None) -> None:
        """Update progress percentage and current item."""
        self.progress_percent = min(percent, 100.0)
        if current_item:
            self.current_item = current_item

    def increment_processed(self, count: int = 1) -> None:
        """Increment items processed count."""
        self.items_processed += count

    def increment_produced(self, count: int = 1) -> None:
        """Increment items produced count."""
        self.items_produced += count

    def increment_api_calls(self, count: int = 1) -> None:
        """Increment API calls count."""
        self.api_calls_made += count

    def increment_llm_calls(self, count: int = 1, tokens: int = 0, cost: float = 0.0) -> None:
        """Increment LLM calls count and track costs."""
        self.llm_calls_made += count
        if tokens:
            self.tokens_used = (self.tokens_used or 0) + tokens
        if cost:
            self.estimated_cost_usd = (self.estimated_cost_usd or 0.0) + cost

    def record_rate_limit(self, wait_seconds: float) -> None:
        """Record rate limit wait time."""
        self.rate_limited = True
        self.rate_limit_wait_seconds = (self.rate_limit_wait_seconds or 0.0) + wait_seconds

    def add_error(self, error: str) -> None:
        """Add a non-fatal error to the errors list."""
        if not self.errors:
            self.errors = []
        self.errors.append({
            "message": error,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

    def increment_retry(self) -> None:
        """Increment retry count."""
        self.retry_count += 1

    # Checkpoint management
    def save_checkpoint(self, data: Dict[str, Any]) -> None:
        """Save checkpoint data for resume capability."""
        self.checkpoint_data = data

    def clear_checkpoint(self) -> None:
        """Clear checkpoint data."""
        self.checkpoint_data = None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary for API responses."""
        return {
            "id": str(self.id),
            "campaign_id": str(self.campaign_id),
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "progress_percent": self.progress_percent,
            "current_item": self.current_item,
            "items_processed": self.items_processed,
            "items_produced": self.items_produced,
            "api_calls": self.api_calls_made,
            "llm_calls": self.llm_calls_made,
            "tokens_used": self.tokens_used,
            "estimated_cost_usd": self.estimated_cost_usd,
            "success_rate": self.success_rate,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "has_errors": self.has_errors,
            "rate_limited": self.rate_limited,
        }

    @classmethod
    def create_for_agent(
        cls,
        campaign_id,
        agent_type: AgentType,
        input_data: Dict[str, Any],
        parent_run_id = None
    ) -> "AgentRun":
        """Factory method to create a new agent run."""
        return cls(
            campaign_id=campaign_id,
            agent_type=agent_type,
            input_data=input_data,
            parent_run_id=parent_run_id,
            status=AgentRunStatus.PENDING
        )
