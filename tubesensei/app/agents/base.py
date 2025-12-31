"""
Base Agent Framework

Provides abstract base class and common functionality for all discovery agents.
Each agent is a modular, reusable component that can run independently or
as part of a coordinated discovery pipeline.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Callable, TYPE_CHECKING
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.rate_limiter import RateLimiter
from app.models.agent_run import AgentRun, AgentType, AgentRunStatus

if TYPE_CHECKING:
    from app.models.topic_campaign import TopicCampaign

logger = logging.getLogger(__name__)


class AgentEventType(Enum):
    """Types of events agents can emit."""
    STARTED = "started"
    PROGRESS = "progress"
    ITEM_DISCOVERED = "item_discovered"
    ITEM_PROCESSED = "item_processed"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class AgentEvent:
    """Event emitted by an agent for progress tracking."""
    event_type: AgentEventType
    agent_type: AgentType
    campaign_id: UUID
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None


@dataclass
class AgentResult:
    """Result returned by an agent after execution."""
    success: bool
    items_processed: int = 0
    items_produced: int = 0
    api_calls_made: int = 0
    llm_calls_made: int = 0
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


@dataclass
class AgentContext:
    """
    Shared context for agent execution.

    Contains configuration, database session, rate limiters,
    and callbacks for inter-agent communication.
    """
    campaign_id: UUID
    campaign: "TopicCampaign"
    db: AsyncSession
    config: Dict[str, Any]

    # Rate limiters
    youtube_rate_limiter: RateLimiter = field(default_factory=lambda: RateLimiter(requests_per_minute=120))
    llm_rate_limiter: RateLimiter = field(default_factory=lambda: RateLimiter(requests_per_minute=60))

    # Event callbacks
    event_callback: Optional[Callable[[AgentEvent], None]] = None

    # Cancellation token
    cancelled: bool = False

    # Parent agent run (for sub-agents)
    parent_run_id: Optional[UUID] = None

    # Shared state between agents
    shared_state: Dict[str, Any] = field(default_factory=dict)

    def emit_event(self, event: AgentEvent) -> None:
        """Emit an event to registered callbacks."""
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def is_cancelled(self) -> bool:
        """Check if the agent should stop."""
        return self.cancelled

    def cancel(self) -> None:
        """Request cancellation of all agents."""
        self.cancelled = True


class BaseAgent(ABC):
    """
    Abstract base class for all discovery agents.

    Provides common functionality for:
    - Rate limiting
    - Progress tracking
    - Error handling
    - Database persistence of agent runs
    - Event emission for real-time updates
    """

    agent_type: AgentType = AgentType.COORDINATOR  # Override in subclasses

    def __init__(self, context: AgentContext):
        """
        Initialize the agent.

        Args:
            context: Shared execution context with config, DB, and rate limiters
        """
        self.context = context
        self.campaign_id = context.campaign_id
        self.config = context.config
        self.db = context.db

        # Execution tracking
        self._start_time: Optional[datetime] = None
        self._agent_run: Optional[AgentRun] = None
        self._items_processed = 0
        self._items_produced = 0
        self._api_calls = 0
        self._llm_calls = 0
        self._tokens_used = 0
        self._cost_usd = 0.0
        self._errors: List[str] = []

        # Progress tracking
        self._progress_percent = 0.0
        self._current_item: Optional[str] = None

    @property
    def total_video_limit(self) -> int:
        """Get total video limit from campaign config."""
        return self.config.get("total_video_limit", 3000)

    @property
    def per_channel_limit(self) -> int:
        """Get per-channel video limit from campaign config."""
        return self.config.get("per_channel_limit", 5)

    @property
    def filter_threshold(self) -> float:
        """Get relevance filter threshold from campaign config."""
        return self.config.get("filter_threshold", 0.7)

    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent with full lifecycle management.

        Creates AgentRun record, handles errors, and persists results.

        Args:
            input_data: Input parameters for the agent

        Returns:
            AgentResult with execution results
        """
        self._start_time = datetime.utcnow()

        # Create agent run record
        self._agent_run = AgentRun.create_for_agent(
            campaign_id=self.campaign_id,
            agent_type=self.agent_type,
            input_data=input_data,
            parent_run_id=self.context.parent_run_id
        )
        self.db.add(self._agent_run)
        await self.db.flush()

        try:
            # Start the run
            self._agent_run.start()
            await self.db.flush()

            # Emit started event
            self._emit_event(AgentEventType.STARTED, message=f"{self.agent_type.value} agent started")

            # Run the agent implementation
            result = await self.run(input_data)

            # Complete the run
            self._agent_run.complete(output_data=result.data)
            self._agent_run.items_processed = result.items_processed
            self._agent_run.items_produced = result.items_produced
            self._agent_run.api_calls_made = result.api_calls_made
            self._agent_run.llm_calls_made = result.llm_calls_made
            self._agent_run.tokens_used = result.tokens_used
            self._agent_run.estimated_cost_usd = result.estimated_cost_usd
            if result.errors:
                self._agent_run.errors = [{"message": e, "timestamp": datetime.utcnow().isoformat()} for e in result.errors]

            await self.db.flush()

            # Emit completed event
            self._emit_event(
                AgentEventType.COMPLETED,
                data={"items_processed": result.items_processed, "items_produced": result.items_produced},
                message=f"{self.agent_type.value} agent completed"
            )

            return result

        except asyncio.CancelledError:
            # Handle cancellation
            self._agent_run.cancel()
            await self.db.flush()

            self._emit_event(AgentEventType.CANCELLED, message=f"{self.agent_type.value} agent cancelled")

            return AgentResult(
                success=False,
                items_processed=self._items_processed,
                items_produced=self._items_produced,
                errors=["Agent was cancelled"]
            )

        except Exception as e:
            # Handle errors
            error_msg = str(e)
            logger.exception(f"{self.agent_type.value} agent failed: {error_msg}")

            self._agent_run.fail(error_msg)
            await self.db.flush()

            self._emit_event(
                AgentEventType.ERROR,
                data={"error": error_msg},
                message=f"{self.agent_type.value} agent failed: {error_msg}"
            )

            return AgentResult(
                success=False,
                items_processed=self._items_processed,
                items_produced=self._items_produced,
                errors=[error_msg]
            )

    @abstractmethod
    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's main logic.

        Override this method in subclasses to implement agent-specific behavior.

        Args:
            input_data: Input parameters for the agent

        Returns:
            AgentResult with execution results
        """
        pass

    async def check_should_stop(self) -> bool:
        """
        Check if the agent should stop execution.

        Checks for cancellation and limit reached conditions.
        """
        if self.context.is_cancelled():
            return True

        # Check if campaign has reached video limit
        campaign = self.context.campaign
        if campaign.total_videos_relevant >= self.total_video_limit:
            logger.info(f"Campaign {self.campaign_id} reached video limit ({self.total_video_limit})")
            return True

        return False

    def update_progress(self, percent: float, current_item: Optional[str] = None) -> None:
        """Update progress and emit progress event."""
        self._progress_percent = min(percent, 100.0)
        self._current_item = current_item

        if self._agent_run:
            self._agent_run.update_progress(percent, current_item)

        self._emit_event(
            AgentEventType.PROGRESS,
            data={
                "progress_percent": self._progress_percent,
                "current_item": self._current_item,
                "items_processed": self._items_processed,
                "items_produced": self._items_produced,
            }
        )

    def increment_processed(self, count: int = 1) -> None:
        """Increment items processed count."""
        self._items_processed += count
        if self._agent_run:
            self._agent_run.increment_processed(count)

    def increment_produced(self, count: int = 1) -> None:
        """Increment items produced count."""
        self._items_produced += count
        if self._agent_run:
            self._agent_run.increment_produced(count)

    def increment_api_calls(self, count: int = 1) -> None:
        """Increment API calls count."""
        self._api_calls += count
        if self._agent_run:
            self._agent_run.increment_api_calls(count)

    def increment_llm_calls(self, count: int = 1, tokens: int = 0, cost: float = 0.0) -> None:
        """Increment LLM calls count and track costs."""
        self._llm_calls += count
        self._tokens_used += tokens
        self._cost_usd += cost
        if self._agent_run:
            self._agent_run.increment_llm_calls(count, tokens, cost)

    def add_error(self, error: str) -> None:
        """Add a non-fatal error."""
        self._errors.append(error)
        if self._agent_run:
            self._agent_run.add_error(error)

        self._emit_event(
            AgentEventType.ERROR,
            data={"error": error},
            message=error
        )

    def _emit_event(
        self,
        event_type: AgentEventType,
        data: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ) -> None:
        """Emit an event through the context."""
        event = AgentEvent(
            event_type=event_type,
            agent_type=self.agent_type,
            campaign_id=self.campaign_id,
            data=data or {},
            message=message
        )
        self.context.emit_event(event)

    def _get_duration(self) -> float:
        """Get execution duration in seconds."""
        if self._start_time:
            return (datetime.utcnow() - self._start_time).total_seconds()
        return 0.0

    def _build_result(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """Build an AgentResult with current state."""
        return AgentResult(
            success=success,
            items_processed=self._items_processed,
            items_produced=self._items_produced,
            api_calls_made=self._api_calls,
            llm_calls_made=self._llm_calls,
            tokens_used=self._tokens_used,
            estimated_cost_usd=self._cost_usd,
            data=data or {},
            errors=self._errors.copy(),
            duration_seconds=self._get_duration()
        )
