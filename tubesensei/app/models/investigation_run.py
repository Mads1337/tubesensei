"""
InvestigationRun model - Tracks individual LLM-based investigation runs for ideas.

Each InvestigationAgent execution against a specific idea creates an InvestigationRun
record, enabling detailed monitoring and auditability of the investigation pipeline.
"""
import enum

from sqlalchemy import Column, Integer, Float, Text, ForeignKey, Index
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.models.base import BaseModel


class InvestigationRunStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class InvestigationRun(BaseModel):
    """
    Tracks execution of a single investigation agent run against an idea.

    Stores the raw LLM output, structured results, token usage, cost estimates,
    and any error details for each investigation invocation.
    """
    __tablename__ = "investigation_runs"

    agent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("investigation_agents.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    idea_id = Column(
        UUID(as_uuid=True),
        ForeignKey("ideas.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    status = Column(
        SQLEnum(InvestigationRunStatus),
        nullable=False,
        default=InvestigationRunStatus.PENDING,
        index=True
    )

    result = Column(
        Text,
        nullable=True
    )

    result_structured = Column(
        JSONB,
        nullable=True
    )

    tokens_used = Column(
        Integer,
        nullable=True
    )

    estimated_cost_usd = Column(
        Float,
        nullable=True
    )

    error_message = Column(
        Text,
        nullable=True
    )

    # Relationships
    agent = relationship("InvestigationAgent", backref="runs", lazy="joined")
    idea = relationship("Idea", backref="investigation_runs", lazy="select")

    __table_args__ = (
        Index("idx_investigation_run_agent_idea", "agent_id", "idea_id"),
    )

    def __repr__(self) -> str:
        return f"<InvestigationRun(id={self.id}, agent_id={self.agent_id}, status={self.status.value})>"
