from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Enum as SQLEnum, Index, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import enum
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from app.models.base import BaseModel


class JobType(enum.Enum):
    CHANNEL_DISCOVERY = "channel_discovery"
    VIDEO_DISCOVERY = "video_discovery"
    TRANSCRIPT_EXTRACTION = "transcript_extraction"
    VALUABLE_DETECTION = "valuable_detection"
    IDEA_EXTRACTION = "idea_extraction"
    BULK_PROCESSING = "bulk_processing"
    CLEANUP = "cleanup"


class JobStatus(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(enum.Enum):
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class ProcessingJob(BaseModel):
    __tablename__ = "processing_jobs"
    
    job_type = Column(
        SQLEnum(JobType),
        nullable=False,
        index=True
    )
    
    status = Column(
        SQLEnum(JobStatus),
        nullable=False,
        default=JobStatus.PENDING,
        index=True
    )
    
    priority = Column(
        SQLEnum(JobPriority),
        nullable=False,
        default=JobPriority.NORMAL,
        index=True
    )
    
    entity_type = Column(
        String(50),
        nullable=False,
        index=True
    )
    
    entity_id = Column(
        UUID(as_uuid=True),
        nullable=False,
        index=True
    )
    
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("processing_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True
    )
    
    started_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    completed_at = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    scheduled_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
    
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
    
    retry_after = Column(
        DateTime(timezone=True),
        nullable=True
    )
    
    progress_percent = Column(
        Float,
        nullable=False,
        default=0.0
    )
    
    progress_message = Column(
        String,
        nullable=True
    )
    
    input_data = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    output_data = Column(
        JSONB,
        nullable=True
    )
    
    error_message = Column(
        String,
        nullable=True
    )
    
    error_details = Column(
        JSONB,
        nullable=True
    )
    
    metadata = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    worker_id = Column(
        String(100),
        nullable=True
    )
    
    execution_time_seconds = Column(
        Float,
        nullable=True
    )
    
    session = relationship(
        "ProcessingSession",
        back_populates="jobs",
        lazy="joined"
    )
    
    __table_args__ = (
        Index("idx_job_status_priority", "status", "priority"),
        Index("idx_job_entity", "entity_type", "entity_id"),
        Index("idx_job_scheduled", "scheduled_at", "status"),
        Index("idx_job_type_status", "job_type", "status"),
        Index("idx_job_session", "session_id"),
    )
    
    def __repr__(self) -> str:
        return f"<ProcessingJob(id={self.id}, type={self.job_type.value}, status={self.status.value})>"
    
    @property
    def is_complete(self) -> bool:
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    @property
    def is_running(self) -> bool:
        return self.status == JobStatus.RUNNING
    
    @property
    def can_retry(self) -> bool:
        return (
            self.status == JobStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def start(self, worker_id: Optional[str] = None) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        self.worker_id = worker_id
        self.progress_percent = 0.0
    
    def complete(self, output_data: Optional[Dict[str, Any]] = None) -> None:
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress_percent = 100.0
        if output_data:
            self.output_data = output_data
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def fail(self, error_message: str, error_details: Optional[Dict[str, Any]] = None) -> None:
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        if error_details:
            self.error_details = error_details
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def retry(self, delay_seconds: int = 60) -> None:
        if not self.can_retry:
            raise ValueError("Job cannot be retried")
        
        self.status = JobStatus.RETRYING
        self.retry_count += 1
        self.retry_after = datetime.utcnow() + timedelta(seconds=delay_seconds)
        self.error_message = None
        self.error_details = None
    
    def cancel(self) -> None:
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()
        if self.started_at:
            self.execution_time_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def update_progress(self, percent: float, message: Optional[str] = None) -> None:
        self.progress_percent = min(100.0, max(0.0, percent))
        if message:
            self.progress_message = message