from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from enum import Enum


class IdeaStatus(str, Enum):
    EXTRACTED = "extracted"
    REVIEWED = "reviewed"
    SELECTED = "selected"
    REJECTED = "rejected"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"


class IdeaPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IdeaBase(BaseModel):
    """Base idea schema"""
    title: str = Field(..., max_length=500)
    description: str
    category: Optional[str] = Field(None, max_length=100)
    priority: IdeaPriority = IdeaPriority.MEDIUM
    confidence_score: float = Field(..., ge=0, le=1)
    complexity_score: Optional[int] = Field(None, ge=1, le=10)
    market_size_estimate: Optional[str] = Field(None, max_length=50)
    target_audience: Optional[str] = Field(None, max_length=200)
    implementation_time_estimate: Optional[str] = Field(None, max_length=50)
    tags: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    competitive_advantage: Optional[str] = None
    potential_challenges: List[str] = Field(default_factory=list)
    monetization_strategies: List[str] = Field(default_factory=list)


class IdeaCreate(IdeaBase):
    """Schema for creating an idea"""
    video_id: UUID
    source_timestamp: Optional[int] = None
    source_context: Optional[str] = None
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)


class IdeaUpdate(BaseModel):
    """Schema for updating an idea"""
    status: Optional[IdeaStatus] = None
    priority: Optional[IdeaPriority] = None
    category: Optional[str] = Field(None, max_length=100)
    review_notes: Optional[str] = None
    tags: Optional[List[str]] = None
    technologies: Optional[List[str]] = None


class IdeaResponse(BaseModel):
    """Idea response schema"""
    id: UUID
    video_id: UUID
    title: str
    description: str
    category: Optional[str]
    status: IdeaStatus
    priority: IdeaPriority
    confidence_score: float
    complexity_score: Optional[int]
    market_size_estimate: Optional[str]
    target_audience: Optional[str]
    implementation_time_estimate: Optional[str]
    source_timestamp: Optional[int]
    source_context: Optional[str]
    tags: List[str]
    technologies: List[str]
    competitive_advantage: Optional[str]
    potential_challenges: List[str]
    monetization_strategies: List[str]
    review_notes: Optional[str]
    reviewed_by: Optional[UUID]
    reviewed_at: Optional[datetime]
    selected_at: Optional[datetime]
    selected_by: Optional[UUID]
    export_count: int
    last_exported_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class IdeaWithContext(IdeaResponse):
    """Idea with video and channel context"""
    video: Dict[str, Any]
    channel: Dict[str, Any]
    transcript_excerpt: Optional[str] = None


class IdeaListResponse(BaseModel):
    """Idea list response schema"""
    items: List[IdeaWithContext]
    total: int
    limit: int
    offset: int
    has_more: bool


class IdeaBulkAction(BaseModel):
    """Schema for bulk idea actions"""
    action: str = Field(..., pattern="^(select|reject|review|update_category)$")
    idea_ids: List[UUID]
    category: Optional[str] = Field(None, max_length=100)


class IdeaBulkActionResponse(BaseModel):
    """Bulk action response schema"""
    updated: int
    errors: List[Dict[str, Any]]