from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID


class TranscriptSegment(BaseModel):
    """Individual transcript segment with timing information."""
    
    text: str = Field(..., description="The transcript text for this segment")
    start: float = Field(..., description="Start time in seconds")
    duration: float = Field(..., description="Duration of segment in seconds")
    end: float = Field(None, description="End time in seconds")
    
    @validator('end', always=True)
    def calculate_end(cls, v, values):
        if v is None and 'start' in values and 'duration' in values:
            return values['start'] + values['duration']
        return v
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TranscriptData(BaseModel):
    """Complete transcript data with metadata."""
    
    content: str = Field(..., description="Full transcript text")
    segments: List[TranscriptSegment] = Field(
        default_factory=list,
        description="List of timed transcript segments"
    )
    language: str = Field(..., description="Language code (e.g., 'en', 'es')")
    language_code: str = Field(..., description="Full language code (e.g., 'en-US')")
    is_auto_generated: bool = Field(
        default=True,
        description="Whether transcript is auto-generated"
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score of transcript quality"
    )
    has_timestamps: bool = Field(
        default=True,
        description="Whether transcript includes timing information"
    )
    
    @validator('content')
    def validate_content_length(cls, v):
        from ..config import settings
        if len(v) > settings.MAX_TRANSCRIPT_LENGTH:
            raise ValueError(f"Transcript exceeds maximum length of {settings.MAX_TRANSCRIPT_LENGTH} characters")
        return v
    
    @property
    def word_count(self) -> int:
        """Calculate word count of transcript."""
        return len(self.content.split())
    
    @property
    def duration_seconds(self) -> float:
        """Calculate total duration from segments."""
        if not self.segments:
            return 0.0
        return max(seg.end for seg in self.segments if seg.end is not None)
    
    def merge_segments(self, max_duration: float = 30.0) -> List[TranscriptSegment]:
        """
        Merge small segments into larger chunks.
        
        Args:
            max_duration: Maximum duration for merged segments
            
        Returns:
            List of merged segments
        """
        if not self.segments:
            return []
        
        merged = []
        current_segment = None
        
        for segment in self.segments:
            if current_segment is None:
                current_segment = TranscriptSegment(
                    text=segment.text,
                    start=segment.start,
                    duration=segment.duration,
                    end=segment.end
                )
            elif current_segment.duration + segment.duration <= max_duration:
                current_segment.text += " " + segment.text
                current_segment.duration += segment.duration
                current_segment.end = segment.end
            else:
                merged.append(current_segment)
                current_segment = TranscriptSegment(
                    text=segment.text,
                    start=segment.start,
                    duration=segment.duration,
                    end=segment.end
                )
        
        if current_segment:
            merged.append(current_segment)
        
        return merged


class TranscriptInfo(BaseModel):
    """Information about available transcripts for a video."""
    
    language: str = Field(..., description="Language name")
    language_code: str = Field(..., description="Language code")
    is_auto_generated: bool = Field(
        default=False,
        description="Whether transcript is auto-generated"
    )
    is_translatable: bool = Field(
        default=False,
        description="Whether transcript can be translated"
    )
    is_available: bool = Field(
        default=True,
        description="Whether transcript is available for extraction"
    )
    base_url: Optional[str] = Field(
        None,
        description="Base URL for transcript (internal use)"
    )


class TranscriptMetrics(BaseModel):
    """Metrics and analysis results for a transcript."""
    
    word_count: int = Field(..., description="Total word count")
    sentence_count: int = Field(..., description="Total sentence count")
    unique_words: int = Field(..., description="Number of unique words")
    avg_sentence_length: float = Field(..., description="Average sentence length")
    detected_language: str = Field(..., description="Detected language code")
    language_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Language detection confidence"
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall quality score"
    )
    has_timestamps: bool = Field(
        default=True,
        description="Whether transcript has timing data"
    )
    is_complete: bool = Field(
        default=True,
        description="Whether transcript appears complete"
    )
    reading_level: Optional[str] = Field(
        None,
        description="Estimated reading level"
    )
    
    @validator('quality_score')
    def validate_quality_score(cls, v):
        return round(v, 2)
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert metrics to metadata dictionary for storage."""
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "unique_words": self.unique_words,
            "avg_sentence_length": round(self.avg_sentence_length, 1),
            "detected_language": self.detected_language,
            "language_confidence": round(self.language_confidence, 2),
            "quality_score": round(self.quality_score, 2),
            "has_timestamps": self.has_timestamps,
            "is_complete": self.is_complete,
            "reading_level": self.reading_level
        }


class ProcessingResult(BaseModel):
    """Result of batch transcript processing."""
    
    total_processed: int = Field(0, description="Total videos processed")
    successful: int = Field(0, description="Successfully processed count")
    failed: int = Field(0, description="Failed processing count")
    skipped: int = Field(0, description="Skipped videos count")
    errors: List['ProcessingError'] = Field(
        default_factory=list,
        description="List of processing errors"
    )
    processing_time_seconds: float = Field(
        0.0,
        description="Total processing time"
    )
    average_time_per_video: Optional[float] = Field(
        None,
        description="Average processing time per video"
    )
    
    @validator('average_time_per_video', always=True)
    def calculate_average_time(cls, v, values):
        if v is None and values.get('total_processed', 0) > 0:
            return values.get('processing_time_seconds', 0) / values['total_processed']
        return v
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.successful / self.total_processed) * 100
    
    def add_error(self, video_id: UUID, error_type: str, error_message: str):
        """Add an error to the processing result."""
        self.errors.append(ProcessingError(
            video_id=video_id,
            error_type=error_type,
            error_message=error_message,
            timestamp=datetime.utcnow()
        ))
        self.failed += 1
    
    def merge(self, other: 'ProcessingResult') -> 'ProcessingResult':
        """Merge another result into this one."""
        self.total_processed += other.total_processed
        self.successful += other.successful
        self.failed += other.failed
        self.skipped += other.skipped
        self.errors.extend(other.errors)
        self.processing_time_seconds += other.processing_time_seconds
        self.average_time_per_video = None  # Recalculate
        return self


class ProcessingError(BaseModel):
    """Individual processing error details."""
    
    video_id: UUID = Field(..., description="Video ID that failed")
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When error occurred"
    )
    retry_count: int = Field(0, description="Number of retry attempts")
    
    class Config:
        json_encoders = {
            UUID: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }


class TranscriptExtractionRequest(BaseModel):
    """Request model for transcript extraction."""
    
    video_id: UUID = Field(..., description="Video ID to extract transcript for")
    languages: Optional[List[str]] = Field(
        None,
        description="Preferred languages for extraction"
    )
    force_refresh: bool = Field(
        False,
        description="Force refresh even if cached"
    )
    include_timestamps: bool = Field(
        True,
        description="Include segment timestamps"
    )
    clean_content: bool = Field(
        True,
        description="Clean transcript content"
    )
    calculate_metrics: bool = Field(
        True,
        description="Calculate quality metrics"
    )


class TranscriptBatchRequest(BaseModel):
    """Request model for batch transcript processing."""
    
    video_ids: List[UUID] = Field(
        ...,
        description="List of video IDs to process"
    )
    concurrent_limit: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of concurrent extractions"
    )
    languages: Optional[List[str]] = Field(
        None,
        description="Preferred languages for all videos"
    )
    skip_existing: bool = Field(
        True,
        description="Skip videos that already have transcripts"
    )
    retry_failed: bool = Field(
        False,
        description="Retry previously failed extractions"
    )