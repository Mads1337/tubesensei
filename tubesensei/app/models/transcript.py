from sqlalchemy import Column, String, ForeignKey, Enum as SQLEnum, Index, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import enum
from typing import Optional, Dict, Any, List
from datetime import datetime

from app.models.base import BaseModel


class TranscriptSource(enum.Enum):
    YOUTUBE_AUTO = "youtube_auto"
    YOUTUBE_MANUAL = "youtube_manual"
    WHISPER = "whisper"
    CUSTOM = "custom"


class TranscriptLanguage(enum.Enum):
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    PT = "pt"
    IT = "it"
    JA = "ja"
    KO = "ko"
    ZH = "zh"
    OTHER = "other"


class Transcript(BaseModel):
    __tablename__ = "transcripts"
    
    video_id = Column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    content = Column(
        String,
        nullable=False
    )
    
    source = Column(
        SQLEnum(TranscriptSource),
        nullable=False,
        default=TranscriptSource.YOUTUBE_AUTO,
        index=True
    )
    
    language = Column(
        SQLEnum(TranscriptLanguage),
        nullable=False,
        default=TranscriptLanguage.EN
    )
    
    language_code = Column(
        String(10),
        nullable=False,
        default="en"
    )
    
    is_auto_generated = Column(
        Boolean,
        nullable=False,
        default=True
    )
    
    word_count = Column(
        Integer,
        nullable=True
    )
    
    char_count = Column(
        Integer,
        nullable=True
    )
    
    confidence_score = Column(
        Integer,
        nullable=True
    )
    
    transcript_metadata = Column(
        JSONB,
        nullable=False,
        default=dict
    )
    
    segments = Column(
        JSONB,
        nullable=True
    )
    
    processed_content = Column(
        String,
        nullable=True
    )
    
    is_complete = Column(
        Boolean,
        nullable=False,
        default=True
    )
    
    video = relationship(
        "Video",
        back_populates="transcripts",
        lazy="joined"
    )
    
    __table_args__ = (
        Index("idx_transcript_video_source", "video_id", "source"),
        Index("idx_transcript_language", "language", "language_code"),
    )
    
    def __repr__(self) -> str:
        return f"<Transcript(id={self.id}, video_id={self.video_id}, source={self.source.value}, language={self.language_code})>"
    
    @property
    def content_preview(self) -> str:
        if not self.content:
            return ""
        return self.content[:500] + "..." if len(self.content) > 500 else self.content
    
    @property
    def is_english(self) -> bool:
        return self.language == TranscriptLanguage.EN
    
    @property
    def needs_translation(self) -> bool:
        return not self.is_english and self.language != TranscriptLanguage.OTHER
    
    def calculate_stats(self) -> None:
        if self.content:
            self.word_count = len(self.content.split())
            self.char_count = len(self.content)
    
    def clean_content(self) -> str:
        if not self.content:
            return ""
        
        import re
        
        cleaned = self.content
        cleaned = re.sub(r'\[.*?\]', '', cleaned)
        cleaned = re.sub(r'\(.*?\)', '', cleaned)
        cleaned = re.sub(r'<.*?>', '', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def extract_timestamps(self) -> List[Dict[str, Any]]:
        if not self.segments:
            return []
        
        timestamps = []
        for segment in self.segments:
            if isinstance(segment, dict) and 'start' in segment and 'text' in segment:
                timestamps.append({
                    'start': segment.get('start', 0),
                    'end': segment.get('end', 0),
                    'duration': segment.get('duration', 0),
                    'text': segment.get('text', '')
                })
        
        return timestamps