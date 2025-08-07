import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

from ..models.transcript import Transcript, TranscriptSource, TranscriptLanguage
from ..models.video import Video, VideoStatus
from ..models.transcript_data import TranscriptMetrics

logger = logging.getLogger(__name__)


class TranscriptRepository:
    """
    Repository for transcript database operations.
    Handles CRUD operations and complex queries for transcripts.
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        logger.debug("Initialized TranscriptRepository")
    
    async def create(
        self,
        video_id: UUID,
        content: str,
        source: TranscriptSource,
        language: str,
        language_code: str,
        is_auto_generated: bool = True,
        segments: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence_score: Optional[float] = None
    ) -> Transcript:
        """
        Create a new transcript record.
        
        Args:
            video_id: Video UUID
            content: Transcript content
            source: Transcript source
            language: Language enum value
            language_code: Full language code
            is_auto_generated: Whether auto-generated
            segments: Transcript segments with timestamps
            metadata: Additional metadata
            confidence_score: Quality confidence score
            
        Returns:
            Created Transcript object
        """
        try:
            # Map language code to enum
            language_enum = self._map_language_to_enum(language)
            
            transcript = Transcript(
                video_id=video_id,
                content=content,
                source=source,
                language=language_enum,
                language_code=language_code,
                is_auto_generated=is_auto_generated,
                segments=segments,
                metadata=metadata or {},
                confidence_score=int(confidence_score * 100) if confidence_score else None
            )
            
            # Calculate basic stats
            transcript.calculate_stats()
            
            self.session.add(transcript)
            await self.session.commit()
            await self.session.refresh(transcript)
            
            logger.info(f"Created transcript for video {video_id} (language: {language_code})")
            return transcript
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error creating transcript for video {video_id}: {e}")
            raise
    
    async def get_by_id(
        self,
        transcript_id: UUID,
        include_video: bool = False
    ) -> Optional[Transcript]:
        """
        Get transcript by ID.
        
        Args:
            transcript_id: Transcript UUID
            include_video: Whether to include video relationship
            
        Returns:
            Transcript object or None
        """
        query = select(Transcript).where(Transcript.id == transcript_id)
        
        if include_video:
            query = query.options(joinedload(Transcript.video))
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_by_video(
        self,
        video_id: UUID,
        source: Optional[TranscriptSource] = None,
        language: Optional[str] = None
    ) -> Optional[Transcript]:
        """
        Get transcript for a specific video.
        
        Args:
            video_id: Video UUID
            source: Optional source filter
            language: Optional language filter
            
        Returns:
            Transcript object or None
        """
        query = select(Transcript).where(Transcript.video_id == video_id)
        
        if source:
            query = query.where(Transcript.source == source)
        
        if language:
            language_enum = self._map_language_to_enum(language)
            query = query.where(Transcript.language == language_enum)
        
        # Prefer manual over auto-generated
        query = query.order_by(
            Transcript.is_auto_generated.asc(),
            Transcript.confidence_score.desc()
        )
        
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    async def get_all_by_video(
        self,
        video_id: UUID
    ) -> List[Transcript]:
        """
        Get all transcripts for a video.
        
        Args:
            video_id: Video UUID
            
        Returns:
            List of Transcript objects
        """
        query = select(Transcript).where(
            Transcript.video_id == video_id
        ).order_by(
            Transcript.is_auto_generated.asc(),
            Transcript.language
        )
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def exists(
        self,
        video_id: UUID,
        source: Optional[TranscriptSource] = None,
        language: Optional[str] = None
    ) -> bool:
        """
        Check if transcript exists for video.
        
        Args:
            video_id: Video UUID
            source: Optional source filter
            language: Optional language filter
            
        Returns:
            True if exists
        """
        query = select(func.count(Transcript.id)).where(
            Transcript.video_id == video_id
        )
        
        if source:
            query = query.where(Transcript.source == source)
        
        if language:
            language_enum = self._map_language_to_enum(language)
            query = query.where(Transcript.language == language_enum)
        
        result = await self.session.execute(query)
        count = result.scalar()
        return count > 0
    
    async def update(
        self,
        transcript_id: UUID,
        **update_data
    ) -> Optional[Transcript]:
        """
        Update transcript record.
        
        Args:
            transcript_id: Transcript UUID
            **update_data: Fields to update
            
        Returns:
            Updated Transcript or None
        """
        try:
            # Get existing transcript
            transcript = await self.get_by_id(transcript_id)
            if not transcript:
                return None
            
            # Update fields
            for key, value in update_data.items():
                if hasattr(transcript, key):
                    setattr(transcript, key, value)
            
            # Recalculate stats if content changed
            if 'content' in update_data or 'processed_content' in update_data:
                transcript.calculate_stats()
            
            transcript.updated_at = datetime.utcnow()
            
            await self.session.commit()
            await self.session.refresh(transcript)
            
            logger.info(f"Updated transcript {transcript_id}")
            return transcript
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error updating transcript {transcript_id}: {e}")
            raise
    
    async def update_quality_metrics(
        self,
        transcript_id: UUID,
        metrics: TranscriptMetrics
    ) -> Optional[Transcript]:
        """
        Update transcript quality metrics.
        
        Args:
            transcript_id: Transcript UUID
            metrics: TranscriptMetrics object
            
        Returns:
            Updated Transcript or None
        """
        transcript = await self.get_by_id(transcript_id)
        if not transcript:
            return None
        
        # Update metrics in metadata
        transcript.metadata = transcript.metadata or {}
        transcript.metadata.update(metrics.to_metadata())
        
        # Update confidence score
        transcript.confidence_score = int(metrics.quality_score * 100)
        
        # Update completeness
        transcript.is_complete = metrics.is_complete
        
        # Update word count
        transcript.word_count = metrics.word_count
        
        transcript.updated_at = datetime.utcnow()
        
        await self.session.commit()
        await self.session.refresh(transcript)
        
        logger.info(f"Updated quality metrics for transcript {transcript_id}")
        return transcript
    
    async def delete(self, transcript_id: UUID) -> bool:
        """
        Delete transcript record.
        
        Args:
            transcript_id: Transcript UUID
            
        Returns:
            True if deleted
        """
        try:
            result = await self.session.execute(
                delete(Transcript).where(Transcript.id == transcript_id)
            )
            await self.session.commit()
            
            if result.rowcount > 0:
                logger.info(f"Deleted transcript {transcript_id}")
                return True
            return False
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error deleting transcript {transcript_id}: {e}")
            raise
    
    async def get_videos_without_transcripts(
        self,
        limit: int = 100,
        status_filter: Optional[VideoStatus] = None
    ) -> List[Video]:
        """
        Get videos that don't have transcripts.
        
        Args:
            limit: Maximum number of results
            status_filter: Optional video status filter
            
        Returns:
            List of Video objects
        """
        # Subquery for videos with transcripts
        has_transcript = select(Transcript.video_id).subquery()
        
        query = select(Video).where(
            ~Video.id.in_(has_transcript)
        )
        
        if status_filter:
            query = query.where(Video.status == status_filter)
        else:
            # Default to discovered videos
            query = query.where(
                Video.status.in_([VideoStatus.DISCOVERED, VideoStatus.QUEUED])
            )
        
        # Prioritize videos with captions
        query = query.where(Video.has_captions == True)
        
        # Order by discovery date
        query = query.order_by(Video.discovered_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_failed_extractions(
        self,
        limit: int = 100,
        max_retries: int = 3
    ) -> List[Video]:
        """
        Get videos with failed transcript extraction attempts.
        
        Args:
            limit: Maximum number of results
            max_retries: Maximum retry count filter
            
        Returns:
            List of Video objects
        """
        # Subquery for videos with successful transcripts
        has_transcript = select(Transcript.video_id).subquery()
        
        query = select(Video).where(
            and_(
                Video.status == VideoStatus.FAILED,
                ~Video.id.in_(has_transcript),
                Video.retry_count < max_retries,
                Video.has_captions == True
            )
        )
        
        # Order by retry count and age
        query = query.order_by(
            Video.retry_count.asc(),
            Video.updated_at.asc()
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_low_quality_transcripts(
        self,
        quality_threshold: float = 0.5,
        limit: int = 100
    ) -> List[Transcript]:
        """
        Get transcripts with low quality scores.
        
        Args:
            quality_threshold: Quality score threshold
            limit: Maximum number of results
            
        Returns:
            List of Transcript objects
        """
        query = select(Transcript).where(
            or_(
                Transcript.confidence_score < int(quality_threshold * 100),
                Transcript.confidence_score.is_(None)
            )
        ).options(
            joinedload(Transcript.video)
        ).order_by(
            Transcript.confidence_score.asc()
        ).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get transcript statistics.
        
        Returns:
            Dictionary of statistics
        """
        # Total transcripts
        total_query = select(func.count(Transcript.id))
        total_result = await self.session.execute(total_query)
        total_count = total_result.scalar()
        
        # By source
        source_query = select(
            Transcript.source,
            func.count(Transcript.id)
        ).group_by(Transcript.source)
        source_result = await self.session.execute(source_query)
        source_stats = {row[0].value: row[1] for row in source_result}
        
        # By language
        language_query = select(
            Transcript.language,
            func.count(Transcript.id)
        ).group_by(Transcript.language)
        language_result = await self.session.execute(language_query)
        language_stats = {row[0].value: row[1] for row in language_result}
        
        # Auto-generated vs manual
        auto_query = select(func.count(Transcript.id)).where(
            Transcript.is_auto_generated == True
        )
        auto_result = await self.session.execute(auto_query)
        auto_count = auto_result.scalar()
        
        # Average quality score
        quality_query = select(func.avg(Transcript.confidence_score)).where(
            Transcript.confidence_score.isnot(None)
        )
        quality_result = await self.session.execute(quality_query)
        avg_quality = quality_result.scalar()
        
        # Word count statistics
        word_query = select(
            func.sum(Transcript.word_count),
            func.avg(Transcript.word_count),
            func.min(Transcript.word_count),
            func.max(Transcript.word_count)
        ).where(Transcript.word_count.isnot(None))
        word_result = await self.session.execute(word_query)
        word_stats = word_result.one()
        
        return {
            "total_transcripts": total_count,
            "by_source": source_stats,
            "by_language": language_stats,
            "auto_generated_count": auto_count,
            "manual_count": total_count - auto_count,
            "average_quality_score": avg_quality / 100 if avg_quality else None,
            "word_statistics": {
                "total_words": word_stats[0] or 0,
                "average_words": word_stats[1] or 0,
                "min_words": word_stats[2] or 0,
                "max_words": word_stats[3] or 0
            }
        }
    
    async def bulk_create(
        self,
        transcripts: List[Dict[str, Any]]
    ) -> List[Transcript]:
        """
        Bulk create transcript records.
        
        Args:
            transcripts: List of transcript data dictionaries
            
        Returns:
            List of created Transcript objects
        """
        try:
            transcript_objects = []
            
            for data in transcripts:
                # Map language to enum
                language = data.get('language', 'en')
                language_enum = self._map_language_to_enum(language)
                
                transcript = Transcript(
                    video_id=data['video_id'],
                    content=data['content'],
                    source=data.get('source', TranscriptSource.YOUTUBE_AUTO),
                    language=language_enum,
                    language_code=data.get('language_code', language),
                    is_auto_generated=data.get('is_auto_generated', True),
                    segments=data.get('segments'),
                    metadata=data.get('metadata', {}),
                    confidence_score=int(data.get('confidence_score', 0.5) * 100)
                )
                
                # Calculate stats
                transcript.calculate_stats()
                transcript_objects.append(transcript)
            
            self.session.add_all(transcript_objects)
            await self.session.commit()
            
            logger.info(f"Bulk created {len(transcript_objects)} transcripts")
            return transcript_objects
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Error in bulk create: {e}")
            raise
    
    def _map_language_to_enum(self, language: str) -> TranscriptLanguage:
        """
        Map language code to TranscriptLanguage enum.
        
        Args:
            language: Language code (e.g., 'en', 'es')
            
        Returns:
            TranscriptLanguage enum value
        """
        # Extract base language code if needed
        base_lang = language.split('-')[0].lower()
        
        language_map = {
            'en': TranscriptLanguage.EN,
            'es': TranscriptLanguage.ES,
            'fr': TranscriptLanguage.FR,
            'de': TranscriptLanguage.DE,
            'pt': TranscriptLanguage.PT,
            'it': TranscriptLanguage.IT,
            'ja': TranscriptLanguage.JA,
            'ko': TranscriptLanguage.KO,
            'zh': TranscriptLanguage.ZH
        }
        
        return language_map.get(base_lang, TranscriptLanguage.OTHER)