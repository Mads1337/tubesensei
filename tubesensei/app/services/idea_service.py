from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from datetime import datetime
from uuid import UUID
import json

from app.models.idea import Idea, IdeaStatus, IdeaPriority
from app.models.video import Video
from app.models.channel import Channel
from app.models.transcript import Transcript
from app.core.exceptions import NotFoundException


class IdeaService:
    """Service for idea operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_ideas(
        self,
        status: Optional[IdeaStatus] = None,
        min_confidence: float = 0.0,
        category: Optional[str] = None,
        channel_id: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List ideas with filtering"""
        query = select(Idea).join(Video)
        
        # Apply filters
        conditions = []
        
        if status:
            conditions.append(Idea.status == status)
        
        if min_confidence > 0:
            conditions.append(Idea.confidence_score >= min_confidence)
        
        if category:
            conditions.append(Idea.category == category)
        
        if channel_id:
            conditions.append(Video.channel_id == channel_id)
        
        if search:
            conditions.append(
                or_(
                    Idea.title.ilike(f"%{search}%"),
                    Idea.description.ilike(f"%{search}%")
                )
            )
        
        if conditions:
            query = query.where(and_(*conditions))
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.scalar(count_query)
        
        # Get paginated results
        query = query.order_by(Idea.confidence_score.desc(), Idea.created_at.desc())
        query = query.limit(limit).offset(offset)
        
        result = await self.db.execute(query)
        ideas = result.scalars().all()
        
        # Enrich with video and channel data
        enriched = []
        for idea in ideas:
            video = await self.db.get(Video, idea.video_id)
            channel = await self.db.get(Channel, video.channel_id)
            
            idea_dict = idea.to_dict()
            idea_dict["video"] = {
                "id": str(video.id),
                "title": video.title,
                "url": video.youtube_url,
                "thumbnail": video.thumbnail_url
            }
            idea_dict["channel"] = {
                "id": str(channel.id),
                "name": channel.name
            }
            enriched.append(idea_dict)
        
        return {
            "items": enriched,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
    
    async def get_idea(self, idea_id: str) -> Idea:
        """Get idea by ID"""
        idea = await self.db.get(Idea, idea_id)
        if not idea:
            raise NotFoundException("Idea", idea_id)
        return idea
    
    async def create_idea(self, data: Dict[str, Any]) -> Idea:
        """Create a new idea"""
        idea = Idea(
            video_id=data["video_id"],
            title=data["title"],
            description=data["description"],
            category=data.get("category"),
            status=IdeaStatus.EXTRACTED,
            priority=data.get("priority", IdeaPriority.MEDIUM),
            confidence_score=data.get("confidence_score", 0.5),
            complexity_score=data.get("complexity_score"),
            market_size_estimate=data.get("market_size_estimate"),
            target_audience=data.get("target_audience"),
            implementation_time_estimate=data.get("implementation_time_estimate"),
            source_timestamp=data.get("source_timestamp"),
            source_context=data.get("source_context"),
            tags=data.get("tags", []),
            technologies=data.get("technologies", []),
            competitive_advantage=data.get("competitive_advantage"),
            potential_challenges=data.get("potential_challenges", []),
            monetization_strategies=data.get("monetization_strategies", []),
            extraction_metadata=data.get("extraction_metadata", {})
        )
        
        self.db.add(idea)
        await self.db.commit()
        await self.db.refresh(idea)
        
        return idea
    
    async def update_idea(
        self,
        idea_id: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Idea:
        """Update idea"""
        idea = await self.get_idea(idea_id)
        
        # Update fields
        for field, value in data.items():
            if hasattr(idea, field) and field not in ["id", "created_at", "video_id"]:
                setattr(idea, field, value)
        
        # Handle status changes
        if "status" in data:
            if data["status"] == IdeaStatus.REVIEWED and user_id:
                idea.reviewed_by = UUID(user_id)
                idea.reviewed_at = datetime.utcnow()
            elif data["status"] == IdeaStatus.SELECTED and user_id:
                idea.selected_by = UUID(user_id)
                idea.selected_at = datetime.utcnow()
                if not idea.reviewed_by:
                    idea.reviewed_by = UUID(user_id)
                    idea.reviewed_at = datetime.utcnow()
        
        idea.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(idea)
        
        return idea
    
    async def bulk_update(
        self,
        idea_ids: List[str],
        action: str,
        user_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform bulk update on ideas"""
        updated = 0
        errors = []
        
        for idea_id in idea_ids:
            try:
                idea = await self.get_idea(idea_id)
                
                if action == "select":
                    idea.select(UUID(user_id) if user_id else None)
                elif action == "reject":
                    idea.reject(UUID(user_id) if user_id else None)
                elif action == "review":
                    idea.mark_as_reviewed(UUID(user_id) if user_id else None)
                elif action == "update_category":
                    idea.category = kwargs.get("category")
                
                idea.updated_at = datetime.utcnow()
                updated += 1
                
            except Exception as e:
                errors.append({
                    "idea_id": idea_id,
                    "error": str(e)
                })
        
        await self.db.commit()
        
        return {
            "updated": updated,
            "errors": errors
        }
    
    async def get_categories(self) -> List[str]:
        """Get all unique categories"""
        result = await self.db.execute(
            select(Idea.category).distinct().where(
                Idea.category.isnot(None)
            )
        )
        return [row[0] for row in result]
    
    async def get_idea_context(self, idea_id: str) -> Dict[str, Any]:
        """Get full context for an idea"""
        idea = await self.get_idea(idea_id)
        video = await self.db.get(Video, idea.video_id)
        channel = await self.db.get(Channel, video.channel_id)
        
        # Get transcript excerpt if available
        transcript_result = await self.db.execute(
            select(Transcript).where(Transcript.video_id == video.id)
        )
        transcript = transcript_result.scalar_one_or_none()
        
        return {
            "idea": idea.to_dict(),
            "video": {
                "id": str(video.id),
                "title": video.title,
                "description": video.description,
                "url": video.youtube_url,
                "published_at": video.published_at.isoformat(),
                "duration": video.duration_seconds,
                "views": video.view_count
            },
            "channel": {
                "id": str(channel.id),
                "name": channel.name,
                "url": channel.channel_url
            },
            "transcript_excerpt": self._get_transcript_excerpt(
                transcript,
                idea.source_timestamp
            ) if transcript else None
        }
    
    async def export_ideas(
        self,
        idea_ids: List[str],
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export selected ideas"""
        ideas = []
        
        for idea_id in idea_ids:
            idea = await self.get_idea(idea_id)
            context = await self.get_idea_context(idea_id)
            ideas.append(context)
            
            # Mark as exported
            idea.mark_as_exported()
        
        await self.db.commit()
        
        if format == "json":
            return {
                "format": "json",
                "data": ideas,
                "exported_at": datetime.utcnow().isoformat(),
                "count": len(ideas)
            }
        elif format == "csv":
            # Convert to CSV format
            import csv
            import io
            
            output = io.StringIO()
            if ideas:
                writer = csv.DictWriter(
                    output,
                    fieldnames=[
                        "title", "description", "category", "status",
                        "confidence_score", "channel_name", "video_title",
                        "video_url", "created_at"
                    ]
                )
                writer.writeheader()
                
                for idea_context in ideas:
                    idea = idea_context["idea"]
                    writer.writerow({
                        "title": idea["title"],
                        "description": idea["description"],
                        "category": idea.get("category", ""),
                        "status": idea["status"],
                        "confidence_score": idea["confidence_score"],
                        "channel_name": idea_context["channel"]["name"],
                        "video_title": idea_context["video"]["title"],
                        "video_url": idea_context["video"]["url"],
                        "created_at": idea["created_at"]
                    })
            
            return {
                "format": "csv",
                "data": output.getvalue(),
                "exported_at": datetime.utcnow().isoformat(),
                "count": len(ideas)
            }
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _get_transcript_excerpt(
        self,
        transcript,
        timestamp: Optional[int],
        context_seconds: int = 60
    ) -> Optional[str]:
        """Get transcript excerpt around timestamp"""
        if not transcript or not timestamp:
            return None
        
        # Parse transcript content
        try:
            if isinstance(transcript.content, str):
                content = json.loads(transcript.content)
            else:
                content = transcript.content
            
            # Find segments around timestamp
            segments = []
            start_time = max(0, timestamp - context_seconds)
            end_time = timestamp + context_seconds
            
            for segment in content.get("segments", []):
                seg_start = segment.get("start", 0)
                seg_end = segment.get("end", 0)
                
                if seg_start <= end_time and seg_end >= start_time:
                    segments.append(segment.get("text", ""))
            
            return " ".join(segments)
        except:
            # If parsing fails, return a simple excerpt
            words = str(transcript.content).split()
            start_word = max(0, timestamp - context_seconds) * 2  # Rough estimate
            end_word = min(len(words), (timestamp + context_seconds) * 2)
            
            return " ".join(words[start_word:end_word])