"""
AI-powered video filtering to identify high-value content.

This module provides:
- Intelligent video relevance scoring
- Batch processing for efficiency
- Learning from feedback
- Performance metrics tracking
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.llm_manager import LLMManager, ModelType
from app.ai.prompt_templates import PromptTemplates, PromptType
from app.ai.response_parser import ResponseParser
from app.models.video import Video, VideoStatus
from app.models.processing_job import ProcessingJob, JobType, JobStatus
from app.database import get_db_context
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class VideoFilter:
    """AI-powered video filtering system."""
    
    def __init__(self):
        """Initialize video filter."""
        self.llm_manager = LLMManager()
        self.metrics = {
            "total_processed": 0,
            "filtered_in": 0,
            "filtered_out": 0,
            "average_confidence": 0.0,
            "processing_time": 0.0
        }
    
    async def initialize(self):
        """Initialize the video filter."""
        await self.llm_manager.initialize()
    
    async def filter_video(
        self,
        video: Video,
        use_fast_model: bool = True
    ) -> Dict[str, Any]:
        """
        Filter a single video using AI analysis.
        
        Args:
            video: Video object to analyze
            use_fast_model: Use faster/cheaper model for initial filtering
            
        Returns:
            Filtering result with score and reasoning
        """
        try:
            # Prepare prompt variables
            variables = {
                "title": video.title,
                "description": video.description or "No description available",
                "channel_name": video.channel.name if video.channel else "Unknown",
                "duration_minutes": video.duration_seconds // 60 if video.duration_seconds else 0,
                "view_count": video.view_count or 0,
                "published_date": video.published_at.strftime("%Y-%m-%d") if video.published_at else "Unknown"
            }
            
            # Get prompt
            system_prompt, user_prompt = PromptTemplates.get_prompt(
                PromptType.VIDEO_FILTER,
                variables
            )
            
            # Prepare messages for LLM manager
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            
            # Get LLM response
            model_type = ModelType.FAST if use_fast_model else ModelType.BALANCED
            response = await self.llm_manager.complete(
                messages=messages,
                model_type=model_type,
                temperature=0.3,  # Lower temperature for more consistent filtering
                max_tokens=500
            )
            
            # Parse response
            parsed_result = ResponseParser.parse_video_filter_response(response.content)
            
            # Convert to dict format expected by caller
            result = {
                "video_id": str(video.id),
                "is_valuable": parsed_result.is_valuable,
                "confidence_score": parsed_result.confidence_score,
                "reasoning": parsed_result.reasoning,
                "detected_topics": parsed_result.detected_topics,
                "predicted_idea_count": parsed_result.predicted_idea_count,
                "processing_cost": response.cost,
                "model_used": response.model,
                "processing_time": response.processing_time,
                "error": False
            }
            
            # Update metrics
            self._update_metrics(result)
            
            logger.info("Video filtered",
                       video_id=video.id,
                       title=video.title,
                       is_valuable=result["is_valuable"],
                       confidence=result["confidence_score"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error filtering video {video.id}: {str(e)}")
            return {
                "video_id": str(video.id),
                "is_valuable": False,
                "confidence_score": 0.0,
                "reasoning": f"Error during filtering: {str(e)}",
                "detected_topics": [],
                "predicted_idea_count": 0,
                "error": True,
                "processing_cost": 0.0,
                "model_used": "error",
                "processing_time": 0.0
            }
    
    async def filter_videos_batch(
        self,
        videos: List[Video],
        batch_size: int = 10,
        use_fast_model: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Filter multiple videos in batches for efficiency.
        
        Args:
            videos: List of videos to filter
            batch_size: Number of videos to process concurrently
            use_fast_model: Use faster model for initial filtering
            
        Returns:
            List of filtering results
        """
        results = []
        
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.filter_video(video, use_fast_model)
                for video in batch
            ]
            
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            
            # Small delay between batches to respect rate limits
            if i + batch_size < len(videos):
                await asyncio.sleep(1)
        
        return results
    
    async def apply_filtering_to_channel(
        self,
        channel_id: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Apply AI filtering to all unprocessed videos from a channel.
        
        Args:
            channel_id: Channel ID to process
            limit: Maximum number of videos to process
            
        Returns:
            Summary of filtering results
        """
        async with get_db_context() as session:
            # Get unprocessed videos
            query = select(Video).where(
                Video.channel_id == channel_id,
                Video.status == VideoStatus.DISCOVERED
            ).order_by(Video.published_at.desc())
            
            if limit:
                query = query.limit(limit)
            
            result = await session.execute(query)
            videos = result.scalars().all()
            
            if not videos:
                return {
                    "channel_id": channel_id,
                    "videos_processed": 0,
                    "message": "No unprocessed videos found"
                }
            
            logger.info(f"Starting filtering for {len(videos)} videos from channel {channel_id}")
            
            # Filter videos in batches
            filtering_results = await self.filter_videos_batch(videos)
            
            # Update video statuses in database
            valuable_videos = []
            filtered_out_videos = []
            
            for video, result in zip(videos, filtering_results):
                if result.get("is_valuable"):
                    video.status = VideoStatus.QUEUED
                    video.valuable_score = result["confidence_score"]
                    video.valuable_reason = result.get("reasoning", "")
                    if not video.processing_metadata:
                        video.processing_metadata = {}
                    video.processing_metadata["filtering_result"] = result
                    valuable_videos.append(video)
                else:
                    video.status = VideoStatus.FILTERED_OUT
                    video.valuable_score = result["confidence_score"]
                    video.valuable_reason = result.get("reasoning", "")
                    if not video.processing_metadata:
                        video.processing_metadata = {}
                    video.processing_metadata["filtering_result"] = result
                    filtered_out_videos.append(video)
            
            # Commit changes
            await session.commit()
            
            # Prepare summary
            summary = {
                "channel_id": channel_id,
                "videos_processed": len(videos),
                "valuable_videos": len(valuable_videos),
                "filtered_out": len(filtered_out_videos),
                "average_confidence": sum(r["confidence_score"] for r in filtering_results) / len(filtering_results),
                "total_cost": sum(r.get("processing_cost", 0) for r in filtering_results),
                "valuable_video_ids": [str(v.id) for v in valuable_videos]
            }
            
            logger.info("Filtering complete for channel",
                       channel_id=channel_id,
                       total=len(videos),
                       valuable=len(valuable_videos),
                       filtered_out=len(filtered_out_videos))
            
            return summary
    
    async def reprocess_with_better_model(
        self,
        video_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Reprocess videos with a higher quality model for better accuracy.
        
        Args:
            video_ids: List of video IDs to reprocess
            
        Returns:
            Updated filtering results
        """
        async with get_db_context() as session:
            # Get videos
            result = await session.execute(
                select(Video).where(Video.id.in_(video_ids))
            )
            videos = result.scalars().all()
            
            # Reprocess with better model
            results = await self.filter_videos_batch(
                videos,
                use_fast_model=False  # Use better model
            )
            
            # Update database with new results
            for video, result in zip(videos, results):
                video.valuable_score = result["confidence_score"]
                video.valuable_reason = result.get("reasoning", "")
                
                # Update processing metadata
                if not video.processing_metadata:
                    video.processing_metadata = {}
                video.processing_metadata["reprocessing_result"] = result
                
                if result.get("is_valuable") and video.status == VideoStatus.FILTERED_OUT:
                    video.status = VideoStatus.QUEUED
                    logger.info("Video recategorized as valuable after reprocessing",
                               video_id=video.id)
            
            await session.commit()
            
            return results
    
    def _update_metrics(self, result: Dict[str, Any]):
        """Update internal metrics."""
        self.metrics["total_processed"] += 1
        
        if result.get("is_valuable"):
            self.metrics["filtered_in"] += 1
        else:
            self.metrics["filtered_out"] += 1
        
        # Update running average confidence
        n = self.metrics["total_processed"]
        prev_avg = self.metrics["average_confidence"]
        self.metrics["average_confidence"] = (prev_avg * (n - 1) + result["confidence_score"]) / n
        
        # Add processing time
        self.metrics["processing_time"] += result.get("processing_time", 0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current filtering metrics."""
        return self.metrics.copy()
    
    async def close(self):
        """Cleanup resources."""
        await self.llm_manager.close()

class FilteringFeedback:
    """
    Manage feedback for improving filtering accuracy.
    
    This class tracks manual corrections to AI filtering decisions
    and can be used to improve prompts or fine-tune models.
    """
    
    def __init__(self):
        """Initialize feedback system."""
        self.feedback_data = []
    
    async def record_feedback(
        self,
        video_id: str,
        ai_decision: bool,
        human_decision: bool,
        reason: Optional[str] = None
    ):
        """
        Record feedback on filtering decision.
        
        Args:
            video_id: Video that was evaluated
            ai_decision: What the AI decided (valuable or not)
            human_decision: What the human decided
            reason: Optional reason for disagreement
        """
        feedback = {
            "video_id": video_id,
            "ai_decision": ai_decision,
            "human_decision": human_decision,
            "agreement": ai_decision == human_decision,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.feedback_data.append(feedback)
        
        # Store in database
        async with get_db_context() as session:
            video = await session.get(Video, video_id)
            if video:
                if not video.processing_metadata:
                    video.processing_metadata = {}
                video.processing_metadata["human_feedback"] = feedback
                await session.commit()
        
        logger.info("Feedback recorded for video",
                   video_id=video_id,
                   agreement=feedback["agreement"])
    
    def calculate_accuracy(self) -> Dict[str, float]:
        """Calculate filtering accuracy based on feedback."""
        if not self.feedback_data:
            return {"accuracy": 0.0, "sample_size": 0}
        
        agreements = sum(1 for f in self.feedback_data if f["agreement"])
        accuracy = agreements / len(self.feedback_data)
        
        # Calculate precision and recall
        true_positives = sum(1 for f in self.feedback_data 
                            if f["ai_decision"] and f["human_decision"])
        false_positives = sum(1 for f in self.feedback_data
                             if f["ai_decision"] and not f["human_decision"])
        false_negatives = sum(1 for f in self.feedback_data
                             if not f["ai_decision"] and f["human_decision"])
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "sample_size": len(self.feedback_data),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }