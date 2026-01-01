"""
Transcription Agent

Manages the bulk transcription of videos discovered in a campaign.
Runs in a controlled loop to allow pausing, resuming, and progress tracking.
"""
import logging
from typing import Dict, Any, List
from uuid import UUID

from sqlalchemy import select, and_

from app.agents.base import BaseAgent, AgentResult, AgentEventType
from app.models.agent_run import AgentType
from app.models.campaign_video import CampaignVideo
from app.models.topic_campaign import TopicCampaign
from app.services.transcript_processor import TranscriptProcessor

logger = logging.getLogger(__name__)


class TranscriptionAgent(BaseAgent):
    """
    Agent responsible for extracting transcripts for relevant videos in a campaign.
    
    Processing loop:
    1. Find relevant videos in campaign without transcripts
    2. Process them in batches
    3. Update progress and handle pause/resume/cancel signals
    """

    agent_type = AgentType.TRANSCRIPTION

    def __init__(self, context):
        super().__init__(context)
        self.processor = TranscriptProcessor(session=self.db)

    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Run the transcription process.

        Args:
            input_data:
                - batch_size: Number of videos to process in parallel (default: 5)
        
        Returns:
            AgentResult
        """
        batch_size = input_data.get("batch_size", 5)

        logger.info(f"TranscriptionAgent: Starting execution for campaign {self.campaign_id}")

        # Send initial heartbeat
        self.context.campaign.heartbeat()
        await self.db.commit()

        processed_count = 0
        success_count = 0
        failed_count = 0
        
        # Main processing loop
        while True:
            # Check for cancellation/pause
            if await self.check_should_stop():
                logger.info("TranscriptionAgent: Stopping due to signal")
                break
                
            # Get next batch of videos
            videos = await self._get_pending_videos(limit=batch_size)
            
            if not videos:
                logger.info("TranscriptionAgent: No more pending videos")
                break
                
            logger.info(f"TranscriptionAgent: Processing batch of {len(videos)} videos")
            
            # Process batch
            results = await self._process_batch(videos)
            
            # Update counts
            batch_processed = len(results)
            batch_success = sum(1 for r in results if r["success"])
            batch_failed = batch_processed - batch_success
            
            processed_count += batch_processed
            success_count += batch_success
            failed_count += batch_failed

            # Increment campaign counter (for progress tracking)
            self.context.campaign.increment_transcripts(batch_success)
            self.context.campaign.increment_api_calls(batch_processed) # 1 call per video roughly
            
            # Send progress event
            total_pending = await self._count_pending_videos()
            total_relevant = self.context.campaign.total_videos_relevant
            progress_percent = 0
            if total_relevant > 0:
                completed = self.context.campaign.total_transcripts_extracted
                progress_percent = (completed / total_relevant) * 100

            await self.send_event(
                event_type=AgentEventType.AGENT_PROGRESS,
                message=f"Processed {processed_count} videos ({success_count} successful)",
                progress=progress_percent,
                data={
                    "processed_count": processed_count,
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "remaining": total_pending
                }
            )
            
            # Send heartbeat to indicate worker is still alive
            self.context.campaign.heartbeat()

            # Commit changes
            await self.db.commit()

            # Small delay to prevent tight loop if database is fast
            import asyncio
            await asyncio.sleep(0.1)

        return self._build_result(
            success=True,
            data={
                "processed_count": processed_count,
                "success_count": success_count,
                "failed_count": failed_count
            }
        )

    async def _count_pending_videos(self) -> int:
        """Count total videos pending transcription."""
        from sqlalchemy import func
        query = (
            select(func.count())
            .where(
                CampaignVideo.campaign_id == self.campaign_id,
                CampaignVideo.is_topic_relevant == True,
                CampaignVideo.transcript_extracted == False
            )
        )
        result = await self.db.execute(query)
        return result.scalar()

    async def _process_batch(self, videos: List[CampaignVideo]) -> List[Dict[str, Any]]:
        """Process a batch of videos."""
        results = []
        
        # We process sequentially within the batch for now because the 
        # TranscriptProcessor isn't fully designed for external concurrency 
        # (it manages its own session if not provided, but here we share session)
        # To be safe with the shared session, we do it one by one or we'd need
        # separate sessions for parallel execution.
        # Given we are in an async agent, one-by-one is safer for the shared db transaction.
        
        for vid in videos:
            try:
                # Mark as attempting (optional, but good for debugging)
                
                # Extract
                transcript = await self.processor.extract_transcript(
                    video_id=vid.video_id,
                    force_refresh=False,
                    save_to_db=True
                )
                
                success = transcript is not None
                
                # Update CampaignVideo status
                vid.transcript_extracted = success
                # If failed, we might want to track that so we don't retry forever.
                # Currently CampaignVideo only has boolean.
                # Ideally we'd have a 'transcription_status' column.
                # For now, if it fails, we assume it won't work and leave transcript_extracted=False
                # BUT this means we'll retry it forever in the loop.
                # FIX: We need a way to mark failure.
                # Since we can't change schema right now without Alebmic, 
                # we will trust TranscriptProcessor to mark the VIDEO status as FAILED or SKIPPED
                # If the underlying Video is Failed/Skipped, we should probably not try it again?
                
                # Let's check the video status.
                # The video object is loaded lazily or we can access via vid.video_id
                
                results.append({
                    "video_id": str(vid.video_id),
                    "success": success
                })
                
                # If failed, we need to mark it so we don't pick it up again in _get_pending_videos
                # But wait, _get_pending_videos filters by transcript_extracted == False
                # If it fails, it remains False.
                # We need to filter out failed videos in _get_pending_videos.
                # The TranscriptProcessor updates the Video table status.
                # So we should join with Video table in _get_pending_videos.
                
            except Exception as e:
                logger.error(f"Error processing video {vid.video_id}: {e}")
                results.append({
                    "video_id": str(vid.video_id),
                    "success": False,
                    "error": str(e)
                })

        return results

    async def _get_pending_videos(self, limit: int) -> List[CampaignVideo]:
        """Get batch of relevant videos pending transcription."""
        from app.models.video import Video, VideoStatus
        
        query = (
            select(CampaignVideo)
            .join(Video, CampaignVideo.video_id == Video.id)
            .where(
                CampaignVideo.campaign_id == self.campaign_id,
                CampaignVideo.is_topic_relevant == True,
                CampaignVideo.transcript_extracted == False,
                # Only pick videos that are not in a terminal failed state for transcription
                # TranscriptProcessor marks them as SKIPPED or FAILED if caption missing.
                # We should exclude those.
                Video.status.notin_([VideoStatus.FAILED, VideoStatus.SKIPPED])
            )
            .limit(limit)
        )
        result = await self.db.execute(query)
        return result.scalars().all()
