"""
Idea Extraction Agent

Extracts business ideas from video transcripts using LLM analysis.
Runs in a controlled loop similar to TranscriptionAgent to allow
pausing, resuming, and progress tracking.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional

from sqlalchemy import select

from app.agents.base import BaseAgent, AgentResult, AgentEventType
from app.models.agent_run import AgentType
from app.models.campaign_video import CampaignVideo
from app.models.transcript import Transcript
from app.models.video import Video
from app.models.idea import Idea, IdeaStatus, IdeaPriority
from app.ai.llm_manager import LLMManager, ModelType
from app.ai.prompt_templates import PromptTemplates, PromptType
from app.ai.response_parser import ResponseParser

logger = logging.getLogger(__name__)

# Maximum words before we chunk a transcript
MAX_TRANSCRIPT_WORDS = 8000


class IdeaExtractionAgent(BaseAgent):
    """
    Agent responsible for extracting business ideas from video transcripts.

    Processing loop:
    1. Find relevant videos with transcripts but no ideas extracted
    2. For each video: load transcript, call LLM, parse ideas, save to DB
    3. Update progress and handle pause/resume/cancel signals
    """

    agent_type = AgentType.IDEA_EXTRACTION

    def __init__(self, context, llm_manager: Optional[LLMManager] = None):
        super().__init__(context)
        self.llm_manager = llm_manager or LLMManager()

    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Run the idea extraction process.

        Args:
            input_data:
                - batch_size: Number of videos to process per iteration (default: 5)

        Returns:
            AgentResult
        """
        batch_size = input_data.get("batch_size", 5)

        logger.info(f"IdeaExtractionAgent: Starting for campaign {self.campaign_id}")

        # Send initial heartbeat
        self.context.campaign.heartbeat()
        await self.db.commit()

        processed_count = 0
        success_count = 0
        failed_count = 0
        total_ideas = 0

        while True:
            # Check for cancellation/pause
            if await self.check_should_stop():
                logger.info("IdeaExtractionAgent: Stopping due to signal")
                break

            # Get next batch
            videos = await self._get_pending_videos(limit=batch_size)

            if not videos:
                logger.info("IdeaExtractionAgent: No more pending videos")
                break

            logger.info(f"IdeaExtractionAgent: Processing batch of {len(videos)} videos")

            for cv in videos:
                if await self.check_should_stop():
                    break

                try:
                    ideas_count = await self._process_video(cv)
                    processed_count += 1
                    success_count += 1
                    total_ideas += ideas_count
                except Exception as e:
                    logger.error(f"Error extracting ideas for video {cv.video_id}: {e}")
                    processed_count += 1
                    failed_count += 1
                    # Mark as extracted to prevent retry loop
                    cv.mark_ideas_extracted()
                    self.add_error(f"Video {cv.video_id}: {str(e)[:200]}")

            # Update progress
            total_pending = await self._count_pending_videos()
            total_with_transcripts = await self._count_total_with_transcripts()
            progress_percent = 0
            if total_with_transcripts > 0:
                completed = total_with_transcripts - total_pending
                progress_percent = (completed / total_with_transcripts) * 100

            await self.send_event(
                event_type=AgentEventType.PROGRESS,
                message=f"Extracted {total_ideas} ideas from {processed_count} videos ({success_count} successful)",
                progress=progress_percent,
                data={
                    "processed_count": processed_count,
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "total_ideas": total_ideas,
                    "remaining": total_pending,
                }
            )

            # Heartbeat
            self.context.campaign.heartbeat()
            await self.db.commit()

            await asyncio.sleep(0.1)

        return self._build_result(
            success=True,
            data={
                "processed_count": processed_count,
                "success_count": success_count,
                "failed_count": failed_count,
                "total_ideas": total_ideas,
            }
        )

    async def _process_video(self, cv: CampaignVideo) -> int:
        """
        Extract ideas from a single video's transcript.

        Returns:
            Number of ideas extracted
        """
        # Load transcript
        transcript_result = await self.db.execute(
            select(Transcript)
            .where(Transcript.video_id == cv.video_id)
            .order_by(Transcript.created_at.desc())
            .limit(1)
        )
        transcript = transcript_result.scalar_one_or_none()

        if not transcript or not transcript.content:
            logger.warning(f"No transcript content for video {cv.video_id}")
            cv.mark_ideas_extracted()
            return 0

        # Load video for metadata
        video = await self.db.get(Video, cv.video_id)
        if not video:
            cv.mark_ideas_extracted()
            return 0

        # Get transcript text
        transcript_text = transcript.content
        word_count = len(transcript_text.split())

        # Chunk if needed
        if word_count > MAX_TRANSCRIPT_WORDS:
            chunks = self._chunk_transcript(transcript_text, MAX_TRANSCRIPT_WORDS)
        else:
            chunks = [transcript_text]

        all_ideas = []
        for chunk in chunks:
            ideas = await self._extract_ideas_from_text(
                transcript_text=chunk,
                video=video,
            )
            all_ideas.extend(ideas)

        # Save ideas to DB
        saved_count = 0
        for idea_data in all_ideas:
            try:
                idea = Idea(
                    video_id=video.id,
                    title=idea_data.title,
                    description=idea_data.description,
                    category=idea_data.category,
                    status=IdeaStatus.EXTRACTED,
                    priority=IdeaPriority.MEDIUM,
                    confidence_score=idea_data.confidence,
                    complexity_score=idea_data.complexity_score,
                    target_audience=idea_data.target_market,
                    source_context=idea_data.source_context,
                    extraction_metadata={
                        "campaign_id": str(self.campaign_id),
                        "value_proposition": idea_data.value_proposition,
                        "transcript_word_count": word_count,
                    },
                )
                self.db.add(idea)
                saved_count += 1
            except Exception as e:
                logger.warning(f"Failed to save idea '{idea_data.title}': {e}")

        # Mark video as processed
        cv.mark_ideas_extracted()
        await self.db.flush()

        self.increment_processed()
        self.increment_produced(saved_count)

        logger.info(f"Extracted {saved_count} ideas from video {video.id} ({video.title[:50]})")
        return saved_count

    async def _extract_ideas_from_text(
        self,
        transcript_text: str,
        video: Video,
    ) -> list:
        """Call LLM to extract ideas from transcript text."""
        system_prompt, user_prompt = PromptTemplates.get_prompt(
            PromptType.IDEA_EXTRACTION,
            variables={
                "transcript": transcript_text,
                "title": video.title or "Unknown",
                "channel_name": video.channel.name if video.channel else "Unknown",
                "duration_minutes": round((video.duration_seconds or 0) / 60, 1),
            },
        )

        try:
            response = await self.llm_manager.generate(
                prompt=user_prompt,
                model_type=ModelType.BALANCED,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=4000,
            )

            self.increment_llm_calls(
                count=1,
                tokens=response.get("usage", {}).get("total_tokens", 0),
                cost=float(response.get("cost", 0)),
            )

            content = response.get("content", "")
            ideas = ResponseParser.parse_idea_extraction_response(content)
            return ideas

        except Exception as e:
            logger.error(f"LLM call failed for idea extraction: {e}")
            self.add_error(f"LLM error: {str(e)[:200]}")
            return []

    async def _get_pending_videos(self, limit: int) -> List[CampaignVideo]:
        """Get batch of videos with transcripts but no ideas extracted."""
        query = (
            select(CampaignVideo)
            .where(
                CampaignVideo.campaign_id == self.campaign_id,
                CampaignVideo.is_topic_relevant == True,
                CampaignVideo.transcript_extracted == True,
                CampaignVideo.ideas_extracted == False,
            )
            .limit(limit)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def _count_pending_videos(self) -> int:
        """Count videos still pending idea extraction."""
        from sqlalchemy import func
        query = (
            select(func.count())
            .where(
                CampaignVideo.campaign_id == self.campaign_id,
                CampaignVideo.is_topic_relevant == True,
                CampaignVideo.transcript_extracted == True,
                CampaignVideo.ideas_extracted == False,
            )
        )
        result = await self.db.execute(query)
        return result.scalar() or 0

    async def _count_total_with_transcripts(self) -> int:
        """Count total videos with transcripts (for progress calculation)."""
        from sqlalchemy import func
        query = (
            select(func.count())
            .where(
                CampaignVideo.campaign_id == self.campaign_id,
                CampaignVideo.is_topic_relevant == True,
                CampaignVideo.transcript_extracted == True,
            )
        )
        result = await self.db.execute(query)
        return result.scalar() or 0

    @staticmethod
    def _chunk_transcript(text: str, max_words: int) -> List[str]:
        """Split transcript into chunks of approximately max_words each."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        return chunks
