"""
Topic Filter Agent

Uses AI to determine if videos are relevant to the campaign topic.
Analyzes video title and description (fast, cheap filtering).
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import select, update

from app.agents.base import BaseAgent, AgentResult, AgentContext, AgentEventType
from app.models.agent_run import AgentType
from app.models.video import Video
from app.models.campaign_video import CampaignVideo
from app.models.topic_campaign import TopicCampaign
from app.ai.llm_manager import LLMManager, ModelType

logger = logging.getLogger(__name__)


# Topic filter prompt template
TOPIC_FILTER_PROMPT = """You are an expert content analyst. Determine if this YouTube video is relevant to the given topic.

TOPIC: {topic}

VIDEO DETAILS:
- Title: {title}
- Description: {description}
- Channel: {channel_name}
- Duration: {duration_minutes} minutes
- Views: {view_count}

Analyze the video's relevance to the topic. Consider:
1. Direct topic match in title/description
2. Related concepts or subtopics
3. Target audience alignment
4. Content type (tutorial, review, case study, discussion)

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "is_relevant": true or false,
    "relevance_score": 0.0 to 1.0,
    "reasoning": "Brief explanation (max 100 words)",
    "matched_keywords": ["keyword1", "keyword2"],
    "topic_alignment": "exact" or "related" or "tangential" or "unrelated"
}}"""


class TopicFilterAgent(BaseAgent):
    """
    Agent that filters videos by topic relevance using AI.

    Uses video title and description only (no transcript) for fast filtering.

    Input:
        video_ids: List[UUID] - Video IDs to filter
        topic: str - The topic to filter by (optional, uses campaign topic)
        batch_size: int - Batch size for processing (default: 10)

    Output:
        relevant_ids: List[UUID] - Videos that passed the filter
        filtered_ids: List[UUID] - Videos that were filtered out
        total_processed: int - Total videos processed
        acceptance_rate: float - Percentage of videos that passed
    """

    agent_type = AgentType.TOPIC_FILTER

    def __init__(self, context: AgentContext, llm_manager: Optional[LLMManager] = None):
        super().__init__(context)
        self.llm_manager = llm_manager

    async def run(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Filter videos by topic relevance.

        Args:
            input_data: Contains 'video_ids', optional 'topic' and 'batch_size'

        Returns:
            AgentResult with relevant and filtered video IDs
        """
        video_ids = input_data.get("video_ids", [])
        if isinstance(video_ids[0], str) if video_ids else False:
            video_ids = [UUID(vid) for vid in video_ids]

        topic = input_data.get("topic", self.context.campaign.topic)
        batch_size = input_data.get("batch_size", 10)

        logger.info(f"TopicFilterAgent: Filtering {len(video_ids)} videos for topic '{topic[:50]}...'")

        relevant_ids: List[UUID] = []
        filtered_ids: List[UUID] = []

        if not video_ids:
            return self._build_result(success=True, data={
                "relevant_ids": [],
                "filtered_ids": [],
                "total_processed": 0,
                "acceptance_rate": 0.0,
            })

        try:
            # Initialize LLM manager if not provided
            llm = self.llm_manager or LLMManager()

            # Process in batches
            total_batches = (len(video_ids) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(video_ids))
                batch_video_ids = video_ids[start_idx:end_idx]

                # Check if we should stop
                if await self.check_should_stop():
                    logger.info("TopicFilterAgent: Stopping due to limit reached or cancellation")
                    break

                self.update_progress(
                    ((batch_idx + 1) / total_batches) * 100,
                    current_item=f"Batch {batch_idx + 1}/{total_batches}"
                )

                # Process batch
                batch_relevant, batch_filtered = await self._process_batch(
                    batch_video_ids, topic, llm
                )

                relevant_ids.extend(batch_relevant)
                filtered_ids.extend(batch_filtered)

                # Small delay between batches
                if batch_idx < total_batches - 1:
                    await asyncio.sleep(0.5)

            # Commit all changes
            await self.db.commit()

            # Update campaign stats
            self.context.campaign.increment_relevant(len(relevant_ids))
            self.context.campaign.increment_filtered(len(filtered_ids))

            total_processed = len(relevant_ids) + len(filtered_ids)
            acceptance_rate = (len(relevant_ids) / total_processed * 100) if total_processed > 0 else 0.0

            logger.info(
                f"TopicFilterAgent: {len(relevant_ids)} relevant, "
                f"{len(filtered_ids)} filtered out "
                f"({acceptance_rate:.1f}% acceptance rate)"
            )

            return self._build_result(success=True, data={
                "relevant_ids": [str(vid) for vid in relevant_ids],
                "filtered_ids": [str(vid) for vid in filtered_ids],
                "total_processed": total_processed,
                "acceptance_rate": acceptance_rate,
            })

        except Exception as e:
            logger.exception(f"TopicFilterAgent failed: {e}")
            return self._build_result(success=False, data={
                "relevant_ids": [str(vid) for vid in relevant_ids],
                "filtered_ids": [str(vid) for vid in filtered_ids],
                "total_processed": len(relevant_ids) + len(filtered_ids),
                "error": str(e),
            })

    async def _process_batch(
        self,
        video_ids: List[UUID],
        topic: str,
        llm: LLMManager
    ) -> Tuple[List[UUID], List[UUID]]:
        """
        Process a batch of videos.

        Returns:
            Tuple of (relevant_ids, filtered_ids)
        """
        relevant_ids: List[UUID] = []
        filtered_ids: List[UUID] = []

        # Fetch videos with their campaign video records
        for video_id in video_ids:
            try:
                # Get video and campaign video
                video_result = await self.db.execute(
                    select(Video).where(Video.id == video_id)
                )
                video = video_result.scalar_one_or_none()

                cv_result = await self.db.execute(
                    select(CampaignVideo).where(
                        CampaignVideo.campaign_id == self.campaign_id,
                        CampaignVideo.video_id == video_id
                    )
                )
                campaign_video = cv_result.scalar_one_or_none()

                if not video or not campaign_video:
                    logger.warning(f"Video or CampaignVideo not found for {video_id}")
                    continue

                # Skip if already filtered
                if campaign_video.is_topic_relevant is not None:
                    if campaign_video.is_topic_relevant:
                        relevant_ids.append(video_id)
                    else:
                        filtered_ids.append(video_id)
                    self.increment_processed()
                    continue

                # Get channel name
                channel_name = "Unknown"
                if video.channel:
                    channel_name = video.channel.name

                # Build prompt
                prompt = TOPIC_FILTER_PROMPT.format(
                    topic=topic,
                    title=video.title or "Untitled",
                    description=(video.description or "")[:500],  # Limit description
                    channel_name=channel_name,
                    duration_minutes=round((video.duration_seconds or 0) / 60, 1),
                    view_count=video.view_count or 0,
                )

                # Apply rate limiting
                async with self.context.llm_rate_limiter.acquire():
                    # Call LLM
                    response = await llm.generate(
                        prompt=prompt,
                        model_type=ModelType.FAST,
                        max_tokens=300,
                    )
                    self.increment_llm_calls(
                        tokens=response.get("usage", {}).get("total_tokens", 0),
                        cost=response.get("cost", 0.0)
                    )

                # Parse response
                filter_result = self._parse_filter_response(response.get("content", ""))

                if filter_result:
                    is_relevant = filter_result["is_relevant"]
                    relevance_score = filter_result["relevance_score"]

                    # Apply threshold
                    passes_threshold = is_relevant and relevance_score >= self.filter_threshold

                    if passes_threshold:
                        campaign_video.mark_relevant(
                            score=relevance_score,
                            reasoning=filter_result["reasoning"],
                            keywords=filter_result.get("matched_keywords"),
                            alignment=filter_result.get("topic_alignment"),
                        )
                        relevant_ids.append(video_id)
                        self.increment_produced()

                        self._emit_event(
                            AgentEventType.ITEM_PROCESSED,
                            data={
                                "video_id": str(video_id),
                                "relevant": True,
                                "score": relevance_score,
                            },
                            message=f"Video relevant (score: {relevance_score:.2f})"
                        )
                    else:
                        campaign_video.mark_irrelevant(
                            score=relevance_score,
                            reasoning=filter_result["reasoning"],
                            alignment=filter_result.get("topic_alignment"),
                        )
                        filtered_ids.append(video_id)

                        self._emit_event(
                            AgentEventType.ITEM_PROCESSED,
                            data={
                                "video_id": str(video_id),
                                "relevant": False,
                                "score": relevance_score,
                            },
                            message=f"Video filtered out (score: {relevance_score:.2f})"
                        )
                else:
                    # Failed to parse - mark as error but don't filter out
                    self.add_error(f"Failed to parse filter response for video {video_id}")

                self.increment_processed()
                await self.db.flush()

            except Exception as e:
                error_msg = f"Error filtering video {video_id}: {e}"
                logger.error(error_msg)
                self.add_error(error_msg)
                self.increment_processed()

        return relevant_ids, filtered_ids

    def _parse_filter_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM response into a filter result."""
        try:
            # Try to extract JSON from response
            content = content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

            # Parse JSON
            result = json.loads(content)

            # Validate required fields
            if "is_relevant" not in result or "relevance_score" not in result:
                logger.warning(f"Missing required fields in filter response: {content}")
                return None

            # Normalize values
            result["is_relevant"] = bool(result["is_relevant"])
            result["relevance_score"] = float(result["relevance_score"])
            result["relevance_score"] = max(0.0, min(1.0, result["relevance_score"]))
            result["reasoning"] = str(result.get("reasoning", ""))[:500]
            result["matched_keywords"] = result.get("matched_keywords", [])
            result["topic_alignment"] = result.get("topic_alignment", "unrelated")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse filter JSON: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing filter response: {e}")
            return None
