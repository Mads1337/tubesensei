"""
Topic Filter Service

Standalone service for AI-based topic relevance filtering.
Can be used independently of the full campaign system.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.video import Video
from app.ai.llm_manager import LLMManager, ModelType
from app.utils.rate_limiter import RateLimiter

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


@dataclass
class FilterResult:
    """Result of topic filtering for a single video."""
    video_id: UUID
    is_relevant: bool
    relevance_score: float
    reasoning: str
    matched_keywords: List[str]
    topic_alignment: str
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None


class TopicFilterService:
    """
    Service for filtering videos by topic relevance using AI.

    This is a standalone service that can be used independently
    of the full campaign system. It analyzes video title and
    description to determine topic relevance.
    """

    def __init__(
        self,
        db: AsyncSession,
        llm_manager: Optional[LLMManager] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.db = db
        self.llm_manager = llm_manager or LLMManager()
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_minute=60)

    async def filter_video(
        self,
        video: Video,
        topic: str,
        threshold: float = 0.7,
    ) -> FilterResult:
        """
        Filter a single video for topic relevance.

        Args:
            video: Video to filter
            topic: Topic to check relevance against
            threshold: Minimum relevance score to consider relevant

        Returns:
            FilterResult with relevance decision
        """
        try:
            # Get channel name
            channel_name = "Unknown"
            if video.channel:
                channel_name = video.channel.name

            # Build prompt
            prompt = TOPIC_FILTER_PROMPT.format(
                topic=topic,
                title=video.title or "Untitled",
                description=(video.description or "")[:500],
                channel_name=channel_name,
                duration_minutes=round((video.duration_seconds or 0) / 60, 1),
                view_count=video.view_count or 0,
            )

            # Apply rate limiting
            async with self.rate_limiter.acquire():
                # Call LLM
                response = await self.llm_manager.generate(
                    prompt=prompt,
                    model_type=ModelType.FAST,
                    max_tokens=300,
                )

            # Parse response
            content = response.get("content", "")
            parsed = self._parse_response(content)

            if parsed:
                is_relevant = parsed["is_relevant"] and parsed["relevance_score"] >= threshold

                return FilterResult(
                    video_id=video.id,
                    is_relevant=is_relevant,
                    relevance_score=parsed["relevance_score"],
                    reasoning=parsed["reasoning"],
                    matched_keywords=parsed.get("matched_keywords", []),
                    topic_alignment=parsed.get("topic_alignment", "unrelated"),
                    tokens_used=response.get("usage", {}).get("total_tokens", 0),
                    cost_usd=response.get("cost", 0.0),
                )
            else:
                return FilterResult(
                    video_id=video.id,
                    is_relevant=False,
                    relevance_score=0.0,
                    reasoning="Failed to parse LLM response",
                    matched_keywords=[],
                    topic_alignment="unrelated",
                    error="Parse error",
                )

        except Exception as e:
            logger.error(f"Error filtering video {video.id}: {e}")
            return FilterResult(
                video_id=video.id,
                is_relevant=False,
                relevance_score=0.0,
                reasoning=str(e),
                matched_keywords=[],
                topic_alignment="unrelated",
                error=str(e),
            )

    async def filter_video_by_id(
        self,
        video_id: UUID,
        topic: str,
        threshold: float = 0.7,
    ) -> FilterResult:
        """Filter a video by its ID."""
        result = await self.db.execute(
            select(Video).where(Video.id == video_id)
        )
        video = result.scalar_one_or_none()

        if not video:
            return FilterResult(
                video_id=video_id,
                is_relevant=False,
                relevance_score=0.0,
                reasoning="Video not found",
                matched_keywords=[],
                topic_alignment="unrelated",
                error="Video not found",
            )

        return await self.filter_video(video, topic, threshold)

    async def filter_batch(
        self,
        videos: List[Video],
        topic: str,
        threshold: float = 0.7,
        batch_size: int = 10,
    ) -> List[FilterResult]:
        """
        Filter multiple videos for topic relevance.

        Args:
            videos: List of videos to filter
            topic: Topic to check relevance against
            threshold: Minimum relevance score
            batch_size: Number of videos to process concurrently

        Returns:
            List of FilterResults
        """
        import asyncio

        results = []

        # Process in batches
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]

            # Process batch concurrently
            batch_results = await asyncio.gather(*[
                self.filter_video(video, topic, threshold)
                for video in batch
            ])

            results.extend(batch_results)

            # Small delay between batches
            if i + batch_size < len(videos):
                await asyncio.sleep(0.5)

        return results

    async def filter_batch_by_ids(
        self,
        video_ids: List[UUID],
        topic: str,
        threshold: float = 0.7,
        batch_size: int = 10,
    ) -> List[FilterResult]:
        """Filter multiple videos by their IDs."""
        # Fetch all videos
        result = await self.db.execute(
            select(Video).where(Video.id.in_(video_ids))
        )
        videos = list(result.scalars().all())

        # Create a map for ordering
        video_map = {v.id: v for v in videos}

        # Filter in order
        ordered_videos = [video_map[vid] for vid in video_ids if vid in video_map]

        return await self.filter_batch(ordered_videos, topic, threshold, batch_size)

    def _parse_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM response into a filter result."""
        try:
            content = content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

            # Parse JSON
            result = json.loads(content)

            # Validate required fields
            if "is_relevant" not in result or "relevance_score" not in result:
                return None

            # Normalize values
            result["is_relevant"] = bool(result["is_relevant"])
            result["relevance_score"] = float(result["relevance_score"])
            result["relevance_score"] = max(0.0, min(1.0, result["relevance_score"]))
            result["reasoning"] = str(result.get("reasoning", ""))[:500]
            result["matched_keywords"] = result.get("matched_keywords", [])
            result["topic_alignment"] = result.get("topic_alignment", "unrelated")

            return result

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    # Utility methods

    async def get_filter_stats(
        self,
        results: List[FilterResult],
    ) -> Dict[str, Any]:
        """Get statistics from filter results."""
        if not results:
            return {
                "total": 0,
                "relevant": 0,
                "filtered_out": 0,
                "errors": 0,
                "acceptance_rate": 0.0,
                "avg_relevance_score": 0.0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
            }

        relevant = [r for r in results if r.is_relevant]
        errors = [r for r in results if r.error]
        scores = [r.relevance_score for r in results if r.relevance_score > 0]

        return {
            "total": len(results),
            "relevant": len(relevant),
            "filtered_out": len(results) - len(relevant) - len(errors),
            "errors": len(errors),
            "acceptance_rate": (len(relevant) / len(results)) * 100 if results else 0.0,
            "avg_relevance_score": sum(scores) / len(scores) if scores else 0.0,
            "total_tokens": sum(r.tokens_used for r in results),
            "total_cost_usd": sum(r.cost_usd for r in results),
        }
