"""
Topic Discovery Agents Module

This module contains the agent framework for topic-based video discovery.
Agents work together to discover, filter, and process YouTube videos
related to user-specified topics.

Agent Types:
- Coordinator: Orchestrates the entire discovery pipeline
- Search: Finds initial videos via YouTube search
- ChannelExpansion: Discovers more videos from found channels
- TopicFilter: AI-based relevance filtering
- SimilarVideos: Finds related videos via YouTube API
"""

from app.agents.base import (
    BaseAgent,
    AgentResult,
    AgentContext,
    AgentEventType,
    AgentEvent,
)
from app.agents.search_agent import SearchAgent
from app.agents.channel_expansion_agent import ChannelExpansionAgent
from app.agents.topic_filter_agent import TopicFilterAgent
from app.agents.similar_videos_agent import SimilarVideosAgent
from app.agents.coordinator import CoordinatorAgent

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentResult",
    "AgentContext",
    "AgentEventType",
    "AgentEvent",
    # Agents
    "SearchAgent",
    "ChannelExpansionAgent",
    "TopicFilterAgent",
    "SimilarVideosAgent",
    "CoordinatorAgent",
]
