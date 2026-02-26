"""
Unit tests for the TopicFilterAgent.

Tests relevance classification from mocked LLM response,
JSON parsing of LLM output, and the _parse_filter_response method.
"""
import json
import pytest
from unittest.mock import MagicMock

from app.agents.topic_filter_agent import TopicFilterAgent


def _make_agent() -> TopicFilterAgent:
    """Create a minimal TopicFilterAgent with a mocked context."""
    context = MagicMock()
    context.campaign_id = MagicMock()
    context.campaign = MagicMock()
    context.campaign.topic = "how to make money with AI"
    context.campaign.total_videos_relevant = 0
    context.campaign.total_video_limit = 3000
    context.config = {}
    context.is_cancelled.return_value = False
    context.event_callback = None
    agent = TopicFilterAgent(context)
    return agent


class TestParseFilterResponse:
    """Tests for TopicFilterAgent._parse_filter_response."""

    def test_valid_json_relevant(self):
        """A valid JSON response marking video as relevant should parse correctly."""
        agent = _make_agent()
        content = json.dumps({
            "is_relevant": True,
            "relevance_score": 0.9,
            "reasoning": "This video is about making money with AI",
            "matched_keywords": ["AI", "money"],
            "topic_alignment": "exact",
        })
        result = agent._parse_filter_response(content)

        assert result is not None
        assert result["is_relevant"] is True
        assert result["relevance_score"] == pytest.approx(0.9)
        assert result["reasoning"] == "This video is about making money with AI"
        assert "AI" in result["matched_keywords"]

    def test_valid_json_not_relevant(self):
        """A valid JSON response marking video as not relevant should parse correctly."""
        agent = _make_agent()
        content = json.dumps({
            "is_relevant": False,
            "relevance_score": 0.1,
            "reasoning": "This video is about cooking",
            "matched_keywords": [],
            "topic_alignment": "unrelated",
        })
        result = agent._parse_filter_response(content)

        assert result is not None
        assert result["is_relevant"] is False
        assert result["relevance_score"] == pytest.approx(0.1)

    def test_json_with_markdown_code_fence(self):
        """Response wrapped in markdown code fences should be parsed correctly."""
        agent = _make_agent()
        raw_json = json.dumps({
            "is_relevant": True,
            "relevance_score": 0.8,
            "reasoning": "Relevant content",
            "matched_keywords": ["AI"],
            "topic_alignment": "related",
        })
        content = f"```json\n{raw_json}\n```"
        result = agent._parse_filter_response(content)

        assert result is not None
        assert result["is_relevant"] is True
        assert result["relevance_score"] == pytest.approx(0.8)

    def test_invalid_json_returns_none(self):
        """Invalid JSON should return None."""
        agent = _make_agent()
        result = agent._parse_filter_response("This is not JSON at all")
        assert result is None

    def test_missing_required_field_is_relevant_returns_none(self):
        """Response missing 'is_relevant' should return None."""
        agent = _make_agent()
        content = json.dumps({
            "relevance_score": 0.8,
            "reasoning": "Missing is_relevant field",
        })
        result = agent._parse_filter_response(content)
        assert result is None

    def test_missing_required_field_relevance_score_returns_none(self):
        """Response missing 'relevance_score' should return None."""
        agent = _make_agent()
        content = json.dumps({
            "is_relevant": True,
            "reasoning": "Missing relevance_score field",
        })
        result = agent._parse_filter_response(content)
        assert result is None

    def test_relevance_score_clamped_to_0_1(self):
        """relevance_score outside [0, 1] should be clamped."""
        agent = _make_agent()
        # Score above 1.0
        content = json.dumps({
            "is_relevant": True,
            "relevance_score": 1.5,
            "reasoning": "Very relevant",
        })
        result = agent._parse_filter_response(content)
        assert result is not None
        assert result["relevance_score"] == pytest.approx(1.0)

        # Score below 0.0
        content = json.dumps({
            "is_relevant": False,
            "relevance_score": -0.5,
            "reasoning": "Not relevant",
        })
        result = agent._parse_filter_response(content)
        assert result is not None
        assert result["relevance_score"] == pytest.approx(0.0)

    def test_is_relevant_coerced_to_bool(self):
        """is_relevant values should be coerced to bool."""
        agent = _make_agent()
        content = json.dumps({
            "is_relevant": 1,  # truthy int
            "relevance_score": 0.9,
            "reasoning": "Test",
        })
        result = agent._parse_filter_response(content)
        assert result is not None
        assert isinstance(result["is_relevant"], bool)
        assert result["is_relevant"] is True

    def test_reasoning_truncated_to_500_chars(self):
        """Reasoning string should be truncated to 500 characters."""
        agent = _make_agent()
        long_reasoning = "x" * 600
        content = json.dumps({
            "is_relevant": True,
            "relevance_score": 0.8,
            "reasoning": long_reasoning,
        })
        result = agent._parse_filter_response(content)
        assert result is not None
        assert len(result["reasoning"]) <= 500

    def test_default_values_for_optional_fields(self):
        """Missing optional fields should get default values."""
        agent = _make_agent()
        content = json.dumps({
            "is_relevant": True,
            "relevance_score": 0.7,
            "reasoning": "Relevant",
            # matched_keywords and topic_alignment are omitted
        })
        result = agent._parse_filter_response(content)
        assert result is not None
        assert result["matched_keywords"] == []
        assert result["topic_alignment"] == "unrelated"

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        agent = _make_agent()
        result = agent._parse_filter_response("")
        assert result is None

    def test_valid_topic_alignment_values(self):
        """Valid topic_alignment values should be preserved."""
        agent = _make_agent()
        for alignment in ["exact", "related", "tangential", "unrelated"]:
            content = json.dumps({
                "is_relevant": True,
                "relevance_score": 0.5,
                "reasoning": "Test",
                "topic_alignment": alignment,
            })
            result = agent._parse_filter_response(content)
            assert result is not None
            assert result["topic_alignment"] == alignment
