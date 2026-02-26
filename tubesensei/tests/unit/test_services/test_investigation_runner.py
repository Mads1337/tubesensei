"""
Unit tests for the InvestigationRunner service.

Tests the happy path and error path of run_investigation(),
prompt template substitution, and JSON parsing from LLM output.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone

from app.models.investigation_agent import InvestigationAgent
from app.models.investigation_run import InvestigationRun, InvestigationRunStatus
from app.models.idea import Idea, IdeaStatus, IdeaPriority
from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.services.investigation_runner import InvestigationRunner


async def _create_agent(db_session) -> InvestigationAgent:
    """Helper to create a minimal InvestigationAgent."""
    agent = InvestigationAgent(
        name="Test Runner Agent",
        system_prompt="You are an expert business analyst.",
        user_prompt_template=(
            "Analyze the following idea:\n"
            "Title: {idea_title}\n"
            "Description: {idea_description}\n"
            "Category: {idea_category}"
        ),
        config={"model_type": "balanced"},
        is_active=True,
    )
    db_session.add(agent)
    await db_session.flush()
    return agent


async def _create_idea(db_session) -> Idea:
    """Helper to create a minimal Idea with required parents."""
    channel = Channel(
        youtube_channel_id="UCrunner001",
        name="Runner Test Channel",
        status=ChannelStatus.ACTIVE,
    )
    db_session.add(channel)
    await db_session.flush()

    video = Video(
        youtube_video_id="runner001vid",
        channel_id=channel.id,
        title="Runner Test Video",
        status=VideoStatus.DISCOVERED,
        published_at=datetime.now(timezone.utc),
        tags=[],
        caption_languages=[],
        video_metadata={},
        processing_metadata={},
        has_captions=False,
    )
    db_session.add(video)
    await db_session.flush()

    idea = Idea(
        video_id=video.id,
        title="AI-Powered Customer Support Bot",
        description="An intelligent chatbot that automates customer support using GPT-4.",
        category="SaaS",
        status=IdeaStatus.EXTRACTED,
        priority=IdeaPriority.HIGH,
        confidence_score=0.85,
        tags=["AI", "automation"],
        technologies=["GPT-4", "Python"],
        potential_challenges=[{"text": "API costs"}],
        monetization_strategies=[{"text": "SaaS subscription"}],
        related_ideas=[],
        extraction_metadata={"campaign_id": str(uuid4())},
    )
    db_session.add(idea)
    await db_session.flush()
    return idea


class TestInvestigationRunnerHappyPath:
    """Tests for successful run_investigation() execution."""

    @pytest.mark.asyncio
    async def test_run_investigation_success(self, db_session):
        """Test that run_investigation creates a run and transitions through states."""
        agent = await _create_agent(db_session)
        idea = await _create_idea(db_session)
        await db_session.commit()

        mock_response = {
            "content": '{"market_analysis": "Strong demand", "recommendation": "proceed"}',
            "usage": {"total_tokens": 250},
            "cost": 0.005,
        }

        with patch("app.services.investigation_runner.LLMManager") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)

        assert run is not None
        assert run.status == InvestigationRunStatus.COMPLETED
        assert run.agent_id == agent.id
        assert run.idea_id == idea.id

    @pytest.mark.asyncio
    async def test_run_investigation_stores_result(self, db_session):
        """Test that the LLM result content is stored in the run."""
        agent = await _create_agent(db_session)
        idea = await _create_idea(db_session)
        await db_session.commit()

        result_text = '{"market_analysis": "Strong demand"}'
        mock_response = {
            "content": result_text,
            "usage": {"total_tokens": 100},
            "cost": 0.002,
        }

        with patch("app.services.investigation_runner.LLMManager") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)

        assert run.result == result_text
        assert run.tokens_used == 100
        assert run.estimated_cost_usd == pytest.approx(0.002)

    @pytest.mark.asyncio
    async def test_run_investigation_parses_json_result(self, db_session):
        """Test that JSON in result is parsed into result_structured."""
        agent = await _create_agent(db_session)
        idea = await _create_idea(db_session)
        await db_session.commit()

        structured_data = {"market_analysis": "Strong demand", "score": 8.5}
        mock_response = {
            "content": json.dumps(structured_data),
            "usage": {"total_tokens": 150},
            "cost": 0.003,
        }

        with patch("app.services.investigation_runner.LLMManager") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)

        assert run.result_structured == structured_data

    @pytest.mark.asyncio
    async def test_run_investigation_handles_markdown_fenced_json(self, db_session):
        """Test that markdown-fenced JSON is stripped before parsing."""
        agent = await _create_agent(db_session)
        idea = await _create_idea(db_session)
        await db_session.commit()

        structured_data = {"analysis": "Good idea"}
        fenced_content = f"```json\n{json.dumps(structured_data)}\n```"
        mock_response = {
            "content": fenced_content,
            "usage": {"total_tokens": 80},
            "cost": 0.001,
        }

        with patch("app.services.investigation_runner.LLMManager") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)

        assert run.result_structured == structured_data

    @pytest.mark.asyncio
    async def test_run_investigation_non_json_result_still_completes(self, db_session):
        """Non-JSON result should not prevent COMPLETED status."""
        agent = await _create_agent(db_session)
        idea = await _create_idea(db_session)
        await db_session.commit()

        mock_response = {
            "content": "This is a narrative analysis without JSON structure.",
            "usage": {"total_tokens": 200},
            "cost": 0.004,
        }

        with patch("app.services.investigation_runner.LLMManager") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)

        assert run.status == InvestigationRunStatus.COMPLETED
        assert run.result == "This is a narrative analysis without JSON structure."
        assert run.result_structured is None


class TestInvestigationRunnerErrorPath:
    """Tests for error handling in run_investigation()."""

    @pytest.mark.asyncio
    async def test_run_investigation_llm_failure_sets_failed_status(self, db_session):
        """When LLM raises an exception, the run should be marked as FAILED."""
        agent = await _create_agent(db_session)
        idea = await _create_idea(db_session)
        await db_session.commit()

        with patch("app.services.investigation_runner.LLMManager") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_llm.generate = AsyncMock(side_effect=RuntimeError("LLM API error"))
            mock_llm_class.return_value = mock_llm

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)

        assert run.status == InvestigationRunStatus.FAILED
        assert "LLM API error" in run.error_message

    @pytest.mark.asyncio
    async def test_run_investigation_agent_not_found_raises(self, db_session):
        """When agent_id doesn't exist, NotFoundException should be raised."""
        from app.core.exceptions import NotFoundException

        idea = await _create_idea(db_session)
        await db_session.commit()

        runner = InvestigationRunner(db_session)
        with pytest.raises(NotFoundException):
            await runner.run_investigation(uuid4(), idea.id)

    @pytest.mark.asyncio
    async def test_run_investigation_idea_not_found_raises(self, db_session):
        """When idea_id doesn't exist, NotFoundException should be raised."""
        from app.core.exceptions import NotFoundException

        agent = await _create_agent(db_session)
        await db_session.commit()

        runner = InvestigationRunner(db_session)
        with pytest.raises(NotFoundException):
            await runner.run_investigation(agent.id, uuid4())


class TestInvestigationRunnerPromptBuilding:
    """Tests for prompt template substitution."""

    def test_build_user_prompt_substitutes_idea_fields(self):
        """_build_user_prompt should substitute all {idea_*} placeholders."""
        template = (
            "Title: {idea_title}\n"
            "Description: {idea_description}\n"
            "Category: {idea_category}\n"
            "Tags: {idea_tags}\n"
            "Technologies: {idea_technologies}"
        )
        idea = Idea(
            title="Test Idea",
            description="A test description",
            category="SaaS",
            confidence_score=0.8,
            tags=["tag1", "tag2"],
            technologies=["Python", "FastAPI"],
            potential_challenges=[],
            monetization_strategies=[],
            related_ideas=[],
            extraction_metadata={},
        )
        result = InvestigationRunner._build_user_prompt(template, idea)

        assert "Test Idea" in result
        assert "A test description" in result
        assert "SaaS" in result
        assert "tag1" in result
        assert "tag2" in result
        assert "Python" in result
        assert "FastAPI" in result

    def test_build_user_prompt_handles_none_category(self):
        """_build_user_prompt should use 'Uncategorized' when category is None."""
        template = "Category: {idea_category}"
        idea = Idea(
            title="Test",
            description="desc",
            category=None,
            confidence_score=0.5,
            tags=[],
            technologies=[],
            potential_challenges=[],
            monetization_strategies=[],
            related_ideas=[],
            extraction_metadata={},
        )
        result = InvestigationRunner._build_user_prompt(template, idea)
        assert "Uncategorized" in result

    def test_build_user_prompt_handles_empty_tags(self):
        """_build_user_prompt should show 'None' when tags is empty."""
        template = "Tags: {idea_tags}"
        idea = Idea(
            title="Test",
            description="desc",
            confidence_score=0.5,
            tags=[],
            technologies=[],
            potential_challenges=[],
            monetization_strategies=[],
            related_ideas=[],
            extraction_metadata={},
        )
        result = InvestigationRunner._build_user_prompt(template, idea)
        assert "None" in result

    def test_build_user_prompt_leaves_unknown_placeholders(self):
        """Unknown placeholders should be left as-is without raising KeyError."""
        template = "Title: {idea_title}\nUnknown: {some_unknown_field}"
        idea = Idea(
            title="Test Idea",
            description="desc",
            confidence_score=0.5,
            tags=[],
            technologies=[],
            potential_challenges=[],
            monetization_strategies=[],
            related_ideas=[],
            extraction_metadata={},
        )
        result = InvestigationRunner._build_user_prompt(template, idea)
        assert "Test Idea" in result
        assert "{some_unknown_field}" in result

    def test_format_list_as_bullets_with_strings(self):
        """_format_list_as_bullets should format string items as bullet points."""
        result = InvestigationRunner._format_list_as_bullets(["Item 1", "Item 2", "Item 3"])
        assert "- Item 1" in result
        assert "- Item 2" in result
        assert "- Item 3" in result

    def test_format_list_as_bullets_with_dicts(self):
        """_format_list_as_bullets should extract text from dict items."""
        items = [{"text": "Challenge 1"}, {"name": "Challenge 2"}, {"description": "Challenge 3"}]
        result = InvestigationRunner._format_list_as_bullets(items)
        assert "- Challenge 1" in result
        assert "- Challenge 2" in result
        assert "- Challenge 3" in result

    def test_format_list_as_bullets_empty_list(self):
        """_format_list_as_bullets should return 'None identified' for empty lists."""
        result = InvestigationRunner._format_list_as_bullets([])
        assert result == "None identified"

    def test_format_list_as_bullets_none_input(self):
        """_format_list_as_bullets should handle None input."""
        result = InvestigationRunner._format_list_as_bullets(None)
        assert result == "None identified"


class TestInvestigationRunnerGetRunsForIdea:
    """Tests for get_runs_for_idea and get_run helper methods."""

    @pytest.mark.asyncio
    async def test_get_runs_for_idea_returns_empty_list(self, db_session):
        """get_runs_for_idea should return empty list when no runs exist."""
        idea = await _create_idea(db_session)
        await db_session.commit()

        runner = InvestigationRunner(db_session)
        runs = await runner.get_runs_for_idea(idea.id)
        assert runs == []

    @pytest.mark.asyncio
    async def test_get_run_not_found_raises(self, db_session):
        """get_run with non-existent ID should raise NotFoundException."""
        from app.core.exceptions import NotFoundException

        await db_session.commit()
        runner = InvestigationRunner(db_session)
        with pytest.raises(NotFoundException):
            await runner.get_run(uuid4())
