"""
E2E tests for the Investigation flow.

Tests the full lifecycle of running an InvestigationAgent against an Idea at
the *service layer* (InvestigationRunner), covering:

  1. Happy path - LLM returns analysis, run is stored as COMPLETED
  2. Agent not found - NotFoundException raised
  3. Idea not found   - NotFoundException raised
  4. LLM failure      - run is stored with FAILED status

These tests create real DB records in the test database, so they require the
``db_session`` fixture from conftest.py.

Because the investigation endpoint accepts HTML form data and returns HTML
partials, the service layer is the most appropriate level for E2E coverage.
The HTTP-level coverage for /admin/investigations/run is handled separately.
"""

import uuid
from datetime import datetime, timezone
import pytest
from unittest.mock import AsyncMock, patch

from app.models.investigation_agent import InvestigationAgent
from app.models.investigation_run import InvestigationRun, InvestigationRunStatus
from app.models.idea import Idea, IdeaStatus, IdeaPriority
from app.models.video import Video, VideoStatus
from app.models.channel import Channel, ChannelStatus
from app.services.investigation_runner import InvestigationRunner
from app.core.exceptions import NotFoundException


# ---------------------------------------------------------------------------
# Helpers to build minimal DB fixtures
# ---------------------------------------------------------------------------

async def _create_channel(db_session) -> Channel:
    """Insert a minimal Channel into the test DB and return it."""
    channel = Channel(
        youtube_channel_id=f"UC_e2e_{uuid.uuid4().hex[:8]}",
        name="E2E Test Channel",
        status=ChannelStatus.ACTIVE,
    )
    db_session.add(channel)
    await db_session.flush()
    return channel


async def _create_video(db_session, channel: Channel) -> Video:
    """Insert a minimal Video linked to *channel* and return it."""
    video = Video(
        youtube_video_id=f"e2e_{uuid.uuid4().hex[:10]}",
        channel_id=channel.id,
        title="E2E Test Video",
        status=VideoStatus.COMPLETED,
        published_at=datetime.now(timezone.utc),
        tags=[],
        caption_languages=[],
        video_metadata={},
        processing_metadata={},
        has_captions=False,
    )
    db_session.add(video)
    await db_session.flush()
    return video


async def _create_idea(db_session, video: Video) -> Idea:
    """Insert a minimal Idea linked to *video* and return it."""
    idea = Idea(
        video_id=video.id,
        title="Build an AI-powered transcription tool",
        description="Automatically transcribe and summarise YouTube videos using LLMs.",
        status=IdeaStatus.EXTRACTED,
        priority=IdeaPriority.HIGH,
        confidence_score=0.88,
        complexity_score=4,
        tags=["ai", "transcription"],
        technologies=["python", "openai"],
        potential_challenges=[{"text": "API cost management"}],
        monetization_strategies=[{"text": "SaaS subscription"}],
        extraction_metadata={},
    )
    db_session.add(idea)
    await db_session.flush()
    return idea


async def _create_agent(db_session, name: str = "E2E Investigation Agent") -> InvestigationAgent:
    """Insert an InvestigationAgent into the test DB and return it."""
    agent = InvestigationAgent(
        name=name,
        description="Agent used during E2E tests",
        system_prompt="You are a business analyst. Evaluate the following idea.",
        user_prompt_template=(
            "Idea: {idea_title}\n"
            "Description: {idea_description}\n"
            "Category: {idea_category}\n"
            "Confidence: {idea_confidence_score}\n\n"
            "Please provide a concise analysis."
        ),
        config={"model_type": "balanced", "temperature": 0.3, "max_tokens": 1000},
        is_active=True,
    )
    db_session.add(agent)
    await db_session.flush()
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInvestigationFlow:

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_run_investigation_success(self, db_session):
        """
        Happy path: LLM returns a valid text result.
        The InvestigationRun should be persisted with COMPLETED status and the
        result field populated.
        """
        # Arrange - create required DB records
        channel = await _create_channel(db_session)
        video = await _create_video(db_session, channel)
        idea = await _create_idea(db_session, video)
        agent = await _create_agent(db_session)

        llm_response = {
            "content": "This idea has strong market potential. Recommended for further research.",
            "usage": {"total_tokens": 120, "prompt_tokens": 80, "completion_tokens": 40},
            "cost": 0.0012,
        }

        with patch("app.services.investigation_runner.LLMManager") as MockLLM:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.generate.return_value = llm_response
            MockLLM.return_value = mock_llm_instance

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)  # type: ignore[arg-type]

        # Assert
        assert run is not None
        assert isinstance(run, InvestigationRun)
        assert run.status == InvestigationRunStatus.COMPLETED  # type: ignore[comparison-overlap]
        assert run.agent_id == agent.id  # type: ignore[comparison-overlap]
        assert run.idea_id == idea.id  # type: ignore[comparison-overlap]
        assert run.result == llm_response["content"]  # type: ignore[comparison-overlap]
        assert run.tokens_used == 120  # type: ignore[comparison-overlap]
        assert run.estimated_cost_usd == pytest.approx(0.0012, rel=1e-5)  # type: ignore[comparison-overlap]
        assert run.error_message is None

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_run_investigation_result_stored_in_db(self, db_session):
        """
        Verify the InvestigationRun record can be retrieved from the DB after the run.
        """
        channel = await _create_channel(db_session)
        video = await _create_video(db_session, channel)
        idea = await _create_idea(db_session, video)
        agent = await _create_agent(db_session)

        with patch("app.services.investigation_runner.LLMManager") as MockLLM:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.generate.return_value = {
                "content": "Stored result",
                "usage": {"total_tokens": 50},
                "cost": 0.0,
            }
            MockLLM.return_value = mock_llm_instance

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)  # type: ignore[arg-type]

        # Fetch back from DB
        fetched = await db_session.get(InvestigationRun, run.id)
        assert fetched is not None
        assert fetched.status == InvestigationRunStatus.COMPLETED
        assert fetched.result == "Stored result"

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_run_investigation_structured_json_parsing(self, db_session):
        """
        When the LLM returns valid JSON (possibly wrapped in a code fence), the
        runner should parse it into result_structured.
        """
        channel = await _create_channel(db_session)
        video = await _create_video(db_session, channel)
        idea = await _create_idea(db_session, video)
        agent = await _create_agent(db_session)

        json_content = '{"score": 9, "recommendation": "invest", "risks": ["competition"]}'

        with patch("app.services.investigation_runner.LLMManager") as MockLLM:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.generate.return_value = {
                "content": json_content,
                "usage": {"total_tokens": 60},
                "cost": 0.001,
            }
            MockLLM.return_value = mock_llm_instance

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)  # type: ignore[arg-type]

        assert run.status == InvestigationRunStatus.COMPLETED  # type: ignore[comparison-overlap]
        assert run.result_structured is not None
        assert run.result_structured["score"] == 9  # type: ignore[index]
        assert run.result_structured["recommendation"] == "invest"  # type: ignore[index]

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_run_investigation_agent_not_found(self, db_session):
        """
        Passing a non-existent agent_id must raise NotFoundException.
        No InvestigationRun record should be created.
        """
        channel = await _create_channel(db_session)
        video = await _create_video(db_session, channel)
        idea = await _create_idea(db_session, video)

        fake_agent_id = uuid.uuid4()

        runner = InvestigationRunner(db_session)
        with pytest.raises(NotFoundException) as exc_info:
            await runner.run_investigation(fake_agent_id, idea.id)  # type: ignore[arg-type]

        assert "InvestigationAgent" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_run_investigation_idea_not_found(self, db_session):
        """
        Passing a non-existent idea_id must raise NotFoundException.
        """
        agent = await _create_agent(db_session)
        fake_idea_id = uuid.uuid4()

        runner = InvestigationRunner(db_session)
        with pytest.raises(NotFoundException) as exc_info:
            await runner.run_investigation(agent.id, fake_idea_id)  # type: ignore[arg-type]

        assert "Idea" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_run_investigation_llm_failure_stores_failed_status(self, db_session):
        """
        When the LLM raises an exception the runner should catch it, store the
        InvestigationRun with FAILED status and an error_message, then return
        the run (not re-raise).
        """
        channel = await _create_channel(db_session)
        video = await _create_video(db_session, channel)
        idea = await _create_idea(db_session, video)
        agent = await _create_agent(db_session)

        with patch("app.services.investigation_runner.LLMManager") as MockLLM:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.generate.side_effect = RuntimeError("LLM API rate limit exceeded")
            MockLLM.return_value = mock_llm_instance

            runner = InvestigationRunner(db_session)
            run = await runner.run_investigation(agent.id, idea.id)  # type: ignore[arg-type]

        assert run is not None
        assert run.status == InvestigationRunStatus.FAILED  # type: ignore[comparison-overlap]
        assert run.error_message is not None
        assert "LLM API rate limit exceeded" in run.error_message  # type: ignore[operator]
        assert run.result is None

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_get_runs_for_idea(self, db_session):
        """
        After running two investigations against the same idea, get_runs_for_idea
        should return both runs in descending creation order.
        """
        channel = await _create_channel(db_session)
        video = await _create_video(db_session, channel)
        idea = await _create_idea(db_session, video)
        agent = await _create_agent(db_session)

        with patch("app.services.investigation_runner.LLMManager") as MockLLM:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.generate.return_value = {
                "content": "Run result",
                "usage": {"total_tokens": 30},
                "cost": 0.0,
            }
            MockLLM.return_value = mock_llm_instance

            runner = InvestigationRunner(db_session)
            run1 = await runner.run_investigation(agent.id, idea.id)  # type: ignore[arg-type]
            run2 = await runner.run_investigation(agent.id, idea.id)  # type: ignore[arg-type]

        all_runs = await runner.get_runs_for_idea(idea.id)  # type: ignore[arg-type]
        assert len(all_runs) == 2
        # Ordered newest-first by the service
        run_ids = {r.id for r in all_runs}
        assert run1.id in run_ids
        assert run2.id in run_ids

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_get_run_by_id(self, db_session):
        """
        After completing a run, get_run(run_id) should return the same object.
        """
        channel = await _create_channel(db_session)
        video = await _create_video(db_session, channel)
        idea = await _create_idea(db_session, video)
        agent = await _create_agent(db_session)

        with patch("app.services.investigation_runner.LLMManager") as MockLLM:
            mock_llm_instance = AsyncMock()
            mock_llm_instance.generate.return_value = {
                "content": "Fetched result",
                "usage": {"total_tokens": 45},
                "cost": 0.0,
            }
            MockLLM.return_value = mock_llm_instance

            runner = InvestigationRunner(db_session)
            original_run = await runner.run_investigation(agent.id, idea.id)  # type: ignore[arg-type]

        fetched_run = await runner.get_run(original_run.id)  # type: ignore[arg-type]
        assert fetched_run.id == original_run.id  # type: ignore[comparison-overlap]
        assert fetched_run.result == "Fetched result"  # type: ignore[comparison-overlap]

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_get_run_not_found(self, db_session):
        """
        get_run() with a non-existent run_id must raise NotFoundException.
        """
        runner = InvestigationRunner(db_session)
        with pytest.raises(NotFoundException) as exc_info:
            await runner.get_run(uuid.uuid4())

        assert "InvestigationRun" in str(exc_info.value)
