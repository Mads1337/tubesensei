"""
Unit tests for the InvestigationRun model.

Tests creation, status enum, and FK relationships.
"""
import pytest
from datetime import datetime, timezone

from app.models.investigation_run import InvestigationRun, InvestigationRunStatus
from app.models.investigation_agent import InvestigationAgent
from app.models.idea import Idea, IdeaStatus, IdeaPriority
from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus

_NOW = datetime.now(timezone.utc)


class TestInvestigationRunStatusEnum:
    """Tests for InvestigationRunStatus enum."""

    def test_all_expected_values_exist(self):
        values = {s.value for s in InvestigationRunStatus}
        assert "pending" in values
        assert "running" in values
        assert "completed" in values
        assert "failed" in values


class TestInvestigationRunCreation:
    """Tests for InvestigationRun model creation."""

    async def _create_idea(self, db_session) -> Idea:
        """Helper to create a minimal idea with all required parents."""
        channel = Channel(
            youtube_channel_id="UCinv001",
            name="Inv Test Channel",
            status=ChannelStatus.ACTIVE,
        )
        db_session.add(channel)
        await db_session.flush()

        video = Video(
            youtube_video_id="inv001vid",
            channel_id=channel.id,
            title="Inv Test Video",
            status=VideoStatus.DISCOVERED,
            published_at=_NOW,
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
            title="Inv Test Idea",
            description="Test idea for investigation run",
            status=IdeaStatus.EXTRACTED,
            priority=IdeaPriority.MEDIUM,
            confidence_score=0.8,
            tags=[],
            technologies=[],
            potential_challenges=[],
            monetization_strategies=[],
            related_ideas=[],
            extraction_metadata={},
        )
        db_session.add(idea)
        await db_session.flush()
        return idea

    async def _create_agent(self, db_session) -> InvestigationAgent:
        """Helper to create a minimal investigation agent."""
        agent = InvestigationAgent(
            name="Test Investigation Agent",
            system_prompt="You are an analyst.",
            user_prompt_template="Analyze: {idea_title}\n{idea_description}",
            config={},
            is_active=True,
        )
        db_session.add(agent)
        await db_session.flush()
        return agent

    @pytest.mark.asyncio
    async def test_basic_creation(self, db_session):
        """Test creating an InvestigationRun with required fields."""
        agent = await self._create_agent(db_session)
        idea = await self._create_idea(db_session)

        run = InvestigationRun(
            agent_id=agent.id,
            idea_id=idea.id,
            status=InvestigationRunStatus.PENDING,
        )
        db_session.add(run)
        await db_session.commit()

        assert run.id is not None
        assert run.agent_id == agent.id
        assert run.idea_id == idea.id
        assert run.status == InvestigationRunStatus.PENDING

    @pytest.mark.asyncio
    async def test_default_status_is_pending(self, db_session):
        """Test that status defaults to PENDING."""
        agent = await self._create_agent(db_session)
        idea = await self._create_idea(db_session)

        run = InvestigationRun(
            agent_id=agent.id,
            idea_id=idea.id,
        )
        db_session.add(run)
        await db_session.commit()

        assert run.status == InvestigationRunStatus.PENDING

    @pytest.mark.asyncio
    async def test_optional_fields_start_as_none(self, db_session):
        """Test that optional fields are None by default."""
        agent = await self._create_agent(db_session)
        idea = await self._create_idea(db_session)

        run = InvestigationRun(
            agent_id=agent.id,
            idea_id=idea.id,
            status=InvestigationRunStatus.PENDING,
        )
        db_session.add(run)
        await db_session.commit()

        assert run.result is None
        assert run.result_structured is None
        assert run.tokens_used is None
        assert run.estimated_cost_usd is None
        assert run.error_message is None

    @pytest.mark.asyncio
    async def test_completed_run_with_result(self, db_session):
        """Test an InvestigationRun in COMPLETED state with result data."""
        agent = await self._create_agent(db_session)
        idea = await self._create_idea(db_session)

        run = InvestigationRun(
            agent_id=agent.id,
            idea_id=idea.id,
            status=InvestigationRunStatus.COMPLETED,
            result='{"analysis": "Great idea"}',
            result_structured={"analysis": "Great idea"},
            tokens_used=500,
            estimated_cost_usd=0.01,
        )
        db_session.add(run)
        await db_session.commit()

        assert run.status == InvestigationRunStatus.COMPLETED
        assert run.result == '{"analysis": "Great idea"}'
        assert run.result_structured == {"analysis": "Great idea"}
        assert run.tokens_used == 500
        assert run.estimated_cost_usd == pytest.approx(0.01)

    @pytest.mark.asyncio
    async def test_failed_run_with_error(self, db_session):
        """Test an InvestigationRun in FAILED state with error message."""
        agent = await self._create_agent(db_session)
        idea = await self._create_idea(db_session)

        run = InvestigationRun(
            agent_id=agent.id,
            idea_id=idea.id,
            status=InvestigationRunStatus.FAILED,
            error_message="LLM API call failed",
        )
        db_session.add(run)
        await db_session.commit()

        assert run.status == InvestigationRunStatus.FAILED
        assert run.error_message == "LLM API call failed"

    @pytest.mark.asyncio
    async def test_has_timestamps(self, db_session):
        """Test that timestamps are created automatically."""
        agent = await self._create_agent(db_session)
        idea = await self._create_idea(db_session)

        run = InvestigationRun(
            agent_id=agent.id,
            idea_id=idea.id,
            status=InvestigationRunStatus.PENDING,
        )
        db_session.add(run)
        await db_session.commit()

        assert run.created_at is not None
        assert run.updated_at is not None

    def test_repr(self):
        """Test string representation."""
        from uuid import uuid4
        run = InvestigationRun(
            agent_id=uuid4(),
            idea_id=uuid4(),
            status=InvestigationRunStatus.PENDING,
        )
        repr_str = repr(run)
        assert "InvestigationRun" in repr_str
        assert "pending" in repr_str
