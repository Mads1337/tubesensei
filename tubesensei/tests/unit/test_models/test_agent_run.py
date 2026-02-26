"""
Unit tests for the AgentRun model.

Tests creation, status transitions, computed properties, and helper methods.
"""
import pytest
from uuid import uuid4
from datetime import datetime, timezone, timedelta

from app.models.agent_run import AgentRun, AgentType, AgentRunStatus
from app.models.topic_campaign import TopicCampaign, CampaignStatus


class TestAgentTypeEnum:
    """Tests for AgentType enum."""

    def test_all_expected_values_exist(self):
        values = {t.value for t in AgentType}
        assert "coordinator" in values
        assert "search" in values
        assert "channel_expansion" in values
        assert "topic_filter" in values
        assert "transcription" in values
        assert "similar_videos" in values
        assert "idea_extraction" in values


class TestAgentRunStatusEnum:
    """Tests for AgentRunStatus enum."""

    def test_all_expected_values_exist(self):
        values = {s.value for s in AgentRunStatus}
        assert "pending" in values
        assert "running" in values
        assert "completed" in values
        assert "failed" in values
        assert "cancelled" in values


class TestAgentRunCreation:
    """Tests for AgentRun model creation."""

    @pytest.mark.asyncio
    async def test_basic_creation(self, db_session):
        """Test creating an AgentRun with required fields."""
        campaign = TopicCampaign(
            name="Test Campaign",
            topic="test topic",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        db_session.add(campaign)
        await db_session.flush()

        agent_run = AgentRun(
            campaign_id=campaign.id,
            agent_type=AgentType.SEARCH,
            status=AgentRunStatus.PENDING,
            input_data={"query": "test"},
            errors=[],
            agent_metadata={},
        )
        db_session.add(agent_run)
        await db_session.commit()

        assert agent_run.id is not None
        assert agent_run.campaign_id == campaign.id
        assert agent_run.agent_type == AgentType.SEARCH
        assert agent_run.status == AgentRunStatus.PENDING

    @pytest.mark.asyncio
    async def test_defaults(self, db_session):
        """Test that default values are applied correctly."""
        campaign = TopicCampaign(
            name="Test Campaign 2",
            topic="test topic",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        db_session.add(campaign)
        await db_session.flush()

        agent_run = AgentRun(
            campaign_id=campaign.id,
            agent_type=AgentType.TOPIC_FILTER,
            input_data={},
            errors=[],
            agent_metadata={},
        )
        db_session.add(agent_run)
        await db_session.commit()

        assert agent_run.items_processed == 0
        assert agent_run.items_produced == 0
        assert agent_run.api_calls_made == 0
        assert agent_run.llm_calls_made == 0
        assert agent_run.progress_percent == 0.0
        assert agent_run.retry_count == 0
        assert agent_run.max_retries == 3
        assert agent_run.rate_limited is False

    @pytest.mark.asyncio
    async def test_factory_method(self, db_session):
        """Test create_for_agent factory method."""
        campaign = TopicCampaign(
            name="Test Campaign 3",
            topic="test topic",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        db_session.add(campaign)
        await db_session.flush()

        agent_run = AgentRun.create_for_agent(
            campaign_id=campaign.id,
            agent_type=AgentType.CHANNEL_EXPANSION,
            input_data={"channel_id": "UC123"},
        )
        db_session.add(agent_run)
        await db_session.commit()

        assert agent_run.campaign_id == campaign.id
        assert agent_run.agent_type == AgentType.CHANNEL_EXPANSION
        assert agent_run.status == AgentRunStatus.PENDING
        assert agent_run.input_data == {"channel_id": "UC123"}


class TestAgentRunComputedProperties:
    """Tests for AgentRun computed properties."""

    def _make_run(self, status: AgentRunStatus) -> AgentRun:
        return AgentRun(
            campaign_id=uuid4(),
            agent_type=AgentType.SEARCH,
            status=status,
            input_data={},
            errors=[],
            agent_metadata={},
            items_processed=0,
            items_produced=0,
            api_calls_made=0,
            llm_calls_made=0,
            progress_percent=0.0,
            retry_count=0,
            max_retries=3,
            rate_limited=False,
        )

    def test_is_active_pending(self):
        run = self._make_run(AgentRunStatus.PENDING)
        assert run.is_active is True

    def test_is_active_running(self):
        run = self._make_run(AgentRunStatus.RUNNING)
        assert run.is_active is True

    def test_is_active_completed(self):
        run = self._make_run(AgentRunStatus.COMPLETED)
        assert run.is_active is False

    def test_is_complete_completed(self):
        run = self._make_run(AgentRunStatus.COMPLETED)
        assert run.is_complete is True

    def test_is_complete_failed(self):
        run = self._make_run(AgentRunStatus.FAILED)
        assert run.is_complete is True

    def test_is_complete_cancelled(self):
        run = self._make_run(AgentRunStatus.CANCELLED)
        assert run.is_complete is True

    def test_is_complete_running(self):
        run = self._make_run(AgentRunStatus.RUNNING)
        assert run.is_complete is False

    def test_is_successful(self):
        run = self._make_run(AgentRunStatus.COMPLETED)
        assert run.is_successful is True

    def test_is_successful_failed(self):
        run = self._make_run(AgentRunStatus.FAILED)
        assert run.is_successful is False

    def test_can_retry_failed_within_limit(self):
        run = self._make_run(AgentRunStatus.FAILED)
        run.retry_count = 1
        run.max_retries = 3
        assert run.can_retry is True

    def test_can_retry_failed_at_limit(self):
        run = self._make_run(AgentRunStatus.FAILED)
        run.retry_count = 3
        run.max_retries = 3
        assert run.can_retry is False

    def test_can_retry_not_failed(self):
        run = self._make_run(AgentRunStatus.COMPLETED)
        run.retry_count = 0
        run.max_retries = 3
        assert run.can_retry is False

    def test_has_errors_with_error_message(self):
        run = self._make_run(AgentRunStatus.FAILED)
        run.error_message = "Something went wrong"
        run.errors = []
        assert run.has_errors is True

    def test_has_errors_with_errors_list(self):
        run = self._make_run(AgentRunStatus.COMPLETED)
        run.error_message = None
        run.errors = [{"message": "Non-fatal error"}]
        assert run.has_errors is True

    def test_has_errors_no_errors(self):
        run = self._make_run(AgentRunStatus.COMPLETED)
        run.error_message = None
        run.errors = []
        assert not run.has_errors

    def test_success_rate_no_items(self):
        run = self._make_run(AgentRunStatus.COMPLETED)
        run.items_processed = 0
        run.items_produced = 0
        assert run.success_rate == 0.0

    def test_success_rate_with_items(self):
        run = self._make_run(AgentRunStatus.COMPLETED)
        run.items_processed = 10
        run.items_produced = 8
        assert run.success_rate == 80.0

    def test_duration_seconds_no_start(self):
        run = self._make_run(AgentRunStatus.PENDING)
        run.started_at = None
        assert run.duration_seconds is None

    def test_duration_seconds_running(self):
        run = self._make_run(AgentRunStatus.RUNNING)
        run.started_at = datetime.now(timezone.utc) - timedelta(seconds=60)
        run.completed_at = None
        duration = run.duration_seconds
        assert duration is not None
        assert 55 < duration < 65


class TestAgentRunStateTransitions:
    """Tests for AgentRun state transition methods."""

    def _make_pending_run(self) -> AgentRun:
        return AgentRun(
            campaign_id=uuid4(),
            agent_type=AgentType.SEARCH,
            status=AgentRunStatus.PENDING,
            input_data={},
            errors=[],
            agent_metadata={},
            items_processed=0,
            items_produced=0,
            api_calls_made=0,
            llm_calls_made=0,
            progress_percent=0.0,
            retry_count=0,
            max_retries=3,
            rate_limited=False,
        )

    def test_start(self):
        run = self._make_pending_run()
        run.start()

        assert run.status == AgentRunStatus.RUNNING
        assert run.started_at is not None

    def test_complete(self):
        run = self._make_pending_run()
        run.start()
        run.complete(output_data={"result": "success"})

        assert run.status == AgentRunStatus.COMPLETED
        assert run.completed_at is not None
        assert run.progress_percent == 100.0
        assert run.output_data == {"result": "success"}
        assert run.execution_time_seconds is not None

    def test_complete_without_output(self):
        run = self._make_pending_run()
        run.start()
        run.complete()

        assert run.status == AgentRunStatus.COMPLETED
        assert run.output_data is None

    def test_fail(self):
        run = self._make_pending_run()
        run.start()
        run.fail("Something went wrong", error_details={"traceback": "..."})

        assert run.status == AgentRunStatus.FAILED
        assert run.error_message == "Something went wrong"
        assert run.error_details == {"traceback": "..."}
        assert run.completed_at is not None

    def test_cancel(self):
        run = self._make_pending_run()
        run.start()
        run.cancel()

        assert run.status == AgentRunStatus.CANCELLED
        assert run.completed_at is not None


class TestAgentRunIncrementMethods:
    """Tests for AgentRun increment methods."""

    def _make_run(self) -> AgentRun:
        return AgentRun(
            campaign_id=uuid4(),
            agent_type=AgentType.SEARCH,
            status=AgentRunStatus.RUNNING,
            input_data={},
            errors=[],
            agent_metadata={},
            items_processed=0,
            items_produced=0,
            api_calls_made=0,
            llm_calls_made=0,
            progress_percent=0.0,
            retry_count=0,
            max_retries=3,
            rate_limited=False,
        )

    def test_increment_processed(self):
        run = self._make_run()
        run.increment_processed(5)
        assert run.items_processed == 5

    def test_increment_produced(self):
        run = self._make_run()
        run.increment_produced(3)
        assert run.items_produced == 3

    def test_increment_api_calls(self):
        run = self._make_run()
        run.increment_api_calls(10)
        assert run.api_calls_made == 10

    def test_increment_llm_calls_with_tokens_and_cost(self):
        run = self._make_run()
        run.tokens_used = 0
        run.estimated_cost_usd = 0.0
        run.increment_llm_calls(count=2, tokens=1000, cost=0.01)

        assert run.llm_calls_made == 2
        assert run.tokens_used == 1000
        assert run.estimated_cost_usd == pytest.approx(0.01)

    def test_increment_retry(self):
        run = self._make_run()
        run.increment_retry()
        assert run.retry_count == 1

    def test_add_error(self):
        run = self._make_run()
        run.errors = []
        run.add_error("Non-fatal error")
        assert len(run.errors) == 1
        assert run.errors[0]["message"] == "Non-fatal error"

    def test_record_rate_limit(self):
        run = self._make_run()
        run.rate_limit_wait_seconds = None
        run.record_rate_limit(5.0)
        assert run.rate_limited is True
        assert run.rate_limit_wait_seconds == 5.0

    def test_update_progress(self):
        run = self._make_run()
        run.update_progress(75.0, current_item="Processing video 75/100")
        assert run.progress_percent == 75.0
        assert run.current_item == "Processing video 75/100"

    def test_update_progress_capped_at_100(self):
        run = self._make_run()
        run.update_progress(150.0)
        assert run.progress_percent == 100.0

    def test_get_summary(self):
        run = self._make_run()
        run.start()
        summary = run.get_summary()

        assert "id" in summary
        assert "campaign_id" in summary
        assert "agent_type" in summary
        assert "status" in summary
        assert summary["agent_type"] == "search"
        assert summary["status"] == "running"

    def test_repr(self):
        run = AgentRun(
            campaign_id=uuid4(),
            agent_type=AgentType.SEARCH,
            status=AgentRunStatus.PENDING,
            input_data={},
        )
        repr_str = repr(run)
        assert "AgentRun" in repr_str
        assert "search" in repr_str
        assert "pending" in repr_str
