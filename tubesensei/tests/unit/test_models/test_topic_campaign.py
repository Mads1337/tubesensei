"""
Unit tests for the TopicCampaign model.

Tests creation, status transitions, progress tracking, and computed properties.
"""
import pytest
from datetime import datetime, timezone, timedelta

from app.models.topic_campaign import TopicCampaign, CampaignStatus


class TestCampaignStatusEnum:
    """Tests for CampaignStatus enum."""

    def test_all_expected_values_exist(self):
        values = {s.value for s in CampaignStatus}
        assert "draft" in values
        assert "running" in values
        assert "paused" in values
        assert "completed" in values
        assert "failed" in values
        assert "cancelled" in values


class TestTopicCampaignCreation:
    """Tests for TopicCampaign model creation."""

    @pytest.mark.asyncio
    async def test_basic_creation(self, db_session):
        """Test creating a TopicCampaign with required fields."""
        campaign = TopicCampaign(
            name="Test Campaign",
            topic="how to make money with AI",
            status=CampaignStatus.DRAFT,
            config={},
            campaign_metadata={},
            statistics={},
        )
        db_session.add(campaign)
        await db_session.commit()

        assert campaign.id is not None
        assert campaign.name == "Test Campaign"
        assert campaign.topic == "how to make money with AI"
        assert campaign.status == CampaignStatus.DRAFT

    @pytest.mark.asyncio
    async def test_defaults(self, db_session):
        """Test that defaults are applied correctly."""
        campaign = TopicCampaign(
            name="Default Campaign",
            topic="test topic",
            config={},
            campaign_metadata={},
            statistics={},
        )
        db_session.add(campaign)
        await db_session.commit()

        assert campaign.total_videos_discovered == 0
        assert campaign.total_videos_relevant == 0
        assert campaign.total_videos_filtered == 0
        assert campaign.total_channels_explored == 0
        assert campaign.total_transcripts_extracted == 0
        assert campaign.progress_percent == 0.0
        assert campaign.error_count == 0
        assert campaign.api_calls_made == 0
        assert campaign.llm_calls_made == 0


class TestTopicCampaignStatusChecks:
    """Tests for TopicCampaign status check properties."""

    def _make_campaign(self, status: CampaignStatus) -> TopicCampaign:
        return TopicCampaign(
            name="Campaign",
            topic="test topic",
            status=status,
            config={},
            campaign_metadata={},
            statistics={},
        )

    def test_is_active_running(self):
        campaign = self._make_campaign(CampaignStatus.RUNNING)
        assert campaign.is_active is True

    def test_is_active_paused(self):
        campaign = self._make_campaign(CampaignStatus.PAUSED)
        assert campaign.is_active is True

    def test_is_active_draft(self):
        campaign = self._make_campaign(CampaignStatus.DRAFT)
        assert campaign.is_active is False

    def test_is_complete_completed(self):
        campaign = self._make_campaign(CampaignStatus.COMPLETED)
        assert campaign.is_complete is True

    def test_is_complete_failed(self):
        campaign = self._make_campaign(CampaignStatus.FAILED)
        assert campaign.is_complete is True

    def test_is_complete_cancelled(self):
        campaign = self._make_campaign(CampaignStatus.CANCELLED)
        assert campaign.is_complete is True

    def test_is_complete_running(self):
        campaign = self._make_campaign(CampaignStatus.RUNNING)
        assert campaign.is_complete is False

    def test_can_start_draft(self):
        campaign = self._make_campaign(CampaignStatus.DRAFT)
        assert campaign.can_start is True

    def test_can_start_not_draft(self):
        for status in [CampaignStatus.RUNNING, CampaignStatus.PAUSED, CampaignStatus.COMPLETED]:
            campaign = self._make_campaign(status)
            assert campaign.can_start is False

    def test_can_pause_running(self):
        campaign = self._make_campaign(CampaignStatus.RUNNING)
        assert campaign.can_pause is True

    def test_can_pause_not_running(self):
        campaign = self._make_campaign(CampaignStatus.DRAFT)
        assert campaign.can_pause is False

    def test_can_resume_paused(self):
        campaign = self._make_campaign(CampaignStatus.PAUSED)
        assert campaign.can_resume is True

    def test_can_resume_not_paused(self):
        campaign = self._make_campaign(CampaignStatus.RUNNING)
        assert campaign.can_resume is False

    def test_can_cancel_running(self):
        campaign = self._make_campaign(CampaignStatus.RUNNING)
        assert campaign.can_cancel is True

    def test_can_cancel_paused(self):
        campaign = self._make_campaign(CampaignStatus.PAUSED)
        assert campaign.can_cancel is True

    def test_can_cancel_draft(self):
        campaign = self._make_campaign(CampaignStatus.DRAFT)
        assert campaign.can_cancel is False


class TestTopicCampaignStateTransitions:
    """Tests for TopicCampaign state transition methods."""

    def _make_draft_campaign(self) -> TopicCampaign:
        return TopicCampaign(
            name="Campaign",
            topic="test topic",
            status=CampaignStatus.DRAFT,
            config={},
            campaign_metadata={},
            statistics={},
        )

    def test_start(self):
        campaign = self._make_draft_campaign()
        campaign.start()

        assert campaign.status == CampaignStatus.RUNNING
        assert campaign.started_at is not None

    def test_start_from_non_draft_raises(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        with pytest.raises(ValueError):
            campaign.start()

    def test_pause(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        campaign.pause()

        assert campaign.status == CampaignStatus.PAUSED
        assert campaign.paused_at is not None

    def test_pause_from_non_running_raises(self):
        campaign = self._make_draft_campaign()
        with pytest.raises(ValueError):
            campaign.pause()

    def test_resume(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.PAUSED,
            config={},
            campaign_metadata={},
            statistics={},
        )
        campaign.resume()

        assert campaign.status == CampaignStatus.RUNNING
        assert campaign.paused_at is None

    def test_resume_from_non_paused_raises(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        with pytest.raises(ValueError):
            campaign.resume()

    def test_complete(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        campaign.started_at = datetime.now(timezone.utc) - timedelta(seconds=100)
        campaign.complete()

        assert campaign.status == CampaignStatus.COMPLETED
        assert campaign.completed_at is not None
        assert campaign.progress_percent == 100.0
        assert campaign.execution_time_seconds is not None

    def test_fail(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        campaign.fail("Something went wrong")

        assert campaign.status == CampaignStatus.FAILED
        assert campaign.error_message == "Something went wrong"
        assert campaign.completed_at is not None

    def test_cancel(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        campaign.cancel()

        assert campaign.status == CampaignStatus.CANCELLED
        assert campaign.completed_at is not None

    def test_cancel_from_non_cancellable_raises(self):
        campaign = self._make_draft_campaign()
        with pytest.raises(ValueError):
            campaign.cancel()


class TestTopicCampaignProgressTracking:
    """Tests for TopicCampaign progress tracking methods."""

    def _make_campaign(self) -> TopicCampaign:
        return TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.RUNNING,
            config={"total_video_limit": 100},
            campaign_metadata={},
            statistics={},
            total_videos_discovered=0,
            total_videos_relevant=0,
            total_videos_filtered=0,
            total_channels_explored=0,
            total_transcripts_extracted=0,
            progress_percent=0.0,
            error_count=0,
            api_calls_made=0,
            llm_calls_made=0,
        )

    def test_increment_discovered(self):
        campaign = self._make_campaign()
        campaign.increment_discovered(5)
        assert campaign.total_videos_discovered == 5

    def test_increment_relevant(self):
        campaign = self._make_campaign()
        campaign.increment_relevant(10)
        assert campaign.total_videos_relevant == 10

    def test_increment_filtered(self):
        campaign = self._make_campaign()
        campaign.increment_filtered(3)
        assert campaign.total_videos_filtered == 3

    def test_increment_channels(self):
        campaign = self._make_campaign()
        campaign.increment_channels(2)
        assert campaign.total_channels_explored == 2

    def test_increment_transcripts(self):
        campaign = self._make_campaign()
        campaign.increment_transcripts(7)
        assert campaign.total_transcripts_extracted == 7

    def test_increment_api_calls(self):
        campaign = self._make_campaign()
        campaign.increment_api_calls(15)
        assert campaign.api_calls_made == 15

    def test_increment_llm_calls(self):
        campaign = self._make_campaign()
        campaign.increment_llm_calls(4)
        assert campaign.llm_calls_made == 4

    def test_update_progress(self):
        campaign = self._make_campaign()
        campaign.total_videos_relevant = 50
        campaign.update_progress()
        assert campaign.progress_percent == 50.0

    def test_update_progress_caps_at_100(self):
        campaign = self._make_campaign()
        campaign.total_videos_relevant = 200  # Over limit of 100
        campaign.update_progress()
        assert campaign.progress_percent == 100.0

    def test_filter_acceptance_rate_no_videos(self):
        campaign = self._make_campaign()
        assert campaign.filter_acceptance_rate == 0.0

    def test_filter_acceptance_rate(self):
        campaign = self._make_campaign()
        campaign.total_videos_relevant = 70
        campaign.total_videos_filtered = 30
        rate = campaign.filter_acceptance_rate
        assert rate == 70.0

    def test_has_reached_limit(self):
        campaign = self._make_campaign()
        campaign.total_videos_relevant = 100
        assert campaign.has_reached_limit is True

    def test_has_not_reached_limit(self):
        campaign = self._make_campaign()
        campaign.total_videos_relevant = 50
        assert campaign.has_reached_limit is False

    def test_record_error(self):
        campaign = self._make_campaign()
        campaign.record_error("Something failed")
        assert campaign.error_count == 1
        assert campaign.error_message == "Something failed"


class TestTopicCampaignIsStale:
    """Tests for is_stale property."""

    def test_is_stale_not_running(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.PAUSED,
            config={},
            campaign_metadata={},
            statistics={},
        )
        assert campaign.is_stale is False

    def test_is_stale_running_with_recent_heartbeat(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        campaign.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert campaign.is_stale is False

    def test_is_stale_running_with_old_heartbeat(self):
        campaign = TopicCampaign(
            name="Campaign",
            topic="test",
            status=CampaignStatus.RUNNING,
            config={},
            campaign_metadata={},
            statistics={},
        )
        campaign.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(minutes=15)
        assert campaign.is_stale is True


class TestTopicCampaignConfigAccessors:
    """Tests for TopicCampaign config property accessors."""

    def test_total_video_limit_default(self):
        campaign = TopicCampaign(
            name="c", topic="t", config={}, campaign_metadata={}, statistics={}
        )
        assert campaign.total_video_limit == 3000

    def test_total_video_limit_custom(self):
        campaign = TopicCampaign(
            name="c",
            topic="t",
            config={"total_video_limit": 500},
            campaign_metadata={},
            statistics={},
        )
        assert campaign.total_video_limit == 500

    def test_filter_threshold_default(self):
        campaign = TopicCampaign(
            name="c", topic="t", config={}, campaign_metadata={}, statistics={}
        )
        assert campaign.filter_threshold == 0.7

    def test_get_summary_returns_dict(self):
        campaign = TopicCampaign(
            name="My Campaign",
            topic="test topic",
            status=CampaignStatus.DRAFT,
            config={},
            campaign_metadata={},
            statistics={},
            total_videos_discovered=0,
            total_videos_relevant=0,
            total_videos_filtered=0,
            total_channels_explored=0,
            total_transcripts_extracted=0,
            progress_percent=0.0,
            error_count=0,
            api_calls_made=0,
            llm_calls_made=0,
        )
        summary = campaign.get_summary()
        assert "name" in summary
        assert "topic" in summary
        assert "status" in summary
        assert "progress_percent" in summary
        assert summary["status"] == "draft"
