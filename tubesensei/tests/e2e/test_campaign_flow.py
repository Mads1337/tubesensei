"""
E2E tests for the Campaign workflow.

Tests the TopicDiscoveryService at the service layer, covering:

  1. Creating a campaign (DRAFT status)
  2. Starting a campaign (RUNNING status)
  3. Pausing / resuming a campaign
  4. Completing a campaign
  5. Failing a campaign with an error message
  6. Updating campaign config (only allowed in DRAFT)
  7. Deleting a campaign (only allowed when not RUNNING)
  8. Lifecycle invariants (can_start, can_pause, can_resume, can_cancel)

These tests exercise the service layer without invoking any Celery tasks,
YouTube API calls, or LLM calls -- all external dependencies are either
not reached or mocked out where necessary.
"""

import uuid
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch

from app.models.topic_campaign import TopicCampaign, CampaignStatus
from app.services.topic_discovery import TopicDiscoveryService


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCampaignCreation:

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_create_campaign_minimal(self, db_session):
        """
        A campaign created with only name and topic should be persisted with
        DRAFT status and sensible defaults.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Test Campaign",
            topic="how to monetise YouTube videos with AI",
        )

        assert campaign is not None
        assert campaign.id is not None
        assert campaign.name == "Test Campaign"
        assert campaign.topic == "how to monetise YouTube videos with AI"
        assert campaign.status == CampaignStatus.DRAFT
        assert campaign.config is not None
        assert campaign.total_videos_discovered == 0
        assert campaign.total_videos_relevant == 0
        assert campaign.progress_percent == 0.0
        assert campaign.error_message is None

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_create_campaign_with_custom_config(self, db_session):
        """
        Config values provided at creation time should override the defaults.
        """
        service = TopicDiscoveryService(db_session)
        custom_config = {
            "total_video_limit": 100,
            "per_channel_limit": 2,
            "filter_threshold": 0.9,
        }
        campaign = await service.create_campaign(
            name="Limited Campaign",
            topic="AI automation tools",
            config=custom_config,
        )

        assert campaign.config["total_video_limit"] == 100
        assert campaign.config["per_channel_limit"] == 2
        assert campaign.config["filter_threshold"] == 0.9
        # Defaults for omitted keys should still be set
        assert "search_limit" in campaign.config

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_create_campaign_is_persisted(self, db_session):
        """
        The created campaign should be retrievable from the DB by its ID.
        """
        service = TopicDiscoveryService(db_session)
        created = await service.create_campaign(
            name="Persistence Check",
            topic="content marketing strategies",
        )

        fetched = await service.get_campaign(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.name == "Persistence Check"

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_list_campaigns(self, db_session):
        """
        After creating several campaigns, list_campaigns should return all of them.
        """
        service = TopicDiscoveryService(db_session)
        for i in range(3):
            await service.create_campaign(name=f"Campaign {i}", topic=f"topic {i}")

        campaigns = await service.list_campaigns()
        assert len(campaigns) >= 3


class TestCampaignLifecycle:

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_start_campaign_transitions_to_running(self, db_session):
        """
        start_campaign() should change status from DRAFT to RUNNING.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Start Test", topic="python tutorials"
        )
        assert campaign.status == CampaignStatus.DRAFT
        assert campaign.can_start is True

        started = await service.start_campaign(campaign.id)

        assert started.status == CampaignStatus.RUNNING
        assert started.started_at is not None
        assert started.can_pause is True
        assert started.can_start is False

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_cannot_start_already_running_campaign(self, db_session):
        """
        Attempting to start a campaign that is already RUNNING must raise ValueError.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Double Start", topic="machine learning"
        )
        await service.start_campaign(campaign.id)

        with pytest.raises(ValueError, match="cannot be started"):
            await service.start_campaign(campaign.id)

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_pause_running_campaign(self, db_session):
        """
        pause_campaign() on a RUNNING campaign should transition to PAUSED.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(name="Pause Test", topic="SEO strategy")
        await service.start_campaign(campaign.id)

        paused = await service.pause_campaign(campaign.id)

        assert paused.status == CampaignStatus.PAUSED
        assert paused.paused_at is not None
        assert paused.can_resume is True
        assert paused.can_pause is False

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_resume_paused_campaign(self, db_session):
        """
        resume_campaign() on a PAUSED campaign should transition back to RUNNING.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(name="Resume Test", topic="email marketing")
        await service.start_campaign(campaign.id)
        await service.pause_campaign(campaign.id)

        resumed = await service.resume_campaign(campaign.id)

        assert resumed.status == CampaignStatus.RUNNING
        assert resumed.paused_at is None

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_complete_campaign(self, db_session):
        """
        Calling campaign.complete() on a RUNNING campaign should mark it COMPLETED
        with 100% progress.  (Service layer uses the model method directly.)
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Completion Test", topic="dropshipping business"
        )
        campaign = await service.start_campaign(campaign.id)

        # Simulate the model-level completion (as a Celery worker would)
        campaign.complete()
        await db_session.commit()
        await db_session.refresh(campaign)

        assert campaign.status == CampaignStatus.COMPLETED
        assert campaign.progress_percent == 100.0
        assert campaign.completed_at is not None
        assert campaign.is_complete is True
        assert campaign.is_active is False

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_fail_campaign(self, db_session):
        """
        Calling campaign.fail() on a RUNNING campaign should mark it FAILED with
        the supplied error message.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Failure Test", topic="youtube automation"
        )
        campaign = await service.start_campaign(campaign.id)

        error_msg = "YouTube API quota exhausted"
        campaign.fail(error_msg)
        await db_session.commit()
        await db_session.refresh(campaign)

        assert campaign.status == CampaignStatus.FAILED
        assert campaign.error_message == error_msg
        assert campaign.completed_at is not None
        assert campaign.is_complete is True

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_cancel_running_campaign(self, db_session):
        """
        cancel_campaign() on a RUNNING campaign should transition to CANCELLED.
        The Celery revoke call is patched so no real broker is needed.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Cancel Test", topic="affiliate marketing"
        )
        await service.start_campaign(campaign.id)

        # cancel_campaign imports celery_app internally; patch at that level
        with patch("app.services.topic_discovery.celery_app") as mock_celery:
            mock_celery.control.revoke = MagicMock()
            cancelled = await service.cancel_campaign(campaign.id)

        assert cancelled.status == CampaignStatus.CANCELLED
        assert cancelled.completed_at is not None
        assert cancelled.is_complete is True


class TestCampaignUpdate:

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_update_draft_campaign(self, db_session):
        """
        Updating name and description on a DRAFT campaign should persist the changes.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Original Name", topic="newsletter growth"
        )

        updated = await service.update_campaign(
            campaign.id,
            name="Updated Name",
            description="Now with a description",
        )

        assert updated.name == "Updated Name"
        assert updated.description == "Now with a description"

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_update_running_campaign_raises(self, db_session):
        """
        Updating a campaign that is no longer in DRAFT status must raise ValueError.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Immutable Campaign", topic="crypto trading"
        )
        await service.start_campaign(campaign.id)

        with pytest.raises(ValueError, match="DRAFT"):
            await service.update_campaign(campaign.id, name="Should not work")

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_update_nonexistent_campaign_returns_none(self, db_session):
        """
        update_campaign() for a non-existent ID should return None.
        """
        service = TopicDiscoveryService(db_session)
        result = await service.update_campaign(
            uuid.uuid4(), name="Ghost Campaign"
        )
        assert result is None


class TestCampaignDeletion:

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_delete_draft_campaign(self, db_session):
        """
        A DRAFT campaign can be deleted; subsequent get_campaign should return None.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Deletable Campaign", topic="product review channel"
        )
        campaign_id = campaign.id

        deleted = await service.delete_campaign(campaign_id)
        assert deleted is True

        fetched = await service.get_campaign(campaign_id)
        assert fetched is None

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_delete_running_campaign_raises(self, db_session):
        """
        Deleting a RUNNING campaign must raise ValueError.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Running Campaign", topic="cooking channel"
        )
        await service.start_campaign(campaign.id)

        with pytest.raises(ValueError, match="running"):
            await service.delete_campaign(campaign.id)

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_delete_nonexistent_campaign_returns_false(self, db_session):
        """
        delete_campaign() for a non-existent ID should return False.
        """
        service = TopicDiscoveryService(db_session)
        result = await service.delete_campaign(uuid.uuid4())
        assert result is False


class TestCampaignProgressTracking:

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_increment_discovered_and_relevant(self, db_session):
        """
        Incrementing discovered / relevant counts via model helpers should update
        the persisted campaign.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Progress Campaign", topic="tech reviews"
        )
        campaign = await service.start_campaign(campaign.id)

        campaign.increment_discovered(10)
        campaign.increment_relevant(3)
        await db_session.commit()
        await db_session.refresh(campaign)

        assert campaign.total_videos_discovered == 10
        assert campaign.total_videos_relevant == 3
        # Progress should be calculated (3 / default_limit * 100)
        assert campaign.progress_percent > 0.0

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_record_error(self, db_session):
        """
        record_error() should increment the error count and set the error message.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Error Campaign", topic="travel vlog"
        )
        campaign = await service.start_campaign(campaign.id)

        campaign.record_error("Transcript fetch failed")
        campaign.record_error("Another error")
        await db_session.commit()
        await db_session.refresh(campaign)

        assert campaign.error_count == 2
        assert campaign.error_message == "Another error"

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_get_summary(self, db_session):
        """
        get_summary() should return a dictionary with all expected keys.
        """
        service = TopicDiscoveryService(db_session)
        campaign = await service.create_campaign(
            name="Summary Campaign", topic="fitness content"
        )

        summary = campaign.get_summary()

        expected_keys = {
            "id", "name", "topic", "status", "progress_percent",
            "videos_discovered", "videos_relevant", "videos_filtered",
            "channels_explored", "transcripts_extracted",
            "filter_acceptance_rate", "api_calls", "llm_calls",
            "duration_seconds", "started_at", "completed_at",
            "estimated_completion_at", "error_count", "config",
        }
        assert expected_keys.issubset(set(summary.keys()))
        assert summary["status"] == CampaignStatus.DRAFT.value
        assert summary["id"] == str(campaign.id)


# Avoid a name clash with unittest.mock.MagicMock used inline in the cancel test
from unittest.mock import MagicMock  # noqa: E402 (import at end is fine for tests)
