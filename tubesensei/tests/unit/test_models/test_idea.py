"""
Unit tests for the Idea model.

Tests creation, status transitions, computed properties, and helper methods.
"""
import pytest
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from app.models.idea import Idea, IdeaStatus, IdeaPriority
from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus

_NOW = datetime.now(timezone.utc)


class TestIdeaStatusEnum:
    """Tests for IdeaStatus enum."""

    def test_all_expected_values_exist(self):
        values = {s.value for s in IdeaStatus}
        assert "extracted" in values
        assert "reviewed" in values
        assert "selected" in values
        assert "rejected" in values
        assert "in_progress" in values
        assert "implemented" in values


class TestIdeaPriorityEnum:
    """Tests for IdeaPriority enum."""

    def test_all_expected_values_exist(self):
        values = {p.value for p in IdeaPriority}
        assert "low" in values
        assert "medium" in values
        assert "high" in values
        assert "critical" in values


class TestIdeaCreation:
    """Tests for Idea model creation."""

    @pytest.mark.asyncio
    async def test_basic_creation(self, db_session):
        """Test basic idea creation with required fields."""
        channel = Channel(
            youtube_channel_id="UCtest123",
            name="Test Channel",
            status=ChannelStatus.ACTIVE,
        )
        db_session.add(channel)
        await db_session.flush()

        video = Video(
            youtube_video_id="dQw4w9WgXcQ",
            channel_id=channel.id,
            title="Test Video",
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
            title="Test Idea Title",
            description="Test idea description text",
            status=IdeaStatus.EXTRACTED,
            priority=IdeaPriority.MEDIUM,
            confidence_score=0.75,
            tags=[],
            technologies=[],
            potential_challenges=[],
            monetization_strategies=[],
            related_ideas=[],
            extraction_metadata={},
        )
        db_session.add(idea)
        await db_session.commit()

        assert idea.id is not None
        assert idea.title == "Test Idea Title"  # type: ignore[comparison-overlap]
        assert idea.description == "Test idea description text"  # type: ignore[comparison-overlap]
        assert idea.status == IdeaStatus.EXTRACTED  # type: ignore[comparison-overlap]
        assert idea.priority == IdeaPriority.MEDIUM  # type: ignore[comparison-overlap]
        assert idea.confidence_score == 0.75  # type: ignore[comparison-overlap]

    @pytest.mark.asyncio
    async def test_default_status_is_extracted(self, db_session):
        """Test that default status is EXTRACTED."""
        channel = Channel(
            youtube_channel_id="UCtest456",
            name="Test Channel 2",
            status=ChannelStatus.ACTIVE,
        )
        db_session.add(channel)
        await db_session.flush()

        video = Video(
            youtube_video_id="dQw4w9WgXcR",
            channel_id=channel.id,
            title="Test Video 2",
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
            title="Default Status Idea",
            description="Testing default status",
            confidence_score=0.5,
            tags=[],
            technologies=[],
            potential_challenges=[],
            monetization_strategies=[],
            related_ideas=[],
            extraction_metadata={},
        )
        db_session.add(idea)
        await db_session.commit()

        assert idea.status == IdeaStatus.EXTRACTED  # type: ignore[comparison-overlap]

    @pytest.mark.asyncio
    async def test_optional_fields(self, db_session):
        """Test that optional fields can be None."""
        channel = Channel(
            youtube_channel_id="UCtest789",
            name="Test Channel 3",
            status=ChannelStatus.ACTIVE,
        )
        db_session.add(channel)
        await db_session.flush()

        video = Video(
            youtube_video_id="dQw4w9WgXcS",
            channel_id=channel.id,
            title="Test Video 3",
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
            title="Minimal Idea",
            description="Minimal description",
            confidence_score=0.5,
            tags=[],
            technologies=[],
            potential_challenges=[],
            monetization_strategies=[],
            related_ideas=[],
            extraction_metadata={},
        )
        db_session.add(idea)
        await db_session.commit()

        assert idea.category is None
        assert idea.complexity_score is None
        assert idea.market_size_estimate is None
        assert idea.target_audience is None
        assert idea.review_notes is None
        assert idea.reviewed_by is None
        assert idea.reviewed_at is None
        assert idea.selected_at is None


class TestIdeaComputedProperties:
    """Tests for Idea computed properties."""

    def test_is_selected_when_selected(self):
        idea = Idea(status=IdeaStatus.SELECTED, title="t", description="d", confidence_score=0.5)
        # Use explicit bool() to avoid Pyright's ColumnElement[bool] false-positive on @property
        assert bool(idea.is_selected) is True  # pyright: ignore[reportGeneralTypeIssues]

    def test_is_selected_when_not_selected(self):
        for status in [IdeaStatus.EXTRACTED, IdeaStatus.REVIEWED, IdeaStatus.REJECTED]:
            idea = Idea(status=status, title="t", description="d", confidence_score=0.5)
            assert bool(idea.is_selected) is False  # pyright: ignore[reportGeneralTypeIssues]

    def test_is_reviewed_for_reviewed_status(self):
        idea = Idea(status=IdeaStatus.REVIEWED, title="t", description="d", confidence_score=0.5)
        assert bool(idea.is_reviewed) is True  # pyright: ignore[reportGeneralTypeIssues]

    def test_is_reviewed_for_selected_status(self):
        idea = Idea(status=IdeaStatus.SELECTED, title="t", description="d", confidence_score=0.5)
        assert bool(idea.is_reviewed) is True  # pyright: ignore[reportGeneralTypeIssues]

    def test_is_reviewed_for_rejected_status(self):
        idea = Idea(status=IdeaStatus.REJECTED, title="t", description="d", confidence_score=0.5)
        assert bool(idea.is_reviewed) is True  # pyright: ignore[reportGeneralTypeIssues]

    def test_is_reviewed_for_extracted_status(self):
        idea = Idea(status=IdeaStatus.EXTRACTED, title="t", description="d", confidence_score=0.5)
        assert bool(idea.is_reviewed) is False  # pyright: ignore[reportGeneralTypeIssues]

    def test_confidence_percentage(self):
        idea = Idea(status=IdeaStatus.EXTRACTED, title="t", description="d", confidence_score=0.75)
        assert idea.confidence_percentage == 75.0

    def test_confidence_percentage_zero(self):
        idea = Idea(status=IdeaStatus.EXTRACTED, title="t", description="d", confidence_score=0.0)
        assert idea.confidence_percentage == 0.0

    def test_confidence_percentage_one(self):
        idea = Idea(status=IdeaStatus.EXTRACTED, title="t", description="d", confidence_score=1.0)
        assert idea.confidence_percentage == 100.0


class TestIdeaStatusMethods:
    """Tests for Idea status-changing methods."""

    def test_mark_as_reviewed(self):
        user_id: Any = uuid4()  # SA dialect UUID annotation; Any avoids type mismatch
        idea = Idea(
            status=IdeaStatus.EXTRACTED,
            title="t",
            description="d",
            confidence_score=0.5,
        )
        idea.mark_as_reviewed(user_id, notes="Great idea")  # type: ignore[arg-type]

        assert idea.status == IdeaStatus.REVIEWED
        assert idea.reviewed_by == user_id
        assert idea.reviewed_at is not None
        assert idea.review_notes == "Great idea"

    def test_mark_as_reviewed_without_notes(self):
        user_id: Any = uuid4()  # SA dialect UUID annotation; Any avoids type mismatch
        idea = Idea(
            status=IdeaStatus.EXTRACTED,
            title="t",
            description="d",
            confidence_score=0.5,
        )
        idea.mark_as_reviewed(user_id)  # type: ignore[arg-type]

        assert idea.status == IdeaStatus.REVIEWED
        assert idea.reviewed_by == user_id
        assert idea.review_notes is None

    def test_select(self):
        user_id: Any = uuid4()  # SA dialect UUID annotation; Any avoids type mismatch
        idea = Idea(
            status=IdeaStatus.EXTRACTED,
            title="t",
            description="d",
            confidence_score=0.5,
        )
        idea.select(user_id)  # type: ignore[arg-type]

        assert idea.status == IdeaStatus.SELECTED
        assert idea.selected_by == user_id
        assert idea.selected_at is not None

    def test_select_sets_reviewed_if_not_already(self):
        user_id: Any = uuid4()  # SA dialect UUID annotation; Any avoids type mismatch
        idea = Idea(
            status=IdeaStatus.EXTRACTED,
            title="t",
            description="d",
            confidence_score=0.5,
        )
        # reviewed_by is None at this point
        idea.select(user_id)  # type: ignore[arg-type]

        assert idea.reviewed_by == user_id
        assert idea.reviewed_at is not None

    def test_reject(self):
        user_id: Any = uuid4()  # SA dialect UUID annotation; Any avoids type mismatch
        idea = Idea(
            status=IdeaStatus.EXTRACTED,
            title="t",
            description="d",
            confidence_score=0.5,
        )
        idea.reject(user_id, reason="Not viable")  # type: ignore[arg-type]

        assert idea.status == IdeaStatus.REJECTED  # type: ignore[comparison-overlap]
        assert idea.review_notes == "Not viable"  # type: ignore[comparison-overlap]

    def test_reject_sets_reviewed_if_not_already(self):
        user_id: Any = uuid4()  # SA dialect UUID annotation; Any avoids type mismatch
        idea = Idea(
            status=IdeaStatus.EXTRACTED,
            title="t",
            description="d",
            confidence_score=0.5,
        )
        idea.reject(user_id)  # type: ignore[arg-type]

        assert idea.reviewed_by == user_id
        assert idea.reviewed_at is not None

    def test_mark_as_exported(self):
        idea = Idea(
            status=IdeaStatus.SELECTED,
            title="t",
            description="d",
            confidence_score=0.5,
            export_count=0,
        )
        idea.mark_as_exported()

        assert idea.export_count == 1  # type: ignore[comparison-overlap]
        assert idea.last_exported_at is not None

    def test_mark_as_exported_increments_count(self):
        idea = Idea(
            status=IdeaStatus.SELECTED,
            title="t",
            description="d",
            confidence_score=0.5,
            export_count=3,
        )
        idea.mark_as_exported()
        assert idea.export_count == 4  # type: ignore[comparison-overlap]

    def test_repr(self):
        idea = Idea(
            status=IdeaStatus.EXTRACTED,
            title="A very long title that exceeds fifty characters in length here",
            description="d",
            confidence_score=0.5,
        )
        repr_str = repr(idea)
        assert "Idea" in repr_str
        assert "extracted" in repr_str
