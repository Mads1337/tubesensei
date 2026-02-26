"""
Unit tests for the IdeaService.

Tests list_ideas(), get_idea(), create_idea(), bulk_update(), and export_ideas().
"""
import pytest
from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy import select

from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.models.idea import Idea, IdeaStatus, IdeaPriority
from app.models.user import User, UserRole, UserStatus
from app.services.idea_service import IdeaService
from app.core.exceptions import NotFoundException


async def _create_user(db_session, suffix="001") -> User:
    """Helper to create a minimal test user."""
    user = User(
        email=f"test{suffix}@example.com",
        username=f"testuser{suffix}",
        hashed_password="$2b$12$fakehashfortest",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
        is_active=True,
        is_verified=True,
        is_superuser=False,
        two_factor_enabled=False,
        login_attempts=0,
    )
    db_session.add(user)
    await db_session.flush()
    return user


async def _create_channel(db_session, suffix="001") -> Channel:
    """Helper to create a minimal channel."""
    channel = Channel(
        youtube_channel_id=f"UCsvc{suffix}",
        name=f"Service Test Channel {suffix}",
        status=ChannelStatus.ACTIVE,
    )
    db_session.add(channel)
    await db_session.flush()
    return channel


async def _create_video(db_session, channel: Channel, suffix="001") -> Video:
    """Helper to create a minimal video."""
    video = Video(
        youtube_video_id=f"svc{suffix}vid",
        channel_id=channel.id,
        title=f"Service Test Video {suffix}",
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
    return video


async def _create_idea(
    db_session,
    video: Video,
    title: str = "Test Idea",
    status: IdeaStatus = IdeaStatus.EXTRACTED,
    confidence_score: float = 0.7,
    category: str = "SaaS",
) -> Idea:
    """Helper to create a minimal idea."""
    idea = Idea(
        video_id=video.id,
        title=title,
        description=f"Description for {title}",
        category=category,
        status=status,
        priority=IdeaPriority.MEDIUM,
        confidence_score=confidence_score,
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


class TestIdeaServiceGetIdea:
    """Tests for IdeaService.get_idea()."""

    @pytest.mark.asyncio
    async def test_get_idea_found(self, db_session):
        """get_idea should return the idea when it exists."""
        channel = await _create_channel(db_session)
        video = await _create_video(db_session, channel)
        idea = await _create_idea(db_session, video)
        await db_session.commit()

        service = IdeaService(db_session)
        fetched = await service.get_idea(str(idea.id))

        assert fetched.id == idea.id
        assert fetched.title == idea.title

    @pytest.mark.asyncio
    async def test_get_idea_with_uuid(self, db_session):
        """get_idea should accept UUID objects as well as strings."""
        channel = await _create_channel(db_session, "002")
        video = await _create_video(db_session, channel, "002")
        idea = await _create_idea(db_session, video, title="UUID Idea")
        await db_session.commit()

        service = IdeaService(db_session)
        fetched = await service.get_idea(idea.id)  # Pass UUID directly
        assert fetched.id == idea.id

    @pytest.mark.asyncio
    async def test_get_idea_not_found_raises(self, db_session):
        """get_idea should raise NotFoundException when idea doesn't exist."""
        await db_session.commit()
        service = IdeaService(db_session)

        with pytest.raises(NotFoundException):
            await service.get_idea(str(uuid4()))


class TestIdeaServiceCreateIdea:
    """Tests for IdeaService.create_idea()."""

    @pytest.mark.asyncio
    async def test_create_idea_basic(self, db_session):
        """create_idea should persist and return a new Idea."""
        channel = await _create_channel(db_session, "003")
        video = await _create_video(db_session, channel, "003")
        await db_session.commit()

        service = IdeaService(db_session)
        data = {
            "video_id": video.id,
            "title": "New Idea via Service",
            "description": "Created by the service layer",
            "category": "EdTech",
            "confidence_score": 0.9,
            "tags": ["education", "AI"],
            "technologies": ["Python"],
            "potential_challenges": [],
            "monetization_strategies": [],
        }
        idea = await service.create_idea(data)

        assert idea.id is not None
        assert idea.title == "New Idea via Service"
        assert idea.status == IdeaStatus.EXTRACTED
        assert idea.category == "EdTech"
        assert idea.confidence_score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_create_idea_default_status(self, db_session):
        """create_idea should always set status to EXTRACTED."""
        channel = await _create_channel(db_session, "004")
        video = await _create_video(db_session, channel, "004")
        await db_session.commit()

        service = IdeaService(db_session)
        data = {
            "video_id": video.id,
            "title": "Default Status Idea",
            "description": "Testing default status from service",
        }
        idea = await service.create_idea(data)
        assert idea.status == IdeaStatus.EXTRACTED


class TestIdeaServiceBulkUpdate:
    """Tests for IdeaService.bulk_update()."""

    @pytest.mark.asyncio
    async def test_bulk_update_select(self, db_session):
        """bulk_update with 'select' action should mark ideas as SELECTED."""
        user = await _create_user(db_session, "005")
        channel = await _create_channel(db_session, "005")
        video = await _create_video(db_session, channel, "005")
        idea1 = await _create_idea(db_session, video, title="Bulk Idea 1")
        idea2 = await _create_idea(db_session, video, title="Bulk Idea 2")
        await db_session.commit()

        service = IdeaService(db_session)
        result = await service.bulk_update(
            idea_ids=[str(idea1.id), str(idea2.id)],
            action="select",
            user_id=str(user.id),
        )

        assert result["updated"] == 2
        assert len(result["errors"]) == 0

        # Verify DB state
        updated_idea1 = await service.get_idea(str(idea1.id))
        assert updated_idea1.status == IdeaStatus.SELECTED

    @pytest.mark.asyncio
    async def test_bulk_update_reject(self, db_session):
        """bulk_update with 'reject' action should mark ideas as REJECTED."""
        user = await _create_user(db_session, "006")
        channel = await _create_channel(db_session, "006")
        video = await _create_video(db_session, channel, "006")
        idea = await _create_idea(db_session, video, title="To Reject")
        await db_session.commit()

        service = IdeaService(db_session)
        result = await service.bulk_update(
            idea_ids=[str(idea.id)],
            action="reject",
            user_id=str(user.id),
        )

        assert result["updated"] == 1
        updated = await service.get_idea(str(idea.id))
        assert updated.status == IdeaStatus.REJECTED

    @pytest.mark.asyncio
    async def test_bulk_update_review(self, db_session):
        """bulk_update with 'review' action should mark ideas as REVIEWED."""
        user = await _create_user(db_session, "007")
        channel = await _create_channel(db_session, "007")
        video = await _create_video(db_session, channel, "007")
        idea = await _create_idea(db_session, video, title="To Review")
        await db_session.commit()

        service = IdeaService(db_session)
        result = await service.bulk_update(
            idea_ids=[str(idea.id)],
            action="review",
            user_id=str(user.id),
        )

        assert result["updated"] == 1
        updated = await service.get_idea(str(idea.id))
        assert updated.status == IdeaStatus.REVIEWED

    @pytest.mark.asyncio
    async def test_bulk_update_non_existent_idea_reports_error(self, db_session):
        """bulk_update with a non-existent ID should report an error, not crash."""
        await db_session.commit()

        service = IdeaService(db_session)
        result = await service.bulk_update(
            idea_ids=[str(uuid4())],  # Non-existent
            action="select",
        )

        assert result["updated"] == 0
        assert len(result["errors"]) == 1


class TestIdeaServiceMarkAsExported:
    """Tests that mark_as_exported is called during export."""

    @pytest.mark.asyncio
    async def test_export_ideas_json_format(self, db_session):
        """export_ideas in JSON format should return correct structure."""
        channel = await _create_channel(db_session, "008")
        video = await _create_video(db_session, channel, "008")
        idea = await _create_idea(db_session, video, title="Export Idea")
        await db_session.commit()

        service = IdeaService(db_session)
        result = await service.export_ideas(
            idea_ids=[str(idea.id)],
            format="json",
        )

        assert result["format"] == "json"
        assert result["count"] == 1
        assert "data" in result
        assert "exported_at" in result

    @pytest.mark.asyncio
    async def test_export_ideas_increments_export_count(self, db_session):
        """Exporting an idea should increment its export_count."""
        channel = await _create_channel(db_session, "009")
        video = await _create_video(db_session, channel, "009")
        idea = await _create_idea(db_session, video, title="Countable Export Idea")
        await db_session.commit()

        assert idea.export_count == 0

        service = IdeaService(db_session)
        await service.export_ideas(idea_ids=[str(idea.id)], format="json")

        # Reload idea from DB to check export_count
        refreshed = await service.get_idea(str(idea.id))
        assert refreshed.export_count == 1

    @pytest.mark.asyncio
    async def test_export_ideas_invalid_format_raises(self, db_session):
        """export_ideas with unsupported format should raise ValueError."""
        channel = await _create_channel(db_session, "010")
        video = await _create_video(db_session, channel, "010")
        idea = await _create_idea(db_session, video, title="Format Idea")
        await db_session.commit()

        service = IdeaService(db_session)
        with pytest.raises(ValueError, match="Unsupported export format"):
            await service.export_ideas(idea_ids=[str(idea.id)], format="xml")


class TestIdeaServiceGetCategories:
    """Tests for IdeaService.get_categories()."""

    @pytest.mark.asyncio
    async def test_get_categories_returns_unique_values(self, db_session):
        """get_categories should return distinct non-None category values."""
        channel = await _create_channel(db_session, "011")
        video1 = await _create_video(db_session, channel, "011")
        video2 = await _create_video(db_session, channel, "012")
        video3 = await _create_video(db_session, channel, "013")

        await _create_idea(db_session, video1, title="SaaS Idea", category="SaaS")
        await _create_idea(db_session, video2, title="EdTech Idea", category="EdTech")
        await _create_idea(db_session, video3, title="Another SaaS", category="SaaS")
        await db_session.commit()

        service = IdeaService(db_session)
        categories = await service.get_categories()

        assert "SaaS" in categories
        assert "EdTech" in categories
        # Should be deduplicated
        assert categories.count("SaaS") == 1
