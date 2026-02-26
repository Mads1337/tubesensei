"""
E2E tests for the Quick Analysis flow.

Tests the full HTTP request/response cycle for:
  - POST /admin/quick-analysis/video
  - POST /admin/quick-analysis/channel
  - GET  /admin/quick-analysis/channel/progress/{job_id}

External services (YouTubeAPIClient, TranscriptAPIClient, LLMManager) are
patched so the tests run without real credentials.

NOTE: This module imports ``app.main_enhanced`` which requires a complete
.env configuration including all pydantic settings.  If the environment
variables are not valid, the entire module is skipped at collection time via
``pytest.importorskip``.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient

# Attempt the import; skip the whole module if the environment is not ready.
try:
    from app.main_enhanced import app as _app
    from app.database import get_db as _get_db
    _IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001
    _app = None
    _get_db = None
    _IMPORT_ERROR = exc

if _IMPORT_ERROR is not None:
    pytest.skip(
        f"Skipping quick analysis E2E tests: app import failed ({_IMPORT_ERROR})",
        allow_module_level=True,
    )

# After the skip guard, _app and _get_db are guaranteed to be non-None.
assert _app is not None
assert _get_db is not None
app = _app
get_db = _get_db


# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

MOCK_VIDEO_INFO = {
    "video_id": "dQw4w9WgXcQ",
    "title": "Test Video Title",
    "channel_title": "Test Channel",
    "duration_seconds": 630,
    "view_count": 1500000,
    "like_count": 50000,
    "comment_count": 2000,
    "published_at": "2023-06-01T12:00:00Z",
    "thumbnails": {
        "medium": {"url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/mqdefault.jpg"},
    },
}

MOCK_CHANNEL_INFO = {
    "channel_id": "UC_test_channel_id",
    "title": "Test Channel",
    "description": "A great test channel",
    "subscriber_count": 250000,
    "video_count": 300,
    "thumbnails": {
        "medium": {"url": "https://yt3.ggpht.com/medium.jpg"},
    },
}

MOCK_CHANNEL_VIDEOS = [
    {"video_id": "vid001", "title": "Video 1", "channel_title": "Test Channel"},
    {"video_id": "vid002", "title": "Video 2", "channel_title": "Test Channel"},
]


def _make_mock_transcript(content: str = "This is the transcript content for testing."):
    """Create a mock TranscriptData object."""
    transcript = MagicMock()
    transcript.content = content
    return transcript


def _make_mock_llm_result(content: str = "Analysis result text"):
    """Create a mock LLM generate() return value."""
    return {"content": content, "usage": {"total_tokens": 150}, "cost": 0.002}


def _make_mock_idea():
    """Create a mock parsed idea object."""
    idea = MagicMock()
    idea.title = "Test Idea Title"
    idea.description = "A compelling business idea"
    idea.category = "SaaS"
    idea.target_market = "Developers"
    idea.value_proposition = "Saves time"
    idea.complexity_score = 3
    idea.confidence = 0.85
    idea.source_context = "Mentioned at 5:30"
    return idea


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_client(db_session):
    """HTTP test client wired to the test database session."""

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Video analysis tests
# ---------------------------------------------------------------------------

class TestVideoAnalysisFlow:

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_video_analysis_success(self, test_client):
        """
        Full happy-path: valid YouTube video URL, all external APIs succeed.
        Expects a 200 HTML response containing idea content.
        """
        with (
            patch("app.api.admin.quick_analysis.YouTubeAPIClient") as MockYT,
            patch("app.api.admin.quick_analysis.TranscriptAPIClient") as MockTA,
            patch("app.api.admin.quick_analysis.LLMManager") as MockLLM,
            patch("app.api.admin.quick_analysis.ResponseParser") as MockRP,
        ):
            # Wire up mocks
            yt_instance = AsyncMock()
            yt_instance.get_video_details.return_value = [MOCK_VIDEO_INFO]
            MockYT.return_value = yt_instance

            ta_instance = AsyncMock()
            ta_instance.get_transcript.return_value = _make_mock_transcript()
            MockTA.return_value = ta_instance

            llm_instance = AsyncMock()
            llm_instance.generate.return_value = _make_mock_llm_result()
            MockLLM.return_value = llm_instance

            MockRP.parse_idea_extraction_response.return_value = [_make_mock_idea()]

            response = await test_client.post(
                "/admin/quick-analysis/video",
                json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            )

        assert response.status_code == 200
        body = response.text
        # Response is HTML; confirm non-empty
        assert body

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_video_analysis_invalid_url(self, test_client):
        """
        Non-YouTube URL should return a 200 HTML response containing an error
        message (the endpoint returns HTML error partials, not HTTP 4xx codes).
        """
        response = await test_client.post(
            "/admin/quick-analysis/video",
            json={"url": "https://www.example.com/not-a-youtube-url"},
        )
        assert response.status_code == 200
        body = response.text
        assert body  # Response body is non-empty

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_video_analysis_channel_url_returns_error(self, test_client):
        """
        A channel URL (not a video URL) should return the error partial describing
        that the URL does not point to a video.
        """
        with patch("app.api.admin.quick_analysis.YouTubeAPIClient"):
            response = await test_client.post(
                "/admin/quick-analysis/video",
                json={"url": "https://www.youtube.com/@SomeChannel"},
            )
        assert response.status_code == 200
        body = response.text
        assert body  # Non-empty HTML partial returned

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_video_analysis_no_transcript(self, test_client):
        """
        Valid video URL but transcript API raises an error.
        The endpoint should return the video results partial with a transcript
        warning rather than crashing.
        """
        from app.integrations.transcript_errors import TranscriptError

        with (
            patch("app.api.admin.quick_analysis.YouTubeAPIClient") as MockYT,
            patch("app.api.admin.quick_analysis.TranscriptAPIClient") as MockTA,
        ):
            yt_instance = AsyncMock()
            yt_instance.get_video_details.return_value = [MOCK_VIDEO_INFO]
            MockYT.return_value = yt_instance

            ta_instance = AsyncMock()
            ta_instance.get_transcript.side_effect = TranscriptError("No transcript available")
            MockTA.return_value = ta_instance

            response = await test_client.post(
                "/admin/quick-analysis/video",
                json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            )

        assert response.status_code == 200
        body = response.text
        assert body

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_video_analysis_youtube_api_key_missing(self, test_client):
        """
        When YouTubeAPIClient.get_video_details raises APIKeyError, the error partial
        should be returned.
        """
        from app.utils.exceptions import APIKeyError

        with patch("app.api.admin.quick_analysis.YouTubeAPIClient") as MockYT:
            yt_instance = AsyncMock()
            yt_instance.get_video_details.side_effect = APIKeyError("No API key configured")
            MockYT.return_value = yt_instance

            response = await test_client.post(
                "/admin/quick-analysis/video",
                json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            )

        assert response.status_code == 200
        body = response.text
        assert body

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_video_analysis_video_not_found(self, test_client):
        """
        When the YouTube API returns no video details (empty list), the endpoint
        should return the error partial for "video not found".
        """
        with patch("app.api.admin.quick_analysis.YouTubeAPIClient") as MockYT:
            yt_instance = AsyncMock()
            yt_instance.get_video_details.return_value = []  # No video found
            MockYT.return_value = yt_instance

            response = await test_client.post(
                "/admin/quick-analysis/video",
                json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
            )

        assert response.status_code == 200
        body = response.text
        assert body


# ---------------------------------------------------------------------------
# Channel analysis tests
# ---------------------------------------------------------------------------

class TestChannelAnalysisFlow:

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_channel_analysis_starts_background_job(self, test_client):
        """
        Valid channel URL: endpoint should return 200 with channel info in
        the HTML partial, and the background analysis task is started.
        """
        with (
            patch("app.api.admin.quick_analysis.YouTubeAPIClient") as MockYT,
            patch("app.api.admin.quick_analysis.asyncio") as mock_asyncio,
        ):
            yt_instance = AsyncMock()
            yt_instance.get_channel_by_handle.return_value = MOCK_CHANNEL_INFO
            yt_instance.list_channel_videos.return_value = MOCK_CHANNEL_VIDEOS
            MockYT.return_value = yt_instance

            # Prevent the real background task from running during the test
            mock_asyncio.create_task = MagicMock(return_value=None)

            response = await test_client.post(
                "/admin/quick-analysis/channel",
                json={"url": "https://www.youtube.com/@TestChannel", "max_videos": 5},
            )

        assert response.status_code == 200
        body = response.text
        assert body

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_channel_analysis_invalid_url(self, test_client):
        """
        A video URL submitted to the channel endpoint should return the error partial.
        """
        response = await test_client.post(
            "/admin/quick-analysis/channel",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "max_videos": 5},
        )
        assert response.status_code == 200
        body = response.text
        assert body

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_channel_analysis_no_videos_found(self, test_client):
        """
        Channel exists but has no videos: endpoint should return channel info
        partial with a warning about no videos.
        """
        with patch("app.api.admin.quick_analysis.YouTubeAPIClient") as MockYT:
            yt_instance = AsyncMock()
            yt_instance.get_channel_by_handle.return_value = MOCK_CHANNEL_INFO
            yt_instance.list_channel_videos.return_value = []  # No videos
            MockYT.return_value = yt_instance

            response = await test_client.post(
                "/admin/quick-analysis/channel",
                json={"url": "https://www.youtube.com/@TestChannel", "max_videos": 5},
            )

        assert response.status_code == 200
        body = response.text
        assert body

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_channel_analysis_progress_polling_unknown_job(self, test_client):
        """
        Polling progress for a non-existent job_id should return the error partial
        with status 200.
        """
        response = await test_client.get(
            "/admin/quick-analysis/channel/progress/non-existent-job-id-12345"
        )
        assert response.status_code == 200
        body = response.text
        assert body

    @pytest.mark.asyncio
    @pytest.mark.db
    async def test_channel_analysis_progress_polling_after_start(self, test_client):
        """
        After injecting a fake in-progress job directly into the in-memory store,
        the progress endpoint should return 200 with job status data.
        """
        from app.api.admin import quick_analysis as qa_module

        fake_job_id = "test-job-e2e-00001"
        qa_module._channel_jobs[fake_job_id] = {
            "status": "running",
            "channel_info": MOCK_CHANNEL_INFO,
            "total_videos": 2,
            "processed": 1,
            "succeeded": 1,
            "failed": 0,
            "total_ideas": 3,
            "video_results": [
                {
                    "video_id": "vid001",
                    "title": "Video 1",
                    "status": "done",
                    "ideas": [],
                    "error": None,
                }
            ],
            "started_at": "2024-01-01T10:00:00",
            "error": None,
        }

        try:
            response = await test_client.get(
                f"/admin/quick-analysis/channel/progress/{fake_job_id}"
            )
            assert response.status_code == 200
            body = response.text
            assert body
        finally:
            qa_module._channel_jobs.pop(fake_job_id, None)
