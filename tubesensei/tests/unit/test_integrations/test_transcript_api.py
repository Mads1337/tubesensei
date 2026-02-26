"""
Unit tests for TranscriptAPIClient integration.

Tests the transcript API client including initialization, error handling,
and data processing logic that can be tested without network access.
"""

from unittest.mock import patch

from app.integrations.transcript_api import TranscriptAPIClient
from app.integrations.transcript_errors import (
    TranscriptNotAvailableError,
    TranscriptDisabledError,
    TranscriptRateLimitError,
    TranscriptTimeoutError,
    TranscriptLanguageError,
    is_retryable_error,
    get_retry_delay,
)


class TestTranscriptAPIClientInit:
    """Test suite for TranscriptAPIClient initialization."""

    @patch("app.integrations.transcript_api.YouTubeTranscriptApi")
    @patch("app.integrations.transcript_api.settings")
    def test_init_defaults(self, mock_settings, _mock_yt_api_unused):
        """Test default initialization uses settings values."""
        mock_settings.TRANSCRIPT_TIMEOUT_SECONDS = 30
        mock_settings.TRANSCRIPT_MAX_RETRIES = 3
        mock_settings.TRANSCRIPT_PREFERRED_LANGUAGES = ["en"]
        mock_settings.TRANSCRIPT_PROXY_ENABLED = False

        client = TranscriptAPIClient()

        assert client.timeout_seconds == 30
        assert client.max_retries == 3
        assert client.preferred_languages == ["en"]

    @patch("app.integrations.transcript_api.YouTubeTranscriptApi")
    @patch("app.integrations.transcript_api.settings")
    def test_init_with_custom_values(self, mock_settings, _mock_yt_api):
        """Test initialization with custom values overrides settings."""
        mock_settings.TRANSCRIPT_TIMEOUT_SECONDS = 30
        mock_settings.TRANSCRIPT_MAX_RETRIES = 3
        mock_settings.TRANSCRIPT_PREFERRED_LANGUAGES = ["en"]
        mock_settings.TRANSCRIPT_PROXY_ENABLED = False

        client = TranscriptAPIClient(
            timeout_seconds=60,
            max_retries=5,
            preferred_languages=["en", "es"],
        )

        assert client.timeout_seconds == 60
        assert client.max_retries == 5
        assert client.preferred_languages == ["en", "es"]

    @patch("app.integrations.transcript_api.YouTubeTranscriptApi")
    @patch("app.integrations.transcript_api.settings")
    def test_statistics_start_at_zero(self, mock_settings, _mock_yt_api):
        """Test that statistics counters start at zero."""
        mock_settings.TRANSCRIPT_TIMEOUT_SECONDS = 30
        mock_settings.TRANSCRIPT_MAX_RETRIES = 3
        mock_settings.TRANSCRIPT_PREFERRED_LANGUAGES = ["en"]
        mock_settings.TRANSCRIPT_PROXY_ENABLED = False

        client = TranscriptAPIClient()
        stats = client.get_statistics()

        assert stats["total_requests"] == 0
        assert stats["successful_requests"] == 0
        assert stats["failed_requests"] == 0
        assert stats["success_rate"] == 0


class TestTranscriptAPIClientStatistics:
    """Test suite for TranscriptAPIClient statistics tracking."""

    @patch("app.integrations.transcript_api.YouTubeTranscriptApi")
    @patch("app.integrations.transcript_api.settings")
    def test_get_statistics_format(self, mock_settings, _mock_yt_api):
        """Test that get_statistics returns the expected keys."""
        mock_settings.TRANSCRIPT_TIMEOUT_SECONDS = 30
        mock_settings.TRANSCRIPT_MAX_RETRIES = 3
        mock_settings.TRANSCRIPT_PREFERRED_LANGUAGES = ["en"]
        mock_settings.TRANSCRIPT_PROXY_ENABLED = False

        client = TranscriptAPIClient()
        stats = client.get_statistics()

        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "success_rate" in stats
        assert "preferred_languages" in stats


class TestTranscriptAPIClientConfidenceScore:
    """Test suite for confidence score calculation."""

    @patch("app.integrations.transcript_api.YouTubeTranscriptApi")
    @patch("app.integrations.transcript_api.settings")
    def test_confidence_score_manual_transcript(self, mock_settings, _mock_yt_api):
        """Manual transcripts should have higher confidence than auto-generated."""
        mock_settings.TRANSCRIPT_TIMEOUT_SECONDS = 30
        mock_settings.TRANSCRIPT_MAX_RETRIES = 3
        mock_settings.TRANSCRIPT_PREFERRED_LANGUAGES = ["en"]
        mock_settings.TRANSCRIPT_PROXY_ENABLED = False
        mock_settings.MIN_TRANSCRIPT_WORD_COUNT = 50

        client = TranscriptAPIClient()
        text = " ".join(["word"] * 100)

        score_manual = client._calculate_confidence_score(
            text, is_auto_generated=False, is_preferred_language=True
        )
        score_auto = client._calculate_confidence_score(
            text, is_auto_generated=True, is_preferred_language=True
        )

        assert score_manual > score_auto

    @patch("app.integrations.transcript_api.YouTubeTranscriptApi")
    @patch("app.integrations.transcript_api.settings")
    def test_confidence_score_preferred_language(self, mock_settings, _mock_yt_api):
        """Preferred language should yield higher confidence score."""
        mock_settings.TRANSCRIPT_TIMEOUT_SECONDS = 30
        mock_settings.TRANSCRIPT_MAX_RETRIES = 3
        mock_settings.TRANSCRIPT_PREFERRED_LANGUAGES = ["en"]
        mock_settings.TRANSCRIPT_PROXY_ENABLED = False
        mock_settings.MIN_TRANSCRIPT_WORD_COUNT = 50

        client = TranscriptAPIClient()
        text = " ".join(["word"] * 100)

        score_preferred = client._calculate_confidence_score(
            text, is_auto_generated=False, is_preferred_language=True
        )
        score_not_preferred = client._calculate_confidence_score(
            text, is_auto_generated=False, is_preferred_language=False
        )

        assert score_preferred > score_not_preferred

    @patch("app.integrations.transcript_api.YouTubeTranscriptApi")
    @patch("app.integrations.transcript_api.settings")
    def test_confidence_score_within_bounds(self, mock_settings, _mock_yt_api):
        """Confidence score must always be between 0 and 1."""
        mock_settings.TRANSCRIPT_TIMEOUT_SECONDS = 30
        mock_settings.TRANSCRIPT_MAX_RETRIES = 3
        mock_settings.TRANSCRIPT_PREFERRED_LANGUAGES = ["en"]
        mock_settings.TRANSCRIPT_PROXY_ENABLED = False
        mock_settings.MIN_TRANSCRIPT_WORD_COUNT = 50

        client = TranscriptAPIClient()
        text = " ".join(["word"] * 100)

        score = client._calculate_confidence_score(
            text, is_auto_generated=True, is_preferred_language=False
        )
        assert 0.0 <= score <= 1.0


class TestTranscriptErrors:
    """Test suite for transcript error classes."""

    def test_transcript_not_available_error(self):
        """Test TranscriptNotAvailableError creation."""
        err = TranscriptNotAvailableError(video_id="abc123", available_languages=["en", "es"])
        assert err.video_id == "abc123"
        assert err.available_languages == ["en", "es"]
        assert "abc123" in str(err)

    def test_transcript_disabled_error(self):
        """Test TranscriptDisabledError creation."""
        err = TranscriptDisabledError(video_id="abc123")
        assert err.video_id == "abc123"
        assert "abc123" in str(err)

    def test_transcript_language_error(self):
        """Test TranscriptLanguageError creation."""
        err = TranscriptLanguageError(
            video_id="abc123",
            requested_language="fr",
            available_languages=["en"],
        )
        assert err.video_id == "abc123"
        assert err.requested_language == "fr"
        assert err.available_languages == ["en"]

    def test_transcript_rate_limit_error(self):
        """Test TranscriptRateLimitError creation."""
        err = TranscriptRateLimitError(video_id="abc123", retry_after=60)
        assert err.video_id == "abc123"
        assert err.retry_after == 60

    def test_transcript_timeout_error(self):
        """Test TranscriptTimeoutError creation."""
        err = TranscriptTimeoutError(video_id="abc123", timeout_seconds=30)
        assert err.video_id == "abc123"
        assert err.timeout_seconds == 30

    def test_error_to_dict(self):
        """Test that errors serialize to dict correctly."""
        err = TranscriptDisabledError(video_id="abc123")
        d = err.to_dict()
        assert "error_type" in d
        assert "message" in d
        assert d["video_id"] == "abc123"


class TestTranscriptErrorHelpers:
    """Test suite for transcript error helper functions."""

    def test_is_retryable_disabled(self):
        """TranscriptDisabledError should not be retryable."""
        err = TranscriptDisabledError(video_id="abc123")
        assert is_retryable_error(err) is False

    def test_is_retryable_timeout(self):
        """TranscriptTimeoutError should be retryable."""
        err = TranscriptTimeoutError(video_id="abc123", timeout_seconds=30)
        assert is_retryable_error(err) is True

    def test_is_retryable_rate_limit(self):
        """TranscriptRateLimitError should be retryable."""
        err = TranscriptRateLimitError(video_id="abc123")
        assert is_retryable_error(err) is True

    def test_is_retryable_not_available(self):
        """TranscriptNotAvailableError should not be retryable."""
        err = TranscriptNotAvailableError(video_id="abc123")
        assert is_retryable_error(err) is False

    def test_get_retry_delay_uses_retry_after(self):
        """get_retry_delay should use retry_after if set."""
        err = TranscriptRateLimitError(video_id="abc123", retry_after=120)
        delay = get_retry_delay(err, attempt=1)
        assert delay == 120

    def test_get_retry_delay_exponential_backoff(self):
        """get_retry_delay should increase with attempt number."""
        err = TranscriptTimeoutError(video_id="abc123", timeout_seconds=30)
        delay1 = get_retry_delay(err, attempt=1)
        delay2 = get_retry_delay(err, attempt=2)
        assert delay2 > delay1
