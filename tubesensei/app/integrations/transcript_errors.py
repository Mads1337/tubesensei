import logging
from typing import Optional, Any, Dict
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    TooManyRequests,
    NoTranscriptAvailable
)

from ..utils.exceptions import YouTubeAPIError

logger = logging.getLogger(__name__)


class TranscriptError(YouTubeAPIError):
    """Base exception for transcript-related errors."""
    
    def __init__(
        self,
        message: str,
        video_id: Optional[str] = None,
        language: Optional[str] = None,
        retry_after: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.video_id = video_id
        self.language = language
        self.retry_after = retry_after
        self.original_error = original_error
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/storage."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "video_id": self.video_id,
            "language": self.language,
            "retry_after": self.retry_after
        }


class TranscriptNotAvailableError(TranscriptError):
    """Raised when no transcript is available for a video."""
    
    def __init__(
        self,
        video_id: str,
        available_languages: Optional[list] = None,
        original_error: Optional[Exception] = None
    ):
        message = f"No transcript available for video {video_id}"
        if available_languages:
            message += f". Available languages: {', '.join(available_languages)}"
        super().__init__(message, video_id=video_id, original_error=original_error)
        self.available_languages = available_languages or []


class TranscriptDisabledError(TranscriptError):
    """Raised when transcripts are disabled for a video."""
    
    def __init__(self, video_id: str, original_error: Optional[Exception] = None):
        message = f"Transcripts are disabled for video {video_id}"
        super().__init__(message, video_id=video_id, original_error=original_error)


class TranscriptTimeoutError(TranscriptError):
    """Raised when transcript extraction times out."""
    
    def __init__(
        self,
        video_id: str,
        timeout_seconds: int,
        original_error: Optional[Exception] = None
    ):
        message = f"Transcript extraction timed out after {timeout_seconds} seconds for video {video_id}"
        super().__init__(message, video_id=video_id, original_error=original_error)
        self.timeout_seconds = timeout_seconds


class TranscriptLanguageError(TranscriptError):
    """Raised when requested language is not available."""
    
    def __init__(
        self,
        video_id: str,
        requested_language: str,
        available_languages: Optional[list] = None,
        original_error: Optional[Exception] = None
    ):
        message = f"Language '{requested_language}' not available for video {video_id}"
        if available_languages:
            message += f". Available: {', '.join(available_languages)}"
        super().__init__(
            message,
            video_id=video_id,
            language=requested_language,
            original_error=original_error
        )
        self.requested_language = requested_language
        self.available_languages = available_languages or []


class TranscriptRateLimitError(TranscriptError):
    """Raised when rate limited by YouTube."""
    
    def __init__(
        self,
        video_id: Optional[str] = None,
        retry_after: Optional[int] = None,
        original_error: Optional[Exception] = None
    ):
        message = "Rate limited by YouTube transcript API"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(
            message,
            video_id=video_id,
            retry_after=retry_after,
            original_error=original_error
        )


class TranscriptVideoUnavailableError(TranscriptError):
    """Raised when video is unavailable (private, deleted, etc.)."""
    
    def __init__(self, video_id: str, reason: Optional[str] = None, original_error: Optional[Exception] = None):
        message = f"Video {video_id} is unavailable"
        if reason:
            message += f": {reason}"
        super().__init__(message, video_id=video_id, original_error=original_error)
        self.reason = reason


class TranscriptProcessingError(TranscriptError):
    """Raised when transcript processing fails."""
    
    def __init__(
        self,
        video_id: str,
        stage: str,
        details: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        message = f"Failed to process transcript for video {video_id} at stage: {stage}"
        if details:
            message += f". Details: {details}"
        super().__init__(message, video_id=video_id, original_error=original_error)
        self.stage = stage
        self.details = details


class TranscriptQualityError(TranscriptError):
    """Raised when transcript quality is below threshold."""
    
    def __init__(
        self,
        video_id: str,
        quality_score: float,
        threshold: float,
        reason: Optional[str] = None
    ):
        message = f"Transcript quality ({quality_score:.2f}) below threshold ({threshold:.2f}) for video {video_id}"
        if reason:
            message += f": {reason}"
        super().__init__(message, video_id=video_id)
        self.quality_score = quality_score
        self.threshold = threshold
        self.reason = reason


async def handle_transcript_error(
    error: Exception,
    video_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> TranscriptError:
    """
    Convert YouTube transcript API errors to our custom exceptions.
    
    Args:
        error: The original exception
        video_id: YouTube video ID if available
        context: Additional context for error handling
        
    Returns:
        Appropriate TranscriptError subclass
    """
    context = context or {}
    
    # Handle youtube-transcript-api specific errors
    if isinstance(error, TranscriptsDisabled):
        logger.warning(f"Transcripts disabled for video {video_id}")
        return TranscriptDisabledError(video_id=video_id, original_error=error)
    
    elif isinstance(error, NoTranscriptFound):
        logger.warning(f"No transcript found for video {video_id}")
        available_languages = context.get('available_languages', [])
        return TranscriptNotAvailableError(
            video_id=video_id,
            available_languages=available_languages,
            original_error=error
        )
    
    elif isinstance(error, NoTranscriptAvailable):
        logger.warning(f"No transcript available for video {video_id}")
        return TranscriptNotAvailableError(video_id=video_id, original_error=error)
    
    elif isinstance(error, VideoUnavailable):
        logger.warning(f"Video unavailable: {video_id}")
        reason = str(error) if str(error) else None
        return TranscriptVideoUnavailableError(
            video_id=video_id,
            reason=reason,
            original_error=error
        )
    
    elif isinstance(error, TooManyRequests):
        logger.warning(f"Rate limited for video {video_id}")
        retry_after = context.get('retry_after', 60)
        return TranscriptRateLimitError(
            video_id=video_id,
            retry_after=retry_after,
            original_error=error
        )
    
    elif isinstance(error, TimeoutError):
        timeout = context.get('timeout_seconds', 300)
        logger.error(f"Transcript extraction timed out for video {video_id}")
        return TranscriptTimeoutError(
            video_id=video_id,
            timeout_seconds=timeout,
            original_error=error
        )
    
    # Handle our custom errors
    elif isinstance(error, TranscriptError):
        return error
    
    # Generic error handling
    else:
        logger.error(f"Unexpected transcript error for video {video_id}: {error}")
        return TranscriptProcessingError(
            video_id=video_id,
            stage="unknown",
            details=str(error),
            original_error=error
        )


def is_retryable_error(error: TranscriptError) -> bool:
    """
    Determine if an error is retryable.
    
    Args:
        error: The transcript error
        
    Returns:
        True if the error is retryable
    """
    # These errors are typically permanent
    non_retryable = (
        TranscriptDisabledError,
        TranscriptNotAvailableError,
        TranscriptVideoUnavailableError,
        TranscriptQualityError
    )
    
    # These errors might be temporary
    retryable = (
        TranscriptTimeoutError,
        TranscriptRateLimitError,
        TranscriptProcessingError
    )
    
    if isinstance(error, non_retryable):
        return False
    elif isinstance(error, retryable):
        return True
    else:
        # Default to retryable for unknown errors
        return True


def get_retry_delay(error: TranscriptError, attempt: int = 1) -> int:
    """
    Calculate retry delay based on error type and attempt number.
    
    Args:
        error: The transcript error
        attempt: Retry attempt number (1-based)
        
    Returns:
        Delay in seconds before retry
    """
    if isinstance(error, TranscriptRateLimitError) and error.retry_after:
        return error.retry_after
    
    # Exponential backoff: 2^attempt * base_delay
    base_delay = 5
    max_delay = 300  # 5 minutes
    
    if isinstance(error, TranscriptTimeoutError):
        base_delay = 10  # Longer delay for timeouts
    elif isinstance(error, TranscriptRateLimitError):
        base_delay = 30  # Even longer for rate limits
    
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    return delay