from typing import Optional, Any, Dict


class TubeSenseiError(Exception):
    """Base exception for TubeSensei application"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class YouTubeAPIError(TubeSenseiError):
    """Base exception for YouTube API related errors"""
    pass


class QuotaExceededError(YouTubeAPIError):
    """Raised when YouTube API quota is exceeded"""
    
    def __init__(
        self, 
        message: str = "YouTube API quota exceeded",
        quota_used: Optional[int] = None,
        quota_limit: Optional[int] = None
    ):
        super().__init__(
            message=message,
            error_code="QUOTA_EXCEEDED",
            details={
                "quota_used": quota_used,
                "quota_limit": quota_limit
            }
        )


class ChannelNotFoundError(YouTubeAPIError):
    """Raised when a YouTube channel is not found"""
    
    def __init__(self, channel_id: str):
        super().__init__(
            message=f"Channel not found: {channel_id}",
            error_code="CHANNEL_NOT_FOUND",
            details={"channel_id": channel_id}
        )


class VideoNotFoundError(YouTubeAPIError):
    """Raised when a YouTube video is not found"""
    
    def __init__(self, video_id: str):
        super().__init__(
            message=f"Video not found: {video_id}",
            error_code="VIDEO_NOT_FOUND",
            details={"video_id": video_id}
        )


class RateLimitError(YouTubeAPIError):
    """Raised when rate limit is exceeded"""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None
    ):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={"retry_after_seconds": retry_after}
        )


class InvalidURLError(YouTubeAPIError):
    """Raised when a YouTube URL is invalid or unsupported"""
    
    def __init__(self, url: str, reason: Optional[str] = None):
        message = f"Invalid YouTube URL: {url}"
        if reason:
            message += f" - {reason}"
        super().__init__(
            message=message,
            error_code="INVALID_URL",
            details={"url": url, "reason": reason}
        )


class APIKeyError(YouTubeAPIError):
    """Raised when YouTube API key is invalid or missing"""
    
    def __init__(self, message: str = "YouTube API key is invalid or missing"):
        super().__init__(
            message=message,
            error_code="API_KEY_ERROR"
        )


class NetworkError(YouTubeAPIError):
    """Raised when network-related errors occur"""
    
    def __init__(
        self, 
        message: str = "Network error occurred",
        original_error: Optional[Exception] = None
    ):
        super().__init__(
            message=message,
            error_code="NETWORK_ERROR",
            details={"original_error": str(original_error) if original_error else None}
        )


class ProcessingError(TubeSenseiError):
    """Raised when processing errors occur"""
    
    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        step: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            details={
                "job_id": job_id,
                "step": step
            }
        )


class ValidationError(TubeSenseiError):
    """Raised when validation errors occur"""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None
    ):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": value
            }
        )