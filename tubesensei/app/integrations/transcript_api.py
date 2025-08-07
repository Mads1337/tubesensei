import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    TooManyRequests
)

from ..config import settings
from ..models.transcript_data import (
    TranscriptData,
    TranscriptSegment,
    TranscriptInfo
)
from .transcript_errors import (
    TranscriptError,
    TranscriptNotAvailableError,
    TranscriptDisabledError,
    TranscriptVideoUnavailableError,
    TranscriptRateLimitError,
    TranscriptTimeoutError,
    TranscriptLanguageError,
    handle_transcript_error,
    is_retryable_error,
    get_retry_delay
)

logger = logging.getLogger(__name__)


class TranscriptAPIClient:
    """
    Async-compatible client for YouTube transcript extraction.
    Handles rate limiting, retries, and language preferences.
    """
    
    def __init__(
        self,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        preferred_languages: Optional[List[str]] = None
    ):
        self.timeout_seconds = timeout_seconds or settings.TRANSCRIPT_TIMEOUT_SECONDS
        self.max_retries = max_retries or settings.TRANSCRIPT_MAX_RETRIES
        self.preferred_languages = preferred_languages or settings.TRANSCRIPT_PREFERRED_LANGUAGES
        
        # Track rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.5  # Minimum seconds between requests
        
        # Statistics
        self._request_count = 0
        self._success_count = 0
        self._error_count = 0
        
        logger.info(
            f"Initialized TranscriptAPIClient with timeout={self.timeout_seconds}s, "
            f"max_retries={self.max_retries}, languages={self.preferred_languages}"
        )
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            wait_time = self._min_request_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self._last_request_time = time.time()
    
    async def _execute_with_timeout(
        self,
        func,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with timeout in an async context.
        
        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
            
        Raises:
            TranscriptTimeoutError: If execution times out
        """
        try:
            # Run in executor since youtube-transcript-api is synchronous
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, func, *args, **kwargs),
                timeout=self.timeout_seconds
            )
            return result
        except asyncio.TimeoutError as e:
            video_id = kwargs.get('video_id') or (args[0] if args else 'unknown')
            raise TranscriptTimeoutError(
                video_id=video_id,
                timeout_seconds=self.timeout_seconds,
                original_error=e
            )
    
    async def get_transcript(
        self,
        youtube_video_id: str,
        languages: Optional[List[str]] = None
    ) -> TranscriptData:
        """
        Get transcript for a YouTube video.
        
        Args:
            youtube_video_id: YouTube video ID
            languages: Preferred languages (overrides default)
            
        Returns:
            TranscriptData object with content and metadata
            
        Raises:
            Various TranscriptError subclasses
        """
        await self._rate_limit()
        self._request_count += 1
        
        languages = languages or self.preferred_languages
        logger.info(f"Fetching transcript for video {youtube_video_id} with languages {languages}")
        
        try:
            # Try to get transcript with preferred languages
            transcript_list = await self._execute_with_timeout(
                YouTubeTranscriptApi.list_transcripts,
                youtube_video_id
            )
            
            # Try to find transcript in preferred languages
            transcript = None
            selected_language = None
            is_auto_generated = True
            
            # First try manual transcripts in preferred languages
            for lang in languages:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    selected_language = lang
                    is_auto_generated = False
                    logger.info(f"Found manual transcript in {lang}")
                    break
                except NoTranscriptFound:
                    continue
            
            # If no manual transcript, try auto-generated
            if transcript is None:
                for lang in languages:
                    try:
                        transcript = transcript_list.find_generated_transcript([lang])
                        selected_language = lang
                        is_auto_generated = True
                        logger.info(f"Found auto-generated transcript in {lang}")
                        break
                    except NoTranscriptFound:
                        continue
            
            # If still no transcript in preferred languages, try any available
            if transcript is None:
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    # Prefer manual over auto-generated
                    for t in available_transcripts:
                        if not t.is_generated:
                            transcript = t
                            selected_language = t.language_code
                            is_auto_generated = False
                            logger.info(f"Using available manual transcript in {selected_language}")
                            break
                    
                    # Use auto-generated if no manual available
                    if transcript is None:
                        transcript = available_transcripts[0]
                        selected_language = transcript.language_code
                        is_auto_generated = transcript.is_generated
                        logger.info(f"Using available transcript in {selected_language}")
            
            if transcript is None:
                # Get available languages for error message
                available_languages = [t.language_code for t in transcript_list]
                raise TranscriptNotAvailableError(
                    video_id=youtube_video_id,
                    available_languages=available_languages
                )
            
            # Fetch the actual transcript content
            transcript_data = await self._execute_with_timeout(
                transcript.fetch
            )
            
            # Convert to our data model
            segments = [
                TranscriptSegment(
                    text=segment['text'],
                    start=segment['start'],
                    duration=segment['duration']
                )
                for segment in transcript_data
            ]
            
            # Combine all text
            full_text = ' '.join(segment['text'] for segment in transcript_data)
            
            # Calculate confidence score based on various factors
            confidence_score = self._calculate_confidence_score(
                full_text,
                is_auto_generated,
                selected_language in languages
            )
            
            self._success_count += 1
            
            return TranscriptData(
                content=full_text,
                segments=segments,
                language=selected_language.split('-')[0],  # Extract base language
                language_code=selected_language,
                is_auto_generated=is_auto_generated,
                confidence_score=confidence_score,
                has_timestamps=True
            )
            
        except (TranscriptError, TranscriptTimeoutError) as e:
            # Re-raise our custom errors
            self._error_count += 1
            raise
            
        except Exception as e:
            # Convert other errors to our custom errors
            self._error_count += 1
            custom_error = await handle_transcript_error(
                e,
                video_id=youtube_video_id,
                context={'languages': languages}
            )
            raise custom_error
    
    async def list_available_transcripts(
        self,
        youtube_video_id: str
    ) -> List[TranscriptInfo]:
        """
        List all available transcripts for a video.
        
        Args:
            youtube_video_id: YouTube video ID
            
        Returns:
            List of TranscriptInfo objects
            
        Raises:
            Various TranscriptError subclasses
        """
        await self._rate_limit()
        logger.info(f"Listing available transcripts for video {youtube_video_id}")
        
        try:
            transcript_list = await self._execute_with_timeout(
                YouTubeTranscriptApi.list_transcripts,
                youtube_video_id
            )
            
            transcripts = []
            for transcript in transcript_list:
                transcripts.append(TranscriptInfo(
                    language=transcript.language,
                    language_code=transcript.language_code,
                    is_auto_generated=transcript.is_generated,
                    is_translatable=transcript.is_translatable,
                    is_available=True
                ))
            
            return transcripts
            
        except Exception as e:
            custom_error = await handle_transcript_error(
                e,
                video_id=youtube_video_id
            )
            raise custom_error
    
    async def get_transcript_with_timestamps(
        self,
        youtube_video_id: str,
        languages: Optional[List[str]] = None,
        merge_segments: bool = False,
        max_segment_duration: float = 30.0
    ) -> Tuple[str, List[TranscriptSegment]]:
        """
        Get transcript with detailed timestamp information.
        
        Args:
            youtube_video_id: YouTube video ID
            languages: Preferred languages
            merge_segments: Whether to merge small segments
            max_segment_duration: Maximum duration for merged segments
            
        Returns:
            Tuple of (full_text, segments)
        """
        transcript_data = await self.get_transcript(youtube_video_id, languages)
        
        if merge_segments:
            segments = transcript_data.merge_segments(max_segment_duration)
        else:
            segments = transcript_data.segments
        
        return transcript_data.content, segments
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(TranscriptRateLimitError),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def get_transcript_with_retry(
        self,
        youtube_video_id: str,
        languages: Optional[List[str]] = None
    ) -> Optional[TranscriptData]:
        """
        Get transcript with automatic retry on rate limiting.
        
        Args:
            youtube_video_id: YouTube video ID
            languages: Preferred languages
            
        Returns:
            TranscriptData or None if all retries failed
        """
        try:
            return await self.get_transcript(youtube_video_id, languages)
        except TranscriptError as e:
            if not is_retryable_error(e):
                logger.warning(f"Non-retryable error for video {youtube_video_id}: {e}")
                return None
            raise
    
    def _calculate_confidence_score(
        self,
        text: str,
        is_auto_generated: bool,
        is_preferred_language: bool
    ) -> float:
        """
        Calculate confidence score for transcript quality.
        
        Args:
            text: Transcript text
            is_auto_generated: Whether transcript is auto-generated
            is_preferred_language: Whether language matches preference
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 1.0
        
        # Reduce score for auto-generated transcripts
        if is_auto_generated:
            score *= 0.85
        
        # Reduce score if not preferred language
        if not is_preferred_language:
            score *= 0.9
        
        # Check text quality indicators
        word_count = len(text.split())
        
        # Too short
        if word_count < settings.MIN_TRANSCRIPT_WORD_COUNT:
            score *= 0.7
        
        # Check for common auto-generation artifacts
        artifacts = ['[Music]', '[Applause]', '[Laughter]', '[inaudible]']
        artifact_count = sum(1 for artifact in artifacts if artifact.lower() in text.lower())
        if artifact_count > 0:
            score *= max(0.5, 1 - (artifact_count * 0.05))
        
        # Check for repetitive content (might indicate errors)
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                score *= 0.8
        
        return min(1.0, max(0.0, score))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "total_requests": self._request_count,
            "successful_requests": self._success_count,
            "failed_requests": self._error_count,
            "success_rate": (
                (self._success_count / self._request_count * 100)
                if self._request_count > 0 else 0
            ),
            "preferred_languages": self.preferred_languages
        }
    
    async def validate_video_has_transcript(
        self,
        youtube_video_id: str
    ) -> bool:
        """
        Quick check if a video has any transcript available.
        
        Args:
            youtube_video_id: YouTube video ID
            
        Returns:
            True if transcript is available
        """
        try:
            transcripts = await self.list_available_transcripts(youtube_video_id)
            return len(transcripts) > 0
        except (TranscriptDisabledError, TranscriptNotAvailableError, TranscriptVideoUnavailableError):
            return False
        except Exception as e:
            logger.error(f"Error checking transcript availability for {youtube_video_id}: {e}")
            return False