"""
Unit tests for TranscriptAPI integration.

Tests the transcript API client including transcript fetching,
language handling, error handling, and data processing.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from app.integrations.transcript_api import TranscriptAPI, TranscriptFetcher
from app.integrations.transcript_errors import (
    TranscriptNotFoundError,
    TranscriptDisabledError,
    TranscriptPrivateError,
    TranscriptUnavailableError,
    TranscriptLanguageError
)
from tests.fixtures.fixtures import MockTranscriptAPIResponses


class TestTranscriptAPIInit:
    """Test suite for TranscriptAPI initialization."""
    
    def test_init_default(self):
        """Test default initialization."""
        api = TranscriptAPI()
        assert api.default_language == "en"
        assert api.fallback_languages == ["en", "en-US", "en-GB"]
        assert api.max_retries == 3
    
    def test_init_with_custom_settings(self):
        """Test initialization with custom settings."""
        api = TranscriptAPI(
            default_language="es",
            fallback_languages=["es", "en"],
            max_retries=5
        )
        assert api.default_language == "es"
        assert api.fallback_languages == ["es", "en"]
        assert api.max_retries == 5


class TestTranscriptAPIGetTranscript:
    """Test suite for TranscriptAPI.get_transcript method."""
    
    @pytest.mark.asyncio
    async def test_get_transcript_success(self):
        """Test successful transcript retrieval."""
        api = TranscriptAPI()
        
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.return_value = mock_transcript_data
            
            result = await api.get_transcript("test_video_001")
            
            assert result is not None
            assert 'content' in result
            assert 'language' in result
            assert 'segments' in result
            assert len(result['segments']) == 5
            assert result['language'] == 'en'
            
            mock_yt_api.get_transcript.assert_called_once_with("test_video_001", languages=["en"])
    
    @pytest.mark.asyncio
    async def test_get_transcript_with_specific_language(self):
        """Test transcript retrieval with specific language."""
        api = TranscriptAPI()
        
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001", "es")
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.return_value = mock_transcript_data
            
            result = await api.get_transcript("test_video_001", language="es")
            
            assert result['language'] == 'es'
            mock_yt_api.get_transcript.assert_called_once_with("test_video_001", languages=["es"])
    
    @pytest.mark.asyncio
    async def test_get_transcript_with_fallback(self):
        """Test transcript retrieval with language fallback."""
        api = TranscriptAPI(fallback_languages=["en", "es", "fr"])
        
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            # First call fails, second succeeds
            mock_yt_api.get_transcript.side_effect = [
                TranscriptNotFoundError("test_video_001"),
                mock_transcript_data
            ]
            
            result = await api.get_transcript("test_video_001")
            
            assert result is not None
            assert result['language'] == 'en'
            
            # Should have tried default language first, then fallback
            assert mock_yt_api.get_transcript.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_transcript_not_found(self):
        """Test transcript retrieval when not found."""
        api = TranscriptAPI()
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.side_effect = TranscriptNotFoundError("test_video_001")
            
            with pytest.raises(TranscriptNotFoundError):
                await api.get_transcript("test_video_001")
    
    @pytest.mark.asyncio
    async def test_get_transcript_disabled(self):
        """Test transcript retrieval when disabled."""
        api = TranscriptAPI()
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.side_effect = TranscriptDisabledError("test_video_001")
            
            with pytest.raises(TranscriptDisabledError):
                await api.get_transcript("test_video_001")
    
    @pytest.mark.asyncio
    async def test_get_transcript_private_video(self):
        """Test transcript retrieval for private video."""
        api = TranscriptAPI()
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.side_effect = TranscriptPrivateError("test_video_001")
            
            with pytest.raises(TranscriptPrivateError):
                await api.get_transcript("test_video_001")
    
    @pytest.mark.asyncio
    async def test_get_transcript_with_retries(self):
        """Test transcript retrieval with retry logic."""
        api = TranscriptAPI(max_retries=2)
        
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            # First two calls fail with temporary error, third succeeds
            mock_yt_api.get_transcript.side_effect = [
                TranscriptUnavailableError("Temporary error"),
                TranscriptUnavailableError("Temporary error"),
                mock_transcript_data
            ]
            
            result = await api.get_transcript("test_video_001")
            
            assert result is not None
            assert mock_yt_api.get_transcript.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_transcript_max_retries_exceeded(self):
        """Test transcript retrieval when max retries exceeded."""
        api = TranscriptAPI(max_retries=2)
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.side_effect = TranscriptUnavailableError("Persistent error")
            
            with pytest.raises(TranscriptUnavailableError):
                await api.get_transcript("test_video_001")
            
            # Should try max_retries + 1 times (initial + retries)
            assert mock_yt_api.get_transcript.call_count == 3


class TestTranscriptAPIGetAvailableLanguages:
    """Test suite for TranscriptAPI.get_available_languages method."""
    
    @pytest.mark.asyncio
    async def test_get_available_languages_success(self):
        """Test successful available languages retrieval."""
        api = TranscriptAPI()
        
        mock_transcript_list = MockTranscriptAPIResponses.transcript_list("test_video_001")
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.list_transcripts.return_value = Mock()
            mock_yt_api.list_transcripts.return_value.transcript_list = mock_transcript_list
            
            result = await api.get_available_languages("test_video_001")
            
            assert len(result) == 2
            assert 'en' in [lang['language_code'] for lang in result]
            assert 'es' in [lang['language_code'] for lang in result]
            
            # Check language details
            en_lang = next(lang for lang in result if lang['language_code'] == 'en')
            assert en_lang['is_generated'] is True
            assert en_lang['is_translatable'] is True
    
    @pytest.mark.asyncio
    async def test_get_available_languages_not_found(self):
        """Test available languages when video not found."""
        api = TranscriptAPI()
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.list_transcripts.side_effect = TranscriptNotFoundError("test_video_001")
            
            with pytest.raises(TranscriptNotFoundError):
                await api.get_available_languages("test_video_001")
    
    @pytest.mark.asyncio
    async def test_get_available_languages_empty(self):
        """Test available languages when no transcripts available."""
        api = TranscriptAPI()
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.list_transcripts.return_value = Mock()
            mock_yt_api.list_transcripts.return_value.transcript_list = []
            
            result = await api.get_available_languages("test_video_001")
            
            assert result == []


class TestTranscriptAPIBatchOperations:
    """Test suite for TranscriptAPI batch operations."""
    
    @pytest.mark.asyncio
    async def test_get_transcripts_batch_success(self):
        """Test successful batch transcript retrieval."""
        api = TranscriptAPI()
        
        video_ids = ["test_video_001", "test_video_002", "test_video_003"]
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.return_value = mock_transcript_data
            
            results = await api.get_transcripts_batch(video_ids)
            
            assert len(results) == 3
            assert all('video_id' in result for result in results.values())
            assert all('content' in result for result in results.values())
            
            # Should have made one API call per video
            assert mock_yt_api.get_transcript.call_count == 3
    
    @pytest.mark.asyncio
    async def test_get_transcripts_batch_partial_success(self):
        """Test batch transcript retrieval with some failures."""
        api = TranscriptAPI()
        
        video_ids = ["test_video_001", "test_video_002", "test_video_003"]
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            # First succeeds, second fails, third succeeds
            mock_yt_api.get_transcript.side_effect = [
                mock_transcript_data,
                TranscriptNotFoundError("test_video_002"),
                mock_transcript_data
            ]
            
            results = await api.get_transcripts_batch(video_ids)
            
            # Should have 2 successful results
            successful_results = [r for r in results.values() if 'content' in r]
            failed_results = [r for r in results.values() if 'error' in r]
            
            assert len(successful_results) == 2
            assert len(failed_results) == 1
            assert failed_results[0]['error'] == 'TranscriptNotFoundError'
    
    @pytest.mark.asyncio
    async def test_get_transcripts_batch_with_concurrency_limit(self):
        """Test batch transcript retrieval with concurrency limit."""
        api = TranscriptAPI()
        
        video_ids = [f"test_video_{i:03d}" for i in range(10)]
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.return_value = mock_transcript_data
            
            with patch('asyncio.Semaphore') as mock_semaphore:
                mock_semaphore_instance = AsyncMock()
                mock_semaphore.return_value = mock_semaphore_instance
                mock_semaphore_instance.acquire.return_value.__aenter__ = AsyncMock()
                mock_semaphore_instance.acquire.return_value.__aexit__ = AsyncMock()
                
                results = await api.get_transcripts_batch(video_ids, max_concurrent=5)
                
                # Should create semaphore with correct limit
                mock_semaphore.assert_called_once_with(5)
                assert len(results) == 10


class TestTranscriptFetcher:
    """Test suite for TranscriptFetcher class."""
    
    def test_transcript_fetcher_init(self):
        """Test TranscriptFetcher initialization."""
        fetcher = TranscriptFetcher(
            video_id="test_video_001",
            language="en",
            api=TranscriptAPI()
        )
        
        assert fetcher.video_id == "test_video_001"
        assert fetcher.language == "en"
        assert isinstance(fetcher.api, TranscriptAPI)
    
    @pytest.mark.asyncio
    async def test_transcript_fetcher_fetch_success(self):
        """Test successful transcript fetch using TranscriptFetcher."""
        api = TranscriptAPI()
        fetcher = TranscriptFetcher("test_video_001", "en", api)
        
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        with patch.object(api, 'get_transcript', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                'video_id': "test_video_001",
                'content': "Test transcript content",
                'language': 'en',
                'segments': mock_transcript_data
            }
            
            result = await fetcher.fetch()
            
            assert result['video_id'] == "test_video_001"
            assert result['language'] == 'en'
            assert 'content' in result
            assert 'segments' in result
    
    @pytest.mark.asyncio
    async def test_transcript_fetcher_fetch_with_retry(self):
        """Test transcript fetch with retry logic."""
        api = TranscriptAPI(max_retries=2)
        fetcher = TranscriptFetcher("test_video_001", "en", api)
        
        mock_transcript_data = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        with patch.object(api, 'get_transcript', new_callable=AsyncMock) as mock_get:
            # First call fails, second succeeds
            mock_get.side_effect = [
                TranscriptUnavailableError("Temporary error"),
                {
                    'video_id': "test_video_001",
                    'content': "Test transcript content",
                    'language': 'en',
                    'segments': mock_transcript_data
                }
            ]
            
            result = await fetcher.fetch()
            
            assert result is not None
            assert mock_get.call_count == 2


class TestTranscriptAPIDataProcessing:
    """Test suite for TranscriptAPI data processing methods."""
    
    def test_process_transcript_segments(self):
        """Test processing of transcript segments."""
        api = TranscriptAPI()
        
        raw_segments = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        result = api._process_transcript_segments(raw_segments)
        
        assert 'content' in result
        assert 'segments' in result
        assert 'word_count' in result
        assert 'char_count' in result
        assert 'total_duration' in result
        
        # Check content concatenation
        expected_content = " ".join(segment['text'] for segment in raw_segments)
        assert result['content'] == expected_content
        
        # Check statistics
        assert result['word_count'] > 0
        assert result['char_count'] > 0
        assert result['total_duration'] > 0
    
    def test_process_transcript_segments_empty(self):
        """Test processing empty transcript segments."""
        api = TranscriptAPI()
        
        result = api._process_transcript_segments([])
        
        assert result['content'] == ""
        assert result['segments'] == []
        assert result['word_count'] == 0
        assert result['char_count'] == 0
        assert result['total_duration'] == 0
    
    def test_clean_transcript_text(self):
        """Test transcript text cleaning."""
        api = TranscriptAPI()
        
        # Test text with various formatting issues
        dirty_text = "  Hello   world!  \n\n  This is   a test.  "
        cleaned = api._clean_transcript_text(dirty_text)
        
        assert cleaned == "Hello world! This is a test."
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")
        assert "  " not in cleaned  # No double spaces
    
    def test_clean_transcript_text_with_special_chars(self):
        """Test transcript text cleaning with special characters."""
        api = TranscriptAPI()
        
        # Test text with music notations and other artifacts
        text_with_artifacts = "[Music] Hello world! (applause) This is a test. [Laughter]"
        cleaned = api._clean_transcript_text(text_with_artifacts)
        
        # Should remove music notations and applause markers
        assert "[Music]" not in cleaned
        assert "(applause)" not in cleaned
        assert "[Laughter]" not in cleaned
        assert "Hello world!" in cleaned
        assert "This is a test." in cleaned
    
    def test_detect_transcript_language(self):
        """Test transcript language detection."""
        api = TranscriptAPI()
        
        # Test English text
        english_text = "Hello world! This is a test transcript in English."
        detected = api._detect_transcript_language(english_text)
        assert detected in ['en', 'en-US', 'en-GB']  # Various English codes
        
        # Test Spanish text
        spanish_text = "Hola mundo! Esta es una transcripción de prueba en español."
        detected = api._detect_transcript_language(spanish_text)
        assert detected in ['es', 'es-ES', 'es-MX']  # Various Spanish codes
    
    def test_detect_transcript_language_short_text(self):
        """Test language detection with short text."""
        api = TranscriptAPI()
        
        # Very short text should return unknown
        short_text = "Hi"
        detected = api._detect_transcript_language(short_text)
        assert detected == 'unknown'
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        api = TranscriptAPI()
        
        # High quality segments
        high_quality_segments = [
            {'text': 'Clear speech segment', 'start': 0, 'duration': 3},
            {'text': 'Another clear segment', 'start': 3, 'duration': 4},
            {'text': 'Well articulated words', 'start': 7, 'duration': 3}
        ]
        
        score = api._calculate_confidence_score(high_quality_segments, is_auto_generated=False)
        assert score > 80  # Should have high confidence
        
        # Low quality segments (auto-generated with gaps)
        low_quality_segments = [
            {'text': '...', 'start': 0, 'duration': 2},
            {'text': 'um uh', 'start': 5, 'duration': 1},  # Gap from 2-5
            {'text': '[inaudible]', 'start': 10, 'duration': 2}  # Gap from 6-10
        ]
        
        score = api._calculate_confidence_score(low_quality_segments, is_auto_generated=True)
        assert score < 50  # Should have low confidence
    
    def test_extract_metadata(self):
        """Test metadata extraction from transcript."""
        api = TranscriptAPI()
        
        segments = MockTranscriptAPIResponses.transcript_content("test_video_001")
        
        metadata = api._extract_metadata(segments, "en", True)
        
        assert 'total_segments' in metadata
        assert 'average_segment_duration' in metadata
        assert 'speaking_rate_wpm' in metadata
        assert 'language_detected' in metadata
        assert 'is_auto_generated' in metadata
        assert 'confidence_score' in metadata
        assert 'extraction_timestamp' in metadata
        
        # Check values are reasonable
        assert metadata['total_segments'] == len(segments)
        assert metadata['average_segment_duration'] > 0
        assert metadata['speaking_rate_wpm'] > 0
        assert metadata['language_detected'] == 'en'
        assert metadata['is_auto_generated'] is True


class TestTranscriptAPIErrorHandling:
    """Test suite for TranscriptAPI error handling."""
    
    @pytest.mark.asyncio
    async def test_handle_transcript_not_found(self):
        """Test handling of transcript not found error."""
        api = TranscriptAPI()
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.side_effect = TranscriptNotFoundError("test_video_001")
            
            with pytest.raises(TranscriptNotFoundError) as exc_info:
                await api.get_transcript("test_video_001")
            
            assert "test_video_001" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_handle_transcript_disabled(self):
        """Test handling of transcript disabled error."""
        api = TranscriptAPI()
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.side_effect = TranscriptDisabledError("test_video_001")
            
            with pytest.raises(TranscriptDisabledError) as exc_info:
                await api.get_transcript("test_video_001")
            
            assert "test_video_001" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_handle_transcript_language_error(self):
        """Test handling of transcript language error."""
        api = TranscriptAPI()
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.side_effect = TranscriptLanguageError("Language not available")
            
            with pytest.raises(TranscriptLanguageError):
                await api.get_transcript("test_video_001", language="fr")
    
    @pytest.mark.asyncio
    async def test_handle_generic_exception(self):
        """Test handling of generic exceptions."""
        api = TranscriptAPI()
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            mock_yt_api.get_transcript.side_effect = Exception("Unexpected error")
            
            with pytest.raises(TranscriptUnavailableError):
                await api.get_transcript("test_video_001")
    
    @pytest.mark.asyncio
    async def test_retry_logic_with_exponential_backoff(self):
        """Test retry logic with exponential backoff."""
        api = TranscriptAPI(max_retries=3)
        
        with patch('app.integrations.transcript_api.YouTubeTranscriptApi') as mock_yt_api:
            with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                mock_yt_api.get_transcript.side_effect = [
                    TranscriptUnavailableError("Retry 1"),
                    TranscriptUnavailableError("Retry 2"),
                    TranscriptUnavailableError("Retry 3"),
                    MockTranscriptAPIResponses.transcript_content("test_video_001")
                ]
                
                with patch.object(api, '_process_transcript_segments') as mock_process:
                    mock_process.return_value = {
                        'content': 'Test content',
                        'segments': [],
                        'word_count': 2,
                        'char_count': 12,
                        'total_duration': 10.0
                    }
                    
                    result = await api.get_transcript("test_video_001")
                    
                    # Should have slept with exponential backoff
                    assert mock_sleep.call_count == 3
                    sleep_times = [call[0][0] for call in mock_sleep.call_args_list]
                    
                    # Check exponential backoff pattern (1, 2, 4 seconds)
                    assert sleep_times[0] == 1
                    assert sleep_times[1] == 2
                    assert sleep_times[2] == 4
                    
                    assert result is not None