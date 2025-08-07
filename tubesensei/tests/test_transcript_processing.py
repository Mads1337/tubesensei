import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from uuid import uuid4

from app.models.transcript_data import (
    TranscriptData,
    TranscriptSegment,
    TranscriptInfo,
    TranscriptMetrics,
    ProcessingResult
)
from app.services.transcript_cleaner import TranscriptCleaner
from app.services.transcript_analyzer import TranscriptAnalyzer
from app.integrations.transcript_api import TranscriptAPIClient
from app.integrations.transcript_errors import (
    TranscriptNotAvailableError,
    TranscriptDisabledError,
    TranscriptTimeoutError
)


class TestTranscriptDataModels:
    """Test transcript data models."""
    
    def test_transcript_segment_creation(self):
        """Test TranscriptSegment creation and end calculation."""
        segment = TranscriptSegment(
            text="Hello world",
            start=0.0,
            duration=5.0
        )
        
        assert segment.text == "Hello world"
        assert segment.start == 0.0
        assert segment.duration == 5.0
        assert segment.end == 5.0  # Automatically calculated
    
    def test_transcript_data_word_count(self):
        """Test TranscriptData word count property."""
        data = TranscriptData(
            content="This is a test transcript with eight words",
            segments=[],
            language="en",
            language_code="en-US",
            is_auto_generated=True
        )
        
        assert data.word_count == 8
    
    def test_transcript_data_merge_segments(self):
        """Test merging transcript segments."""
        segments = [
            TranscriptSegment(text="First", start=0.0, duration=2.0),
            TranscriptSegment(text="Second", start=2.0, duration=3.0),
            TranscriptSegment(text="Third", start=5.0, duration=2.0),
            TranscriptSegment(text="Fourth", start=7.0, duration=25.0),  # Long segment
            TranscriptSegment(text="Fifth", start=32.0, duration=3.0)
        ]
        
        data = TranscriptData(
            content="First Second Third Fourth Fifth",
            segments=segments,
            language="en",
            language_code="en-US",
            is_auto_generated=True
        )
        
        # Merge with max duration of 10 seconds
        merged = data.merge_segments(max_duration=10.0)
        
        assert len(merged) == 3
        assert merged[0].text == "First Second Third"
        assert merged[0].duration == 7.0
        assert merged[1].text == "Fourth"
        assert merged[2].text == "Fifth"
    
    def test_processing_result_statistics(self):
        """Test ProcessingResult statistics calculation."""
        result = ProcessingResult(
            total_processed=10,
            successful=7,
            failed=2,
            skipped=1,
            processing_time_seconds=100.0
        )
        
        assert result.success_rate == 70.0
        assert result.average_time_per_video == 10.0
        
        # Test merge
        other = ProcessingResult(
            total_processed=5,
            successful=3,
            failed=2,
            processing_time_seconds=50.0
        )
        
        result.merge(other)
        assert result.total_processed == 15
        assert result.successful == 10
        assert result.failed == 4


class TestTranscriptCleaner:
    """Test transcript cleaning functionality."""
    
    @pytest.fixture
    def cleaner(self):
        """Create a TranscriptCleaner instance."""
        return TranscriptCleaner()
    
    def test_remove_artifacts(self, cleaner):
        """Test removal of YouTube artifacts."""
        text = "Hello [Music] world [Applause] test [Laughter] content"
        cleaned = cleaner.remove_artifacts(text)
        
        assert "[Music]" not in cleaned
        assert "[Applause]" not in cleaned
        assert "[Laughter]" not in cleaned
        assert "Hello  world  test  content" == cleaned
    
    def test_normalize_text(self, cleaner):
        """Test text normalization."""
        text = "Hello   world  !  This  is   a   test ."
        normalized = cleaner.normalize_text(text)
        
        assert normalized == "Hello world! This is a test."
    
    def test_fix_common_errors(self, cleaner):
        """Test fixing common OCR/ASR errors."""
        text = "Im going to youre house. Theyre waiting for us."
        fixed = cleaner.fix_common_errors(text)
        
        assert "I'm" in fixed
        assert "you're" in fixed
        assert "They're" in fixed
    
    def test_clean_transcript_full(self, cleaner):
        """Test full transcript cleaning pipeline."""
        text = """
        [Music] Hello everyone, Im here to talk about...
        [Applause] 
        Today   well   discuss   important  topics !
        [Laughter] Thank you for watching [Music]
        """
        
        cleaned = cleaner.clean_transcript(text)
        
        assert "[Music]" not in cleaned
        assert "[Applause]" not in cleaned
        assert "I'm" in cleaned
        assert "we'll" in cleaned
        assert "  " not in cleaned  # No double spaces
    
    def test_remove_timestamps(self, cleaner):
        """Test timestamp removal."""
        text = "00:01 Hello world\n01:23 This is a test\n[02:45] Another line"
        cleaned = cleaner.remove_timestamps(text)
        
        assert "00:01" not in cleaned
        assert "01:23" not in cleaned
        assert "[02:45]" not in cleaned
        assert "Hello world" in cleaned
    
    def test_validate_and_trim(self, cleaner):
        """Test text validation and trimming."""
        # Create a very long text
        long_text = "word " * 100000  # Much longer than max length
        
        with patch('app.config.settings.MAX_TRANSCRIPT_LENGTH', 100):
            trimmed = cleaner.validate_and_trim(long_text)
            assert len(trimmed) <= 100


class TestTranscriptAnalyzer:
    """Test transcript analysis functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a TranscriptAnalyzer instance."""
        return TranscriptAnalyzer()
    
    def test_count_words(self, analyzer):
        """Test word counting."""
        text = "This is a simple test with seven words"
        count = analyzer.count_words(text)
        assert count == 8
    
    def test_count_sentences(self, analyzer):
        """Test sentence counting."""
        text = "First sentence. Second sentence! Third sentence? Fourth."
        count = analyzer.count_sentences(text)
        assert count == 4
    
    def test_detect_language(self, analyzer):
        """Test language detection."""
        # English text
        en_text = "This is a test in English with enough words to detect the language properly"
        lang, confidence = analyzer.detect_language(en_text)
        assert lang == "en"
        assert confidence > 0.5
        
        # Spanish text
        es_text = "Este es un texto en español con suficientes palabras para detectar el idioma correctamente"
        lang, confidence = analyzer.detect_language(es_text)
        assert lang == "es"
        assert confidence > 0.5
    
    def test_calculate_quality_score(self, analyzer):
        """Test quality score calculation."""
        # Good quality text
        good_text = " ".join(["word" + str(i) for i in range(200)])  # 200 unique words
        score = analyzer.calculate_quality_score(
            content=good_text,
            is_auto_generated=False,
            word_count=200,
            sentence_count=10,
            unique_words=200
        )
        assert score > 0.7
        
        # Poor quality text (repetitive)
        poor_text = "word " * 100  # Very repetitive
        score = analyzer.calculate_quality_score(
            content=poor_text,
            is_auto_generated=True,
            word_count=100,
            sentence_count=1,
            unique_words=1
        )
        assert score < 0.5
    
    def test_analyze_transcript(self, analyzer):
        """Test full transcript analysis."""
        text = """
        This is a test transcript for analysis. It contains multiple sentences.
        The analyzer should be able to extract various metrics from this text.
        We're testing word count, sentence count, and quality scoring.
        This should provide a comprehensive analysis of the transcript content.
        """
        
        metrics = analyzer.analyze_transcript(text, is_auto_generated=False)
        
        assert metrics.word_count > 0
        assert metrics.sentence_count == 4
        assert metrics.unique_words > 0
        assert metrics.avg_sentence_length > 0
        assert metrics.quality_score >= 0.0
        assert metrics.quality_score <= 1.0
    
    def test_estimate_reading_level(self, analyzer):
        """Test reading level estimation."""
        # Simple text
        level = analyzer.estimate_reading_level(
            avg_sentence_length=5.0,
            unique_word_ratio=0.3
        )
        assert level == "Elementary"
        
        # Complex text
        level = analyzer.estimate_reading_level(
            avg_sentence_length=25.0,
            unique_word_ratio=0.8
        )
        assert level in ["College", "Graduate"]


class TestTranscriptAPIClient:
    """Test transcript API client functionality."""
    
    @pytest.fixture
    def client(self):
        """Create a TranscriptAPIClient instance."""
        return TranscriptAPIClient(
            timeout_seconds=10,
            max_retries=2,
            preferred_languages=["en", "es"]
        )
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limiting between requests."""
        client._min_request_interval = 0.1  # 100ms for testing
        
        start_time = asyncio.get_event_loop().time()
        await client._rate_limit()
        await client._rate_limit()
        end_time = asyncio.get_event_loop().time()
        
        # Should have waited at least 100ms
        assert (end_time - start_time) >= 0.1
    
    def test_calculate_confidence_score(self, client):
        """Test confidence score calculation."""
        # Good transcript
        score = client._calculate_confidence_score(
            text="This is a high quality transcript with many words " * 20,
            is_auto_generated=False,
            is_preferred_language=True
        )
        assert score > 0.8
        
        # Poor transcript
        score = client._calculate_confidence_score(
            text="[Music] [Applause] short text [inaudible]",
            is_auto_generated=True,
            is_preferred_language=False
        )
        assert score < 0.5
    
    @pytest.mark.asyncio
    async def test_get_transcript_with_mock(self, client):
        """Test transcript extraction with mocked YouTube API."""
        mock_transcript = MagicMock()
        mock_transcript.fetch.return_value = [
            {"text": "Hello", "start": 0.0, "duration": 1.0},
            {"text": "world", "start": 1.0, "duration": 1.0}
        ]
        mock_transcript.language_code = "en"
        mock_transcript.is_generated = False
        
        mock_list = MagicMock()
        mock_list.find_manually_created_transcript.return_value = mock_transcript
        
        with patch('youtube_transcript_api.YouTubeTranscriptApi.list_transcripts', return_value=mock_list):
            with patch.object(client, '_execute_with_timeout', new_callable=AsyncMock) as mock_exec:
                # Setup mock to return the list first, then the transcript data
                mock_exec.side_effect = [
                    mock_list,  # For list_transcripts
                    [{"text": "Hello", "start": 0.0, "duration": 1.0},
                     {"text": "world", "start": 1.0, "duration": 1.0}]  # For fetch
                ]
                
                result = await client.get_transcript("test_video_id")
                
                assert result is not None
                assert result.content == "Hello world"
                assert len(result.segments) == 2
                assert result.language == "en"
                assert not result.is_auto_generated
    
    @pytest.mark.asyncio
    async def test_handle_transcript_not_available(self, client):
        """Test handling of transcript not available error."""
        with patch('youtube_transcript_api.YouTubeTranscriptApi.list_transcripts') as mock_list:
            mock_list.side_effect = Exception("No transcript found")
            
            with pytest.raises(Exception):
                await client.get_transcript("test_video_id")


class TestTranscriptIntegration:
    """Integration tests for transcript processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_processing_pipeline(self):
        """Test the complete transcript processing pipeline."""
        from app.services.transcript_processor import TranscriptProcessor
        
        # Create processor with mocked components
        processor = TranscriptProcessor(use_cache=False)
        
        # Mock the API client
        mock_transcript_data = TranscriptData(
            content="This is a test transcript content with enough words to be valid",
            segments=[
                TranscriptSegment(text="This is a test", start=0.0, duration=2.0),
                TranscriptSegment(text="transcript content", start=2.0, duration=2.0),
                TranscriptSegment(text="with enough words to be valid", start=4.0, duration=3.0)
            ],
            language="en",
            language_code="en-US",
            is_auto_generated=False,
            confidence_score=0.9
        )
        
        with patch.object(processor.api_client, 'get_transcript_with_retry', new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_transcript_data
            
            # Mock database operations
            with patch('app.services.transcript_processor.get_db_context') as mock_db:
                mock_session = AsyncMock()
                mock_video = Mock()
                mock_video.id = uuid4()
                mock_video.youtube_video_id = "test_video_id"
                mock_video.has_captions = True
                mock_video.status = "discovered"
                
                mock_session.get.return_value = mock_video
                mock_db.return_value.__aenter__.return_value = mock_session
                
                # Mock repository
                with patch('app.services.transcript_processor.TranscriptRepository') as mock_repo_class:
                    mock_repo = Mock()
                    mock_repo.get_by_video.return_value = None  # No existing transcript
                    mock_repo.create = AsyncMock()
                    mock_repo_class.return_value = mock_repo
                    
                    # Process transcript
                    result = await processor.extract_transcript(
                        video_id=mock_video.id,
                        force_refresh=False,
                        clean_content=True,
                        calculate_metrics=True,
                        save_to_db=False  # Skip actual DB save for test
                    )
                    
                    # Verify API was called
                    mock_api.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch transcript processing."""
        from app.services.transcript_processor import TranscriptProcessor
        
        processor = TranscriptProcessor(use_cache=False)
        
        # Create mock video IDs
        video_ids = [uuid4() for _ in range(5)]
        
        with patch.object(processor, 'extract_transcript', new_callable=AsyncMock) as mock_extract:
            # Simulate mixed results
            mock_extract.side_effect = [
                Mock(),  # Success
                None,    # Failure
                Mock(),  # Success
                Mock(),  # Success
                None     # Failure
            ]
            
            result = await processor.batch_process_transcripts(
                video_ids=video_ids,
                concurrent_limit=2
            )
            
            assert result.total_processed == 5
            assert result.successful == 3
            assert result.failed == 2
            assert result.success_rate == 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])