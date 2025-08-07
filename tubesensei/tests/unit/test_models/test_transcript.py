"""
Unit tests for Transcript model.

Tests the Transcript model including properties, methods, validations,
and edge cases.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from app.models.transcript import Transcript, TranscriptSource, TranscriptLanguage
from app.models.video import Video
from app.models.channel import Channel
from tests.fixtures.fixtures import TranscriptFactory, VideoFactory, ChannelFactory


class TestTranscriptModel:
    """Test suite for Transcript model."""
    
    @pytest.mark.asyncio
    async def test_transcript_creation(self, db_session):
        """Test basic transcript creation."""
        # Create channel and video first
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        transcript = TranscriptFactory.build(video_id=video.id)
        db_session.add(transcript)
        await db_session.commit()
        
        assert transcript.id is not None
        assert transcript.created_at is not None
        assert transcript.updated_at is not None
        assert transcript.source == TranscriptSource.YOUTUBE_AUTO
        assert transcript.language == TranscriptLanguage.EN
        assert transcript.is_auto_generated is True
        assert transcript.is_complete is True
    
    @pytest.mark.asyncio
    async def test_transcript_required_fields(self, db_session):
        """Test that required fields cannot be null."""
        with pytest.raises(IntegrityError):
            transcript = Transcript(
                content="Test content"
                # Missing video_id
            )
            db_session.add(transcript)
            await db_session.commit()
    
    def test_transcript_repr(self):
        """Test transcript string representation."""
        from uuid import UUID
        video_id = UUID("550e8400-e29b-41d4-a716-446655440000")
        
        transcript = TranscriptFactory.build(
            id="660e8400-e29b-41d4-a716-446655440000",
            video_id=video_id,
            source=TranscriptSource.YOUTUBE_MANUAL,
            language_code="es"
        )
        
        expected = "<Transcript(id=660e8400-e29b-41d4-a716-446655440000, video_id=550e8400-e29b-41d4-a716-446655440000, source=youtube_manual, language=es)>"
        assert repr(transcript) == expected
    
    def test_content_preview_property(self):
        """Test content_preview property."""
        # Short content
        short_content = "This is a short transcript"
        transcript = TranscriptFactory.build(content=short_content)
        assert transcript.content_preview == short_content
        
        # Long content (should be truncated)
        long_content = "A" * 600
        transcript = TranscriptFactory.build(content=long_content)
        expected = "A" * 500 + "..."
        assert transcript.content_preview == expected
        
        # Empty content
        transcript = TranscriptFactory.build(content="")
        assert transcript.content_preview == ""
        
        # None content
        transcript = TranscriptFactory.build(content=None)
        assert transcript.content_preview == ""
    
    def test_is_english_property(self):
        """Test is_english property."""
        # English transcript
        en_transcript = TranscriptFactory.build(language=TranscriptLanguage.EN)
        assert en_transcript.is_english is True
        
        # Non-English transcripts
        non_english_languages = [
            TranscriptLanguage.ES,
            TranscriptLanguage.FR,
            TranscriptLanguage.DE,
            TranscriptLanguage.OTHER
        ]
        
        for lang in non_english_languages:
            transcript = TranscriptFactory.build(language=lang)
            assert transcript.is_english is False
    
    def test_needs_translation_property(self):
        """Test needs_translation property."""
        # English doesn't need translation
        en_transcript = TranscriptFactory.build(language=TranscriptLanguage.EN)
        assert en_transcript.needs_translation is False
        
        # OTHER language doesn't need translation
        other_transcript = TranscriptFactory.build(language=TranscriptLanguage.OTHER)
        assert other_transcript.needs_translation is False
        
        # Non-English known languages need translation
        translatable_languages = [
            TranscriptLanguage.ES,
            TranscriptLanguage.FR,
            TranscriptLanguage.DE,
            TranscriptLanguage.PT,
            TranscriptLanguage.IT,
            TranscriptLanguage.JA,
            TranscriptLanguage.KO,
            TranscriptLanguage.ZH
        ]
        
        for lang in translatable_languages:
            transcript = TranscriptFactory.build(language=lang)
            assert transcript.needs_translation is True
    
    def test_calculate_stats_method(self):
        """Test calculate_stats method."""
        content = "Hello world! This is a test transcript with multiple words."
        transcript = TranscriptFactory.build(content=content)
        
        transcript.calculate_stats()
        
        expected_word_count = len(content.split())  # 11 words
        expected_char_count = len(content)  # Character count
        
        assert transcript.word_count == expected_word_count
        assert transcript.char_count == expected_char_count
    
    def test_calculate_stats_empty_content(self):
        """Test calculate_stats with empty content."""
        transcript = TranscriptFactory.build(content="")
        transcript.calculate_stats()
        
        assert transcript.word_count == 0
        assert transcript.char_count == 0
    
    def test_calculate_stats_none_content(self):
        """Test calculate_stats with None content."""
        transcript = TranscriptFactory.build(content=None)
        transcript.calculate_stats()
        
        # Should handle None gracefully
        assert transcript.word_count is None or transcript.word_count == 0
        assert transcript.char_count is None or transcript.char_count == 0
    
    def test_clean_content_method(self):
        """Test clean_content method."""
        # Content with various formatting to remove
        dirty_content = """
        [Music] Welcome to this tutorial (background noise) about <strong>programming</strong>.
        [Applause]   This   has   multiple   spaces   and   (more noise).
        <em>HTML tags</em> should be removed [sound effects].
        """
        
        transcript = TranscriptFactory.build(content=dirty_content)
        cleaned = transcript.clean_content()
        
        # Should remove brackets, parentheses, HTML tags, and normalize spaces
        assert "[Music]" not in cleaned
        assert "[Applause]" not in cleaned
        assert "[sound effects]" not in cleaned
        assert "(background noise)" not in cleaned
        assert "(more noise)" not in cleaned
        assert "<strong>" not in cleaned
        assert "</strong>" not in cleaned
        assert "<em>" not in cleaned
        assert "</em>" not in cleaned
        
        # Should normalize multiple spaces to single spaces
        assert "   " not in cleaned
        
        # Should be trimmed
        assert not cleaned.startswith(" ")
        assert not cleaned.endswith(" ")
    
    def test_clean_content_empty(self):
        """Test clean_content with empty content."""
        transcript = TranscriptFactory.build(content="")
        assert transcript.clean_content() == ""
        
        transcript = TranscriptFactory.build(content=None)
        assert transcript.clean_content() == ""
    
    def test_extract_timestamps_method(self):
        """Test extract_timestamps method."""
        segments = [
            {"start": 0.0, "end": 3.0, "duration": 3.0, "text": "Welcome to this video"},
            {"start": 3.0, "end": 7.0, "duration": 4.0, "text": "Today we will learn about programming"},
            {"start": 7.0, "end": 10.0, "duration": 3.0, "text": "Let's get started"}
        ]
        
        transcript = TranscriptFactory.build(segments=segments)
        timestamps = transcript.extract_timestamps()
        
        assert len(timestamps) == 3
        
        for i, timestamp in enumerate(timestamps):
            assert timestamp["start"] == segments[i]["start"]
            assert timestamp["end"] == segments[i]["end"]
            assert timestamp["duration"] == segments[i]["duration"]
            assert timestamp["text"] == segments[i]["text"]
    
    def test_extract_timestamps_empty(self):
        """Test extract_timestamps with empty or invalid segments."""
        # No segments
        transcript = TranscriptFactory.build(segments=None)
        assert transcript.extract_timestamps() == []
        
        # Empty segments
        transcript = TranscriptFactory.build(segments=[])
        assert transcript.extract_timestamps() == []
        
        # Invalid segment format
        invalid_segments = [
            {"invalid": "data"},
            {"start": 0.0, "text": "Missing end"},
            {"end": 3.0, "text": "Missing start"}
        ]
        
        transcript = TranscriptFactory.build(segments=invalid_segments)
        timestamps = transcript.extract_timestamps()
        
        # Should handle invalid segments gracefully
        assert len(timestamps) <= len(invalid_segments)
        for timestamp in timestamps:
            assert "start" in timestamp
            assert "text" in timestamp
    
    @pytest.mark.asyncio
    async def test_transcript_default_values(self, db_session):
        """Test default values are set correctly."""
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        transcript = Transcript(
            video_id=video.id,
            content="Test transcript content"
        )
        db_session.add(transcript)
        await db_session.commit()
        
        assert transcript.source == TranscriptSource.YOUTUBE_AUTO
        assert transcript.language == TranscriptLanguage.EN
        assert transcript.language_code == "en"
        assert transcript.is_auto_generated is True
        assert transcript.is_complete is True
        assert transcript.metadata == {}
    
    @pytest.mark.asyncio
    async def test_transcript_json_fields(self, db_session):
        """Test JSONB fields can store and retrieve complex data."""
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        metadata = {
            "api_version": "v3",
            "extraction_method": "youtube_transcript_api",
            "confidence_scores": [0.95, 0.92, 0.88]
        }
        
        segments = [
            {"start": 0.0, "end": 3.0, "text": "First segment"},
            {"start": 3.0, "end": 6.0, "text": "Second segment"}
        ]
        
        transcript = Transcript(
            video_id=video.id,
            content="Test transcript content",
            metadata=metadata,
            segments=segments
        )
        db_session.add(transcript)
        await db_session.commit()
        
        # Refresh from database
        await db_session.refresh(transcript)
        
        assert transcript.metadata == metadata
        assert transcript.segments == segments
    
    @pytest.mark.asyncio
    async def test_transcript_source_enum(self, db_session):
        """Test different transcript source values."""
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        sources = [
            TranscriptSource.YOUTUBE_AUTO,
            TranscriptSource.YOUTUBE_MANUAL,
            TranscriptSource.WHISPER,
            TranscriptSource.CUSTOM
        ]
        
        for i, source in enumerate(sources):
            transcript = TranscriptFactory.build(
                video_id=video.id,
                source=source,
                content=f"Content from {source.value}"
            )
            db_session.add(transcript)
        
        await db_session.commit()
        
        # Query transcripts by source
        for source in sources:
            result = await db_session.execute(
                select(Transcript).filter(Transcript.source == source)
            )
            transcripts = result.scalars().all()
            assert len(transcripts) == 1
            assert transcripts[0].source == source
    
    @pytest.mark.asyncio
    async def test_transcript_language_enum(self, db_session):
        """Test different transcript language values."""
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        languages = [
            (TranscriptLanguage.EN, "en"),
            (TranscriptLanguage.ES, "es"),
            (TranscriptLanguage.FR, "fr"),
            (TranscriptLanguage.DE, "de"),
            (TranscriptLanguage.OTHER, "unknown")
        ]
        
        for i, (language, language_code) in enumerate(languages):
            transcript = TranscriptFactory.build(
                video_id=video.id,
                language=language,
                language_code=language_code,
                content=f"Content in {language.value}"
            )
            db_session.add(transcript)
        
        await db_session.commit()
        
        # Query transcripts by language
        for language, _ in languages:
            result = await db_session.execute(
                select(Transcript).filter(Transcript.language == language)
            )
            transcripts = result.scalars().all()
            assert len(transcripts) == 1
            assert transcripts[0].language == language
    
    def test_transcript_edge_cases(self):
        """Test edge cases and boundary values."""
        # Very long content
        long_content = "A" * 100000
        transcript = TranscriptFactory.build(content=long_content)
        assert len(transcript.content) == 100000
        
        # Content with special characters
        special_content = "Content with émojis 😀 and spëcial châractërs"
        transcript = TranscriptFactory.build(content=special_content)
        assert "😀" in transcript.content
        assert "émojis" in transcript.content
        
        # Very high confidence score
        transcript = TranscriptFactory.build(confidence_score=100)
        assert transcript.confidence_score == 100
        
        # Zero confidence score
        transcript = TranscriptFactory.build(confidence_score=0)
        assert transcript.confidence_score == 0
    
    def test_complex_segments_data(self):
        """Test complex segments data structures."""
        complex_segments = [
            {
                "start": 0.0,
                "end": 3.5,
                "duration": 3.5,
                "text": "Complex segment with metadata",
                "confidence": 0.95,
                "speaker": "narrator",
                "word_level": [
                    {"word": "Complex", "start": 0.0, "end": 0.8},
                    {"word": "segment", "start": 0.8, "end": 1.6}
                ]
            }
        ]
        
        transcript = TranscriptFactory.build(segments=complex_segments)
        timestamps = transcript.extract_timestamps()
        
        assert len(timestamps) == 1
        assert timestamps[0]["text"] == "Complex segment with metadata"
        assert timestamps[0]["start"] == 0.0
        assert timestamps[0]["end"] == 3.5


class TestTranscriptValidation:
    """Test suite for Transcript model validation."""
    
    @pytest.mark.asyncio
    async def test_invalid_source_enum(self, db_session):
        """Test that invalid source enum values are rejected."""
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        with pytest.raises((ValueError, IntegrityError)):
            transcript = Transcript(
                video_id=video.id,
                content="Test content",
                source="invalid_source"  # Invalid enum value
            )
            db_session.add(transcript)
            await db_session.commit()
    
    @pytest.mark.asyncio
    async def test_invalid_language_enum(self, db_session):
        """Test that invalid language enum values are rejected."""
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        with pytest.raises((ValueError, IntegrityError)):
            transcript = Transcript(
                video_id=video.id,
                content="Test content",
                language="invalid_language"  # Invalid enum value
            )
            db_session.add(transcript)
            await db_session.commit()
    
    @pytest.mark.asyncio
    async def test_foreign_key_constraint(self, db_session):
        """Test that video_id must reference existing video."""
        from uuid import uuid4
        
        with pytest.raises(IntegrityError):
            transcript = Transcript(
                video_id=uuid4(),  # Non-existent video
                content="Test content"
            )
            db_session.add(transcript)
            await db_session.commit()


class TestTranscriptRelationships:
    """Test suite for Transcript model relationships."""
    
    @pytest.mark.asyncio
    async def test_transcript_video_relationship(self, db_session):
        """Test that video relationship works correctly."""
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        # Create transcript
        transcript = TranscriptFactory.build(video_id=video.id)
        db_session.add(transcript)
        await db_session.commit()
        
        # Test relationship
        await db_session.refresh(transcript)
        assert transcript.video is not None
        assert transcript.video.id == video.id
        assert transcript.video.title == video.title
    
    @pytest.mark.asyncio
    async def test_video_delete_cascades_to_transcript(self, db_session):
        """Test that deleting video cascades to transcript."""
        # Create channel, video, and transcript
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        transcript = TranscriptFactory.build(video_id=video.id)
        db_session.add(transcript)
        await db_session.commit()
        
        transcript_id = transcript.id
        
        # Delete video
        await db_session.delete(video)
        await db_session.commit()
        
        # Transcript should be deleted too (cascade)
        result = await db_session.execute(select(Transcript).filter(Transcript.id == transcript_id))
        deleted_transcript = result.scalar_one_or_none()
        assert deleted_transcript is None
    
    @pytest.mark.asyncio
    async def test_multiple_transcripts_per_video(self, db_session):
        """Test that a video can have multiple transcripts."""
        # Create channel and video
        channel = ChannelFactory.build()
        db_session.add(channel)
        await db_session.flush()
        
        video = VideoFactory.build(channel_id=channel.id)
        db_session.add(video)
        await db_session.flush()
        
        # Create multiple transcripts for the same video
        transcript_en = TranscriptFactory.build(
            video_id=video.id,
            language=TranscriptLanguage.EN,
            language_code="en",
            content="English transcript"
        )
        transcript_es = TranscriptFactory.build(
            video_id=video.id,
            language=TranscriptLanguage.ES,
            language_code="es",
            content="Spanish transcript"
        )
        
        db_session.add(transcript_en)
        db_session.add(transcript_es)
        await db_session.commit()
        
        # Both transcripts should exist and reference the same video
        result = await db_session.execute(
            select(Transcript).filter(Transcript.video_id == video.id)
        )
        transcripts = result.scalars().all()
        
        assert len(transcripts) == 2
        assert all(t.video_id == video.id for t in transcripts)
        
        languages = [t.language for t in transcripts]
        assert TranscriptLanguage.EN in languages
        assert TranscriptLanguage.ES in languages