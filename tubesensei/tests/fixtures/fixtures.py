"""
Test fixtures for TubeSensei testing.

Provides comprehensive test data including channels, videos, transcripts,
processing jobs, and mock API responses.
"""

import factory
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from uuid import uuid4
import json

from app.models.channel import Channel, ChannelStatus
from app.models.video import Video, VideoStatus
from app.models.transcript import Transcript, TranscriptSource, TranscriptLanguage
from app.models.processing_job import ProcessingJob, JobType, JobStatus, JobPriority
from app.models.processing_session import ProcessingSession, SessionType, SessionStatus


class ChannelFactory(factory.Factory):
    """Factory for creating Channel test instances."""
    
    class Meta:
        model = Channel
    
    youtube_channel_id = factory.Sequence(lambda n: f"UC{'0' * 20}{n:02d}")
    channel_name = factory.Faker("company")
    channel_handle = factory.LazyAttribute(lambda obj: f"@{obj.channel_name.lower().replace(' ', '')}")
    description = factory.Faker("text", max_nb_chars=500)
    subscriber_count = factory.Faker("random_int", min=1000, max=10000000)
    video_count = factory.Faker("random_int", min=10, max=5000)
    view_count = factory.Faker("random_int", min=100000, max=1000000000)
    country = factory.Faker("country_code")
    custom_url = factory.LazyAttribute(lambda obj: f"/{obj.channel_handle}")
    published_at = factory.Faker("date_time_between", start_date="-10y", end_date="-1y", tzinfo=timezone.utc)
    thumbnail_url = factory.LazyAttribute(lambda obj: f"https://yt3.ggpht.com/{obj.youtube_channel_id}/default.jpg")
    status = ChannelStatus.ACTIVE
    priority_level = 5
    check_frequency_hours = 24
    last_checked_at = factory.Faker("date_time_between", start_date="-7d", end_date="now", tzinfo=timezone.utc)
    last_video_published_at = factory.Faker("date_time_between", start_date="-30d", end_date="-1d", tzinfo=timezone.utc)
    metadata = factory.LazyFunction(lambda: {
        "keywords": ["technology", "programming", "tutorial"],
        "featured_channels": [],
        "branding": {"banner_color": "#ff0000"}
    })
    processing_config = factory.LazyFunction(lambda: {
        "auto_transcript": True,
        "languages": ["en"],
        "quality_threshold": 0.8
    })
    auto_process = True
    tags = factory.LazyFunction(lambda: ["tech", "education"])
    notes = factory.Faker("sentence")


class VideoFactory(factory.Factory):
    """Factory for creating Video test instances."""
    
    class Meta:
        model = Video
    
    youtube_video_id = factory.Sequence(lambda n: f"dQw4w9WgXcQ{n:03d}")
    channel_id = factory.LazyFunction(uuid4)
    title = factory.Faker("sentence", nb_words=8)
    description = factory.Faker("text", max_nb_chars=1000)
    thumbnail_url = factory.LazyAttribute(lambda obj: f"https://i.ytimg.com/vi/{obj.youtube_video_id}/hqdefault.jpg")
    duration_seconds = factory.Faker("random_int", min=60, max=3600)
    view_count = factory.Faker("random_int", min=100, max=10000000)
    like_count = factory.Faker("random_int", min=10, max=100000)
    comment_count = factory.Faker("random_int", min=0, max=10000)
    published_at = factory.Faker("date_time_between", start_date="-1y", end_date="-1d", tzinfo=timezone.utc)
    discovered_at = factory.Faker("date_time_between", start_date="-7d", end_date="now", tzinfo=timezone.utc)
    processed_at = None
    status = VideoStatus.DISCOVERED
    is_valuable = None
    valuable_score = None
    valuable_reason = None
    tags = factory.LazyFunction(lambda: ["tutorial", "how-to", "programming"])
    category_id = "22"  # People & Blogs
    language = "en"
    has_captions = True
    caption_languages = factory.LazyFunction(lambda: ["en", "en-US"])
    metadata = factory.LazyFunction(lambda: {
        "definition": "hd",
        "dimension": "2d",
        "licensed_content": False
    })
    processing_metadata = factory.LazyFunction(dict)
    error_message = None
    retry_count = 0


class TranscriptFactory(factory.Factory):
    """Factory for creating Transcript test instances."""
    
    class Meta:
        model = Transcript
    
    video_id = factory.LazyFunction(uuid4)
    content = factory.Faker("text", max_nb_chars=5000)
    source = TranscriptSource.YOUTUBE_AUTO
    language = TranscriptLanguage.EN
    language_code = "en"
    is_auto_generated = True
    word_count = factory.LazyAttribute(lambda obj: len(obj.content.split()) if obj.content else 0)
    char_count = factory.LazyAttribute(lambda obj: len(obj.content) if obj.content else 0)
    confidence_score = factory.Faker("random_int", min=80, max=100)
    metadata = factory.LazyFunction(lambda: {
        "api_version": "v3",
        "extraction_method": "youtube_transcript_api"
    })
    segments = factory.LazyFunction(lambda: [
        {"start": 0.0, "end": 5.0, "text": "Welcome to this video tutorial"},
        {"start": 5.0, "end": 10.0, "text": "Today we'll learn about programming"},
    ])
    processed_content = None
    is_complete = True


class ProcessingJobFactory(factory.Factory):
    """Factory for creating ProcessingJob test instances."""
    
    class Meta:
        model = ProcessingJob
    
    job_type = JobType.TRANSCRIPT_EXTRACTION
    status = JobStatus.PENDING
    priority = JobPriority.NORMAL
    entity_type = "video"
    entity_id = factory.LazyFunction(uuid4)
    session_id = None
    started_at = None
    completed_at = None
    scheduled_at = factory.LazyFunction(lambda: datetime.now(timezone.utc))
    retry_count = 0
    max_retries = 3
    retry_after = None
    progress_percent = 0.0
    progress_message = None
    input_data = factory.LazyFunction(dict)
    output_data = None
    error_message = None
    error_details = None
    metadata = factory.LazyFunction(dict)
    worker_id = None
    execution_time_seconds = None


class ProcessingSessionFactory(factory.Factory):
    """Factory for creating ProcessingSession test instances."""
    
    class Meta:
        model = ProcessingSession
    
    name = factory.Faker("sentence", nb_words=3)
    session_type = SessionType.CHANNEL_DISCOVERY
    status = SessionStatus.PENDING
    total_jobs = 0
    completed_jobs = 0
    failed_jobs = 0
    started_at = None
    completed_at = None
    metadata = factory.LazyFunction(dict)
    config = factory.LazyFunction(lambda: {
        "batch_size": 50,
        "timeout_minutes": 30
    })


# Mock API responses for testing


class MockYouTubeAPIResponses:
    """Mock responses for YouTube API calls."""
    
    @staticmethod
    def channel_info(channel_id: str = "UC_test_channel_id") -> Dict[str, Any]:
        """Mock channel information response."""
        return {
            'kind': 'youtube#channelListResponse',
            'items': [{
                'id': channel_id,
                'kind': 'youtube#channel',
                'snippet': {
                    'title': 'Test Channel',
                    'description': 'A test channel for unit testing',
                    'customUrl': '@testchannel',
                    'publishedAt': '2020-01-01T00:00:00Z',
                    'country': 'US',
                    'thumbnails': {
                        'default': {'url': 'https://yt3.ggpht.com/default.jpg'},
                        'medium': {'url': 'https://yt3.ggpht.com/medium.jpg'},
                        'high': {'url': 'https://yt3.ggpht.com/high.jpg'}
                    }
                },
                'statistics': {
                    'viewCount': '1000000',
                    'subscriberCount': '100000',
                    'videoCount': '500'
                },
                'contentDetails': {
                    'relatedPlaylists': {
                        'uploads': 'UU_test_uploads_playlist'
                    }
                },
                'brandingSettings': {
                    'channel': {
                        'keywords': 'technology programming tutorial'
                    }
                }
            }]
        }
    
    @staticmethod
    def playlist_items(playlist_id: str = "UU_test_uploads_playlist") -> Dict[str, Any]:
        """Mock playlist items response."""
        return {
            'kind': 'youtube#playlistItemListResponse',
            'items': [
                {
                    'id': 'PLtest001',
                    'kind': 'youtube#playlistItem',
                    'snippet': {
                        'title': 'Test Video 1',
                        'description': 'First test video description',
                        'thumbnails': {
                            'default': {'url': 'https://i.ytimg.com/vi/test001/default.jpg'},
                            'high': {'url': 'https://i.ytimg.com/vi/test001/hqdefault.jpg'}
                        },
                        'channelId': 'UC_test_channel_id'
                    },
                    'contentDetails': {
                        'videoId': 'test_video_001',
                        'videoPublishedAt': '2023-01-01T12:00:00Z'
                    }
                },
                {
                    'id': 'PLtest002',
                    'kind': 'youtube#playlistItem',
                    'snippet': {
                        'title': 'Test Video 2',
                        'description': 'Second test video description',
                        'thumbnails': {
                            'default': {'url': 'https://i.ytimg.com/vi/test002/default.jpg'},
                            'high': {'url': 'https://i.ytimg.com/vi/test002/hqdefault.jpg'}
                        },
                        'channelId': 'UC_test_channel_id'
                    },
                    'contentDetails': {
                        'videoId': 'test_video_002',
                        'videoPublishedAt': '2023-01-02T12:00:00Z'
                    }
                }
            ],
            'nextPageToken': None
        }
    
    @staticmethod
    def video_details(video_ids: List[str] = None) -> Dict[str, Any]:
        """Mock video details response."""
        if not video_ids:
            video_ids = ['test_video_001']
        
        items = []
        for i, video_id in enumerate(video_ids):
            items.append({
                'id': video_id,
                'kind': 'youtube#video',
                'snippet': {
                    'title': f'Test Video {i+1}',
                    'description': f'Description for test video {i+1}',
                    'channelId': 'UC_test_channel_id',
                    'channelTitle': 'Test Channel',
                    'publishedAt': f'2023-01-{i+1:02d}T12:00:00Z',
                    'tags': ['tutorial', 'programming', 'test'],
                    'categoryId': '22',
                    'defaultLanguage': 'en',
                    'thumbnails': {
                        'default': {'url': f'https://i.ytimg.com/vi/{video_id}/default.jpg'},
                        'high': {'url': f'https://i.ytimg.com/vi/{video_id}/hqdefault.jpg'}
                    }
                },
                'contentDetails': {
                    'duration': 'PT10M30S',
                    'caption': 'true'
                },
                'statistics': {
                    'viewCount': str(10000 * (i + 1)),
                    'likeCount': str(100 * (i + 1)),
                    'commentCount': str(10 * (i + 1))
                },
                'status': {
                    'privacyStatus': 'public'
                }
            })
        
        return {
            'kind': 'youtube#videoListResponse',
            'items': items
        }


class MockTranscriptAPIResponses:
    """Mock responses for YouTube Transcript API calls."""
    
    @staticmethod
    def transcript_list(video_id: str = "test_video_001") -> List[Dict[str, Any]]:
        """Mock transcript list response."""
        return [
            {
                'language': 'en',
                'language_code': 'en',
                'is_generated': True,
                'is_translatable': True
            },
            {
                'language': 'es',
                'language_code': 'es',
                'is_generated': False,
                'is_translatable': True
            }
        ]
    
    @staticmethod
    def transcript_content(video_id: str = "test_video_001", language: str = "en") -> List[Dict[str, Any]]:
        """Mock transcript content response."""
        return [
            {'text': 'Welcome to this tutorial.', 'start': 0.0, 'duration': 3.0},
            {'text': 'Today we will learn about programming.', 'start': 3.0, 'duration': 4.0},
            {'text': 'Let\'s start with the basics.', 'start': 7.0, 'duration': 3.0},
            {'text': 'First, we need to understand variables.', 'start': 10.0, 'duration': 4.0},
            {'text': 'Variables store data in your program.', 'start': 14.0, 'duration': 4.0},
        ]


# Test database fixtures

class TestDataBuilder:
    """Builder for creating comprehensive test datasets."""
    
    @staticmethod
    def create_test_channel_with_videos(
        db_session,
        channel_count: int = 1,
        videos_per_channel: int = 5,
        transcripts_per_video: int = 1
    ) -> Dict[str, Any]:
        """Create test channels with associated videos and transcripts."""
        channels = []
        videos = []
        transcripts = []
        
        for i in range(channel_count):
            # Create channel
            channel = ChannelFactory.build()
            db_session.add(channel)
            db_session.flush()  # Get the ID
            channels.append(channel)
            
            # Create videos for this channel
            for j in range(videos_per_channel):
                video = VideoFactory.build(channel_id=channel.id)
                db_session.add(video)
                db_session.flush()  # Get the ID
                videos.append(video)
                
                # Create transcripts for this video
                for k in range(transcripts_per_video):
                    transcript = TranscriptFactory.build(video_id=video.id)
                    db_session.add(transcript)
                    transcripts.append(transcript)
        
        db_session.commit()
        
        return {
            'channels': channels,
            'videos': videos,
            'transcripts': transcripts
        }
    
    @staticmethod
    def create_processing_jobs(
        db_session,
        job_count: int = 10,
        job_types: Optional[List[JobType]] = None,
        job_statuses: Optional[List[JobStatus]] = None
    ) -> List[ProcessingJob]:
        """Create test processing jobs with various types and statuses."""
        if not job_types:
            job_types = [JobType.TRANSCRIPT_EXTRACTION, JobType.VIDEO_DISCOVERY]
        if not job_statuses:
            job_statuses = [JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED]
        
        jobs = []
        for i in range(job_count):
            job = ProcessingJobFactory.build(
                job_type=job_types[i % len(job_types)],
                status=job_statuses[i % len(job_statuses)]
            )
            db_session.add(job)
            jobs.append(job)
        
        db_session.commit()
        return jobs


# Celery test fixtures

class CeleryTestConfig:
    """Configuration for Celery testing."""
    
    broker_url = "memory://"
    result_backend = "cache+memory://"
    task_always_eager = True
    task_eager_propagates = True
    task_store_eager_result = True
    
    # Disable rate limiting for tests
    task_annotations = {
        '*': {'rate_limit': None}
    }