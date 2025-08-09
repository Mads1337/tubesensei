import asyncio
import pytest
import pytest_asyncio
import os
import warnings
from typing import AsyncGenerator
from unittest.mock import AsyncMock, Mock
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from datetime import datetime, timezone

from app.database import Base
from app.config import settings
from tests.fixtures.fixtures import (
    ChannelFactory,
    VideoFactory,
    TranscriptFactory,
    ProcessingJobFactory,
    MockYouTubeAPIResponses,
    MockTranscriptAPIResponses,
    TestDataBuilder,
    CeleryTestConfig
)

# Test database configuration
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    settings.DATABASE_URL.replace("/tubesensei", "/tubesensei_test")
)

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=NullPool,
    echo=False,
    future=True
)

TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Configure test markers
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock",
]

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (deselect with '-m \"not integration\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests (deselect with '-m \"not performance\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    
    # Filter warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        
        # Mark async tests as asyncio
        if "async" in item.name or item.get_closest_marker("asyncio"):
            item.add_marker(pytest.mark.asyncio)


# Event loop configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


# Database fixtures
@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing with automatic cleanup."""
    # Create tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    # Create session
    async with TestSessionLocal() as session:
        try:
            yield session
        finally:
            await session.rollback()
            await session.close()
    
    # Cleanup tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def clean_db() -> AsyncGenerator[None, None]:
    """Provide a clean database without a session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    yield
    
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def db_session_with_data(db_session) -> AsyncGenerator[AsyncSession, None]:
    """Database session with pre-populated test data."""
    # Create comprehensive test data
    test_data = TestDataBuilder.create_test_channel_with_videos(
        db_session,
        channel_count=2,
        videos_per_channel=3,
        transcripts_per_video=1
    )
    
    yield db_session


# Factory fixtures
@pytest.fixture
def channel_factory():
    """Channel factory for creating test channels."""
    return ChannelFactory


@pytest.fixture
def video_factory():
    """Video factory for creating test videos."""
    return VideoFactory


@pytest.fixture
def transcript_factory():
    """Transcript factory for creating test transcripts."""
    return TranscriptFactory


@pytest.fixture
def processing_job_factory():
    """Processing job factory for creating test jobs."""
    return ProcessingJobFactory


# Mock API response fixtures
@pytest.fixture
def mock_youtube_api_responses():
    """Mock YouTube API responses."""
    return MockYouTubeAPIResponses


@pytest.fixture
def mock_transcript_api_responses():
    """Mock transcript API responses."""
    return MockTranscriptAPIResponses


# Service mock fixtures
@pytest.fixture
def mock_youtube_client():
    """Mock YouTube API client."""
    client = AsyncMock()
    client.get_channel_info.return_value = MockYouTubeAPIResponses.channel_info()['items'][0]
    client.list_channel_videos.return_value = MockYouTubeAPIResponses.playlist_items()['items']
    client.get_video_details.return_value = MockYouTubeAPIResponses.video_details(['test_video_001'])['items']
    client.search_videos.return_value = []
    return client


@pytest.fixture
def mock_transcript_api():
    """Mock transcript API client."""
    api = AsyncMock()
    api.get_transcript.return_value = {
        'video_id': 'test_video_001',
        'content': 'Test transcript content',
        'language': 'en',
        'segments': MockTranscriptAPIResponses.transcript_content('test_video_001')
    }
    api.get_available_languages.return_value = MockTranscriptAPIResponses.transcript_list('test_video_001')
    return api


# Test data builder fixture
@pytest.fixture
def test_data_builder():
    """Test data builder utility."""
    return TestDataBuilder


# Time-related fixtures
@pytest.fixture
def mock_datetime_now(monkeypatch):
    """Mock datetime.now for consistent testing."""
    fixed_time = datetime(2023, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            if tz:
                return fixed_time.replace(tzinfo=tz)
            return fixed_time.replace(tzinfo=None)
        
        @classmethod
        def utcnow(cls):
            return fixed_time.replace(tzinfo=None)
    
    monkeypatch.setattr("datetime.datetime", MockDatetime)
    return fixed_time


# Configuration fixtures
@pytest.fixture
def test_settings(monkeypatch):
    """Override settings for testing."""
    test_config = {
        'DATABASE_URL': TEST_DATABASE_URL,
        'YOUTUBE_API_KEY': 'test_api_key',
        'YOUTUBE_QUOTA_PER_DAY': 1000,
        'RATE_LIMIT_REQUESTS_PER_MINUTE': 60,
        'TRANSCRIPT_BATCH_SIZE': 10,
        'TRANSCRIPT_MAX_RETRIES': 3,
        'YOUTUBE_REQUEST_TIMEOUT': 30,
        'YOUTUBE_CACHE_TTL': 3600
    }
    
    for key, value in test_config.items():
        monkeypatch.setattr(f"app.config.settings.{key}", value)
    
    return test_config


# Celery testing fixtures
@pytest.fixture
def celery_test_config():
    """Celery configuration for testing."""
    return CeleryTestConfig


@pytest.fixture
def mock_celery_app(celery_test_config):
    """Mock Celery app for testing."""
    from celery import Celery
    
    app = Celery('test_app')
    app.config_from_object(celery_test_config)
    return app


# Performance testing fixtures
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.process = psutil.Process(os.getpid())
            self.start_time = None
            self.start_memory = None
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            if self.start_time is None:
                return {"error": "Monitor not started"}
            
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            return {
                "duration": end_time - self.start_time,
                "memory_start": self.start_memory,
                "memory_end": end_memory,
                "memory_delta": end_memory - self.start_memory
            }
    
    return PerformanceMonitor()


# Error injection fixtures for testing resilience
@pytest.fixture
def error_injector():
    """Utility for injecting errors during tests."""
    class ErrorInjector:
        def __init__(self):
            self.error_count = 0
            self.max_errors = 0
            self.error_type = Exception
            self.error_message = "Injected error"
        
        def configure(self, max_errors=1, error_type=Exception, message="Injected error"):
            self.max_errors = max_errors
            self.error_type = error_type
            self.error_message = message
            self.error_count = 0
        
        def maybe_raise_error(self):
            if self.error_count < self.max_errors:
                self.error_count += 1
                raise self.error_type(self.error_message)
    
    return ErrorInjector()


# Async test utilities
@pytest.fixture
def async_test_utils():
    """Utilities for async testing."""
    class AsyncTestUtils:
        @staticmethod
        async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
            """Wait for a condition to become true."""
            import asyncio
            
            end_time = asyncio.get_event_loop().time() + timeout
            while asyncio.get_event_loop().time() < end_time:
                if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                    return True
                await asyncio.sleep(interval)
            return False
        
        @staticmethod
        async def run_with_timeout(coro, timeout=10.0):
            """Run coroutine with timeout."""
            return await asyncio.wait_for(coro, timeout=timeout)
    
    return AsyncTestUtils