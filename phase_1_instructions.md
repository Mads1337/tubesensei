# TubeSensei Phase 1: Core Infrastructure Development

## Project Context
You are developing TubeSensei, a YouTube content analysis platform that discovers, transcribes, and extracts business ideas from video content. This is Phase 1 of a 4-phase development plan focusing on core infrastructure.

## Phase 1 Scope (4 weeks)
Build the foundational infrastructure including database layer, YouTube API integration, transcript processing, and job queue system. This phase enables bulk discovery and transcript extraction of 500-1000 videos, preparing them for AI-powered filtering and idea extraction in Phase 2.

## Technical Requirements

### Technology Stack
- **Python 3.11+** with FastAPI framework
- **PostgreSQL 14+** with SQLAlchemy 2.0 (async)
- **Redis** for job queuing and caching
- **Celery** for distributed task processing
- **YouTube Data API v3** and youtube-transcript-api
- **Pydantic V2** for data validation
- **Alembic** for database migrations

### Project Structure
```
tubesensei/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration management
│   ├── database.py            # Database setup and connection
│   ├── models/                # SQLAlchemy models
│   │   ├── __init__.py
│   │   ├── channel.py
│   │   ├── video.py
│   │   ├── transcript.py
│   │   ├── processing_job.py
│   │   └── processing_session.py
│   ├── services/              # Business logic services
│   │   ├── __init__.py
│   │   ├── channel_manager.py
│   │   ├── video_discovery.py
│   │   ├── transcript_processor.py
│   │   └── job_queue.py
│   ├── integrations/          # External API integrations
│   │   ├── __init__.py
│   │   ├── youtube_api.py
│   │   └── transcript_api.py
│   ├── workers/               # Celery workers
│   │   ├── __init__.py
│   │   └── processing_tasks.py
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── logging.py
│       └── exceptions.py
├── alembic/                   # Database migrations
├── tests/                     # Test suite
├── requirements.txt
├── docker-compose.yml         # Local development setup
├── .env.example
└── README.md
```

## Implementation Tasks

### Week 1: Database and Basic Setup

#### Task 1.1: Project Setup
- Initialize Python project with Poetry or pip
- Create project structure as outlined above
- Set up development dependencies (pytest, black, ruff, mypy)
- Create Docker Compose for PostgreSQL and Redis
- Set up environment configuration with Pydantic Settings

#### Task 1.2: Database Models
Implement SQLAlchemy models with these exact schemas:

**channels table:**
```python
- id: UUID (Primary Key)
- name: str
- youtube_channel_id: str (Unique)
- channel_url: str
- description: Optional[str]
- subscriber_count: Optional[int]
- video_count: Optional[int]
- status: Enum (active, paused, inactive)
- last_checked_at: Optional[datetime]
- created_at: datetime
- updated_at: datetime
- metadata: dict (JSONB)
- processing_config: dict (JSONB)
```

**videos table:**
```python
- id: UUID (Primary Key)
- channel_id: UUID (Foreign Key)
- youtube_video_id: str (Unique)
- title: str
- description: Optional[str]
- duration_seconds: Optional[int]
- view_count: Optional[int]
- like_count: Optional[int]
- comment_count: Optional[int]
- published_at: Optional[datetime]
- thumbnail_url: Optional[str]
- video_url: str
- status: Enum (discovered, queued, processing, completed, failed)
- processing_priority: int (default 0)
- created_at: datetime
- updated_at: datetime
- metadata: dict (JSONB)
- tags: List[str]
```

**transcripts table:**
```python
- id: UUID (Primary Key)
- video_id: UUID (Foreign Key)
- content: str
- language: Optional[str]
- source: Enum (youtube_auto, youtube_manual)
- confidence_score: Optional[float]
- word_count: Optional[int]
- created_at: datetime
- metadata: dict (JSONB)
```

**processing_jobs table:**
```python
- id: UUID (Primary Key)
- job_type: Enum (channel_discovery, transcript_extraction)
- status: Enum (queued, running, completed, failed, cancelled)
- entity_id: UUID
- entity_type: str
- priority: int (default 0)
- attempts: int (default 0)
- max_attempts: int (default 3)
- started_at: Optional[datetime]
- completed_at: Optional[datetime]
- error_message: Optional[str]
- created_at: datetime
- metadata: dict (JSONB)
- result: dict (JSONB)
```

**processing_sessions table:**
```python
- id: UUID (Primary Key)
- session_name: str
- session_type: Enum (bulk_processing, manual)
- status: Enum (running, completed, failed, paused)
- total_videos: Optional[int]
- processed_videos: int (default 0)
- started_at: Optional[datetime]
- completed_at: Optional[datetime]
- created_at: datetime
- configuration: dict (JSONB)
- metrics: dict (JSONB)
```

#### Task 1.3: Database Setup
- Create Alembic configuration
- Generate initial migration with all tables
- Add required indexes for performance
- Set up database connection with async SQLAlchemy
- Create database utility functions (CRUD operations)

### Week 2: YouTube Integration

#### Task 2.1: YouTube Data API Integration
- Set up Google API client with authentication
- Implement quota management and rate limiting
- Create service class for YouTube API operations:
  - Get channel information
  - List channel videos
  - Get video metadata
  - Handle API errors and retries

#### Task 2.2: Channel Management
- Create ChannelManager service class
- Implement functions:
  - `add_channel(channel_url: str)` - Add new channel
  - `sync_channel_metadata(channel_id: UUID)` - Update channel info
  - `discover_channel_videos(channel_id: UUID)` - Fetch all videos
  - `get_channel_status(channel_id: UUID)` - Check channel health

#### Task 2.3: Video Discovery
- Create VideoDiscovery service class  
- Implement batch video processing
- Add basic filtering (duration, age, etc.)
- Store video metadata in database
- Handle duplicate video detection
- Queue all discovered videos for future processing

### Week 3: Transcript Processing

#### Task 3.1: Transcript API Integration
- Integrate youtube-transcript-api
- Handle multiple transcript sources (auto, manual)
- Implement language detection
- Add error handling for unavailable transcripts

#### Task 3.2: Transcript Processing Service
- Create TranscriptProcessor service class
- Implement functions:
  - `extract_transcript(video_id: UUID)` - Get transcript from YouTube
  - `clean_transcript(content: str)` - Clean and format text
  - `store_transcript(video_id: UUID, content: str)` - Save to database
  - `batch_process_transcripts(video_ids: List[UUID])` - Bulk processing

#### Task 3.3: Content Processing
- Add text cleaning and normalization
- Implement word count and quality metrics
- Store processed transcripts with metadata
- Add caching to avoid reprocessing

### Week 4: Job Queue System

#### Task 4.1: Redis and Celery Setup
- Configure Redis for job queuing and caching
- Set up Celery with Redis as broker
- Create worker configuration
- Add job monitoring and status tracking

#### Task 4.2: Processing Tasks
Create Celery tasks for:
- `discover_channel_videos_task(channel_id: UUID)`
- `extract_transcript_task(video_id: UUID)`
- `batch_process_transcripts_task(video_ids: List[UUID])`
- `sync_channel_metadata_task(channel_id: UUID)`

#### Task 4.3: Job Management
- Implement job creation and queuing
- Add retry logic with exponential backoff
- Create job status tracking
- Add job cancellation and cleanup

## Configuration Requirements

### Environment Variables
```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5433/tubesensei
DATABASE_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379/0

# YouTube API
YOUTUBE_API_KEY=your_youtube_api_key
YOUTUBE_QUOTA_PER_DAY=10000

# Processing
MAX_CONCURRENT_JOBS=10
MAX_VIDEOS_PER_BATCH=50
TRANSCRIPT_TIMEOUT_SECONDS=300

# Logging
LOG_LEVEL=INFO
```

### Pydantic Settings
Create configuration classes using Pydantic BaseSettings for type safety and validation.

## Testing Requirements

### Unit Tests
- Test all database models and relationships
- Test YouTube API integration with mocked responses
- Test transcript processing with sample data
- Test job queue functionality
- Minimum 80% code coverage

### Integration Tests
- Test complete video discovery workflow
- Test transcript extraction end-to-end
- Test job processing with real Redis/Celery
- Test error handling and retry logic

## Acceptance Criteria

Phase 1 is complete when:

1. **Database Operations**: Successfully create, read, update all core entities (channels, videos, transcripts, jobs, processing_sessions)
2. **YouTube Integration**: Fetch channel metadata and video lists with proper quota management and rate limiting
3. **Transcript Processing**: Extract transcripts from 90%+ of videos with available captions
4. **Job Queue System**: Queue, execute, and track jobs with retry logic and status monitoring
5. **Bulk Processing Performance**: Process 100+ videos per hour during transcript extraction
6. **Concurrent Operations**: Support 10+ concurrent jobs without data corruption
7. **Error Handling**: Gracefully handle all API failures, missing transcripts, quota limits, and network issues
8. **System Performance**: Maintain <100ms database query times for indexed operations
9. **Tests**: Full test suite passes with 80%+ code coverage
10. **Documentation**: Clear README with setup, configuration, and usage instructions

## Implementation Notes

1. **Database First**: Start with solid database foundation before integrations
2. **Error Handling**: Implement robust error handling from the beginning
3. **Async Operations**: Use async/await for all I/O operations
4. **Logging**: Add structured logging throughout
5. **Configuration**: Use environment-based configuration for flexibility
6. **Testing**: Write tests as you develop, don't leave for the end

## Success Metrics
- **Processing Rate**: 100+ videos per hour during transcript extraction
- **Transcript Success**: Extract transcripts from 90%+ of videos with available captions
- **System Performance**: <100ms response time for database operations
- **Concurrent Processing**: Support 10+ simultaneous jobs without data corruption  
- **Error Recovery**: Automatic retry and recovery from transient failures
- **Data Integrity**: Zero data corruption or loss during bulk operations
- **Test Coverage**: 80%+ code coverage with all tests passing

This infrastructure will serve as the foundation for Phase 2 (AI Integration) where we'll add LLM-powered video filtering and idea extraction capabilities.