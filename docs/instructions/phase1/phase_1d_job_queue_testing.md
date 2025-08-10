# Phase 1D: Job Queue & Testing

## Overview
This phase implements the distributed job queue system with Celery and comprehensive testing. Expected duration: 1 week.

## Prerequisites
- Phase 1A-1C completed
- Redis running (from docker-compose)
- All core services implemented

## Goals
- Set up Celery with Redis broker
- Create async task processing
- Implement job management
- Complete comprehensive testing
- Achieve 80%+ code coverage

## Task 1: Celery Configuration

### 1.1 Install Dependencies
Add to `requirements.txt`:
```
celery[redis]==5.3.4
flower==2.0.1
celery-redbeat==2.1.0
prometheus-client==0.19.0
```

### 1.2 Celery Configuration
`app/celery_app.py`:
```python
from celery import Celery
from app.config import settings

celery_app = Celery(
    "tubesensei",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.processing_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Task routing
celery_app.conf.task_routes = {
    "app.workers.processing_tasks.discover_channel_videos_task": {
        "queue": "discovery"
    },
    "app.workers.processing_tasks.extract_transcript_task": {
        "queue": "transcripts"
    },
    "app.workers.processing_tasks.batch_process_transcripts_task": {
        "queue": "batch"
    },
}

# Rate limits
celery_app.conf.task_annotations = {
    "app.workers.processing_tasks.extract_transcript_task": {
        "rate_limit": "30/m"
    },
}
```

### 1.3 Update Environment
`.env`:
```env
# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
MAX_CONCURRENT_JOBS=10
MAX_VIDEOS_PER_BATCH=50
WORKER_CONCURRENCY=4
```

## Task 2: Processing Tasks

### 2.1 Core Tasks Implementation
`app/workers/processing_tasks.py`:

```python
from celery import Task
from app.celery_app import celery_app
from typing import List, UUID, Dict, Any

class CallbackTask(Task):
    """Base task with callbacks"""
    def on_success(self, retval, task_id, args, kwargs):
        """Success callback"""
        update_job_status(task_id, "completed", result=retval)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Failure callback"""
        update_job_status(task_id, "failed", error=str(exc))

@celery_app.task(
    base=CallbackTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60
)
async def discover_channel_videos_task(
    self,
    channel_id: UUID
) -> Dict[str, Any]:
    """
    Discover all videos from a channel
    """
    try:
        # Update job status to running
        # Get channel from database
        # Call video discovery service
        # Store discovered videos
        # Queue transcript extraction
        # Return summary
    except Exception as exc:
        # Log error
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

@celery_app.task(
    base=CallbackTask,
    bind=True,
    max_retries=3
)
async def extract_transcript_task(
    self,
    video_id: UUID
) -> Dict[str, Any]:
    """
    Extract transcript for a single video
    """
    try:
        # Get video from database
        # Extract transcript
        # Process and clean
        # Store in database
        # Update video status
        # Return result
    except TranscriptNotAvailableError:
        # Mark as unavailable
        # Don't retry
    except Exception as exc:
        # Retry with backoff
        raise self.retry(exc=exc)

@celery_app.task(
    base=CallbackTask,
    bind=True
)
async def batch_process_transcripts_task(
    self,
    video_ids: List[UUID],
    session_id: UUID
) -> Dict[str, Any]:
    """
    Process multiple transcripts in batch
    """
    results = {
        "successful": [],
        "failed": [],
        "total": len(video_ids)
    }
    
    # Create subtasks
    # Process with chord/group
    # Track progress
    # Update session
    # Return results

@celery_app.task(
    bind=True,
    max_retries=3
)
async def sync_channel_metadata_task(
    self,
    channel_id: UUID
) -> Dict[str, Any]:
    """
    Update channel metadata from YouTube
    """
    # Fetch latest channel info
    # Update database
    # Return updated metadata
```

### 2.2 Task Monitoring
`app/workers/monitoring.py`:

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
task_counter = Counter(
    'celery_task_total',
    'Total number of tasks',
    ['task_name', 'status']
)

task_duration = Histogram(
    'celery_task_duration_seconds',
    'Task execution time',
    ['task_name']
)

queue_size = Gauge(
    'celery_queue_size',
    'Number of tasks in queue',
    ['queue_name']
)

class TaskMonitor:
    @staticmethod
    def record_task_start(task_name: str):
        # Record start time
        # Update metrics
    
    @staticmethod
    def record_task_complete(task_name: str, duration: float):
        # Update counters
        # Record duration
    
    @staticmethod
    def get_queue_stats() -> dict:
        # Get queue sizes
        # Active tasks
        # Failed tasks
        # Return stats
```

## Task 3: Job Management System

### 3.1 Job Queue Service
`app/services/job_queue.py`:

```python
class JobQueueService:
    def __init__(self):
        self.celery = celery_app
        
    async def create_job(
        self,
        job_type: str,
        entity_id: UUID,
        priority: int = 0,
        metadata: Dict = None
    ) -> ProcessingJob:
        # Create job record
        # Queue to Celery
        # Return job object
    
    async def queue_video_discovery(
        self,
        channel_id: UUID,
        priority: int = 0
    ) -> str:
        # Create job record
        # Queue discovery task
        # Return task ID
    
    async def queue_transcript_extraction(
        self,
        video_id: UUID,
        priority: int = 0
    ) -> str:
        # Create job record
        # Queue extraction task
        # Return task ID
    
    async def queue_batch_processing(
        self,
        video_ids: List[UUID],
        session_id: UUID
    ) -> List[str]:
        # Create session
        # Queue batch task
        # Return task IDs
    
    async def get_job_status(
        self,
        job_id: UUID
    ) -> ProcessingJob:
        # Get job from DB
        # Check Celery status
        # Return current status
    
    async def cancel_job(self, job_id: UUID) -> bool:
        # Get job from DB
        # Revoke Celery task
        # Update status
        # Return success
    
    async def retry_failed_jobs(
        self,
        job_type: Optional[str] = None
    ) -> List[ProcessingJob]:
        # Find failed jobs
        # Check retry limits
        # Requeue jobs
        # Return requeued jobs
```

### 3.2 Processing Session Manager
`app/services/session_manager.py`:

```python
class ProcessingSessionManager:
    async def create_session(
        self,
        session_name: str,
        session_type: str,
        configuration: Dict
    ) -> ProcessingSession:
        # Create session record
        # Initialize metrics
        # Return session
    
    async def update_progress(
        self,
        session_id: UUID,
        processed: int,
        total: Optional[int] = None
    ):
        # Update counters
        # Calculate percentage
        # Update timestamp
    
    async def complete_session(
        self,
        session_id: UUID,
        metrics: Dict
    ):
        # Mark as completed
        # Store final metrics
        # Calculate duration
    
    async def get_session_report(
        self,
        session_id: UUID
    ) -> SessionReport:
        # Get session
        # Calculate stats
        # Get related jobs
        # Return report
```

### 3.3 Worker Management
`app/workers/worker_manager.py`:

```python
class WorkerManager:
    @staticmethod
    def start_workers():
        """Start Celery workers"""
        # Start discovery worker
        # Start transcript worker
        # Start batch worker
    
    @staticmethod
    def scale_workers(queue: str, count: int):
        """Scale worker count"""
        # Adjust concurrency
        # Update configuration
    
    @staticmethod
    def get_worker_stats() -> dict:
        """Get worker statistics"""
        # Active workers
        # Queue sizes
        # Processing rates
        # Return stats
```

## Task 4: Comprehensive Testing

### 4.1 Unit Tests Structure
`tests/unit/`:
```
tests/unit/
├── test_models/
│   ├── test_channel.py
│   ├── test_video.py
│   ├── test_transcript.py
│   └── test_processing_job.py
├── test_services/
│   ├── test_channel_manager.py
│   ├── test_video_discovery.py
│   ├── test_transcript_processor.py
│   └── test_job_queue.py
├── test_integrations/
│   ├── test_youtube_api.py
│   └── test_transcript_api.py
└── test_workers/
    └── test_processing_tasks.py
```

### 4.2 Integration Tests
`tests/integration/`:

```python
# test_end_to_end.py
class TestEndToEnd:
    async def test_full_channel_processing(self):
        # Add channel
        # Discover videos
        # Extract transcripts
        # Verify results
    
    async def test_batch_processing(self):
        # Create multiple videos
        # Queue batch job
        # Verify completion
        # Check results
    
    async def test_error_recovery(self):
        # Simulate failures
        # Verify retries
        # Check final state
```

### 4.3 Performance Tests
`tests/performance/`:

```python
# test_load.py
class TestPerformance:
    async def test_concurrent_processing(self):
        # Queue 100 videos
        # Measure throughput
        # Check resource usage
        # Verify no data loss
    
    async def test_database_performance(self):
        # Bulk inserts
        # Query performance
        # Index effectiveness
        # Connection pooling
    
    async def test_api_rate_limits(self):
        # Simulate API calls
        # Verify rate limiting
        # Check quota usage
        # Test backoff
```

### 4.4 Test Fixtures
`tests/fixtures/`:

```python
# fixtures.py
@pytest.fixture
async def test_channel():
    # Create test channel
    return channel

@pytest.fixture
async def test_videos():
    # Create test videos
    return videos

@pytest.fixture
async def mock_youtube_api():
    # Mock API responses
    return mock

@pytest.fixture
async def celery_worker():
    # Start test worker
    yield worker
    # Cleanup
```

## Task 5: Monitoring & Logging

### 5.1 Structured Logging
`app/utils/logging.py`:

```python
import structlog

def setup_logging():
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger()
```

### 5.2 Flower Setup
`docker-compose.yml`:
```yaml
flower:
  image: mher/flower:2.0
  command: celery flower
  environment:
    - CELERY_BROKER_URL=redis://redis:6379/0
    - FLOWER_PORT=5555
  ports:
    - "5555:5555"
  depends_on:
    - redis
```

## Task 6: Documentation

### 6.1 README.md
```markdown
# TubeSensei - Phase 1 Complete

## Quick Start
1. Clone repository
2. Copy .env.example to .env
3. Run `docker-compose up -d`
4. Run `alembic upgrade head`
5. Run `python -m app.main`

## Running Workers
```bash
# Start all workers
celery -A app.celery_app worker --loglevel=info

# Start specific queue
celery -A app.celery_app worker -Q discovery --loglevel=info
```

## Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test
pytest tests/unit/test_models/
```
```

### 6.2 API Documentation
- Auto-generated with FastAPI `/docs`
- Swagger UI available
- ReDoc at `/redoc`

## Deliverables Checklist

### Core Implementation
- [ ] Celery configuration complete
- [ ] All processing tasks implemented
- [ ] Job management system working
- [ ] Worker scaling functional
- [ ] Monitoring metrics exposed

### Testing
- [ ] Unit tests >80% coverage
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Load testing completed

### Documentation
- [ ] README with setup instructions
- [ ] API documentation
- [ ] Configuration guide
- [ ] Deployment instructions

## Success Criteria
- [ ] Process 100+ videos/hour
- [ ] 90%+ transcript extraction success
- [ ] <100ms database query times
- [ ] 10+ concurrent jobs supported
- [ ] Automatic retry on failures
- [ ] Zero data corruption
- [ ] 80%+ test coverage
- [ ] All acceptance criteria met

## Performance Metrics

### Target Benchmarks
- Video Discovery: 500 videos/minute
- Transcript Extraction: 100 videos/hour
- Database Queries: <100ms (indexed)
- Job Queue Latency: <1 second
- Worker Memory: <500MB per worker
- Redis Memory: <1GB for 10k jobs

### Monitoring Dashboard
Access monitoring at:
- Flower: http://localhost:5555
- Prometheus: http://localhost:9090
- API Docs: http://localhost:8000/docs

## Common Issues & Solutions

### Issue: Celery Tasks Not Executing
**Solution**: Check Redis connection, verify worker is running, check queue names

### Issue: Memory Leaks in Workers
**Solution**: Set max_tasks_per_child, implement proper cleanup, monitor memory

### Issue: Database Connection Pool Exhausted
**Solution**: Increase pool size, optimize queries, use connection recycling

### Issue: Test Coverage Below 80%
**Solution**: Add missing test cases, mock external dependencies, test edge cases

## Phase 1 Completion

### Final Checklist
- [ ] All 4 sub-phases completed
- [ ] Core infrastructure operational
- [ ] 500-1000 videos processed successfully
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Ready for Phase 2 (AI Integration)

### Handoff to Phase 2
1. Verify all systems operational
2. Document any deviations from plan
3. Prepare sample data for AI testing
4. Review and optimize performance
5. Create Phase 2 setup guide

## Next Steps
With Phase 1 complete, the system is ready for:
- Phase 2: AI-powered filtering and idea extraction
- Phase 3: Analytics dashboard
- Phase 4: Production deployment