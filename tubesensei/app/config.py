from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List


class Settings(BaseSettings):
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://tubesensei:tubesensei_dev@localhost:5433/tubesensei",
        description="PostgreSQL database connection URL with asyncpg driver"
    )
    DATABASE_POOL_SIZE: int = Field(
        default=20,
        description="Database connection pool size"
    )
    DATABASE_POOL_MAX_OVERFLOW: int = Field(
        default=10,
        description="Maximum overflow for database connection pool"
    )
    DATABASE_POOL_TIMEOUT: int = Field(
        default=30,
        description="Database connection pool timeout in seconds"
    )
    
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    DEBUG: bool = Field(
        default=False,
        description="Debug mode flag"
    )
    
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    BATCH_SIZE: int = Field(
        default=100,
        description="Default batch size for bulk operations"
    )
    
    MAX_RETRIES: int = Field(
        default=3,
        description="Maximum number of retries for failed operations"
    )
    
    APP_NAME: str = Field(
        default="TubeSensei",
        description="Application name"
    )
    
    API_VERSION: str = Field(
        default="1.0.0",
        description="API version"
    )
    
    # YouTube API Configuration
    YOUTUBE_API_KEY: str = Field(
        default="",
        description="YouTube Data API v3 key"
    )
    YOUTUBE_QUOTA_PER_DAY: int = Field(
        default=10000,
        description="YouTube API daily quota limit"
    )
    YOUTUBE_MAX_RESULTS_PER_PAGE: int = Field(
        default=50,
        description="Maximum results per YouTube API page"
    )
    YOUTUBE_REQUEST_TIMEOUT: int = Field(
        default=30,
        description="YouTube API request timeout in seconds"
    )
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=60,
        description="Rate limit for YouTube API requests per minute"
    )
    YOUTUBE_CACHE_TTL: int = Field(
        default=86400,  # 24 hours
        description="Cache TTL for YouTube metadata in seconds"
    )
    
    # Transcript Processing Configuration
    TRANSCRIPT_TIMEOUT_SECONDS: int = Field(
        default=300,
        description="Timeout for transcript extraction in seconds"
    )
    TRANSCRIPT_MAX_RETRIES: int = Field(
        default=3,
        description="Maximum retries for failed transcript extractions"
    )
    TRANSCRIPT_PREFERRED_LANGUAGES: List[str] = Field(
        default=["en", "en-US", "en-GB"],
        description="Preferred languages for transcript extraction"
    )
    TRANSCRIPT_CACHE_TTL_HOURS: int = Field(
        default=168,  # 7 days
        description="Cache TTL for transcripts in hours"
    )
    MIN_TRANSCRIPT_WORD_COUNT: int = Field(
        default=100,
        description="Minimum word count for valid transcripts"
    )
    MAX_TRANSCRIPT_LENGTH: int = Field(
        default=500000,
        description="Maximum transcript length in characters"
    )
    TRANSCRIPT_BATCH_SIZE: int = Field(
        default=10,
        description="Batch size for concurrent transcript processing"
    )
    TRANSCRIPT_QUALITY_THRESHOLD: float = Field(
        default=0.6,
        description="Minimum quality score threshold for transcripts"
    )

    # Celery Configuration
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Celery broker URL (Redis)"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/0",
        description="Celery result backend URL"
    )
    CELERY_TASK_SERIALIZER: str = Field(
        default="json",
        description="Celery task serializer"
    )
    CELERY_RESULT_SERIALIZER: str = Field(
        default="json",
        description="Celery result serializer"
    )
    CELERY_ACCEPT_CONTENT: List[str] = Field(
        default=["json"],
        description="Accepted content types for Celery"
    )
    CELERY_TIMEZONE: str = Field(
        default="UTC",
        description="Celery timezone"
    )
    CELERY_ENABLE_UTC: bool = Field(
        default=True,
        description="Enable UTC for Celery"
    )
    CELERY_TASK_TRACK_STARTED: bool = Field(
        default=True,
        description="Track task start in Celery"
    )
    CELERY_TASK_TIME_LIMIT: int = Field(
        default=1800,  # 30 minutes
        description="Celery task time limit in seconds"
    )
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(
        default=1500,  # 25 minutes
        description="Celery task soft time limit in seconds"
    )
    CELERY_TASK_ACKS_LATE: bool = Field(
        default=True,
        description="Enable late acknowledgment for Celery tasks"
    )
    CELERY_WORKER_PREFETCH_MULTIPLIER: int = Field(
        default=1,
        description="Worker prefetch multiplier for Celery"
    )
    CELERY_WORKER_MAX_TASKS_PER_CHILD: int = Field(
        default=1000,
        description="Maximum tasks per worker child process"
    )

    # Job Queue Configuration
    MAX_CONCURRENT_JOBS: int = Field(
        default=10,
        description="Maximum number of concurrent processing jobs"
    )
    MAX_VIDEOS_PER_BATCH: int = Field(
        default=50,
        description="Maximum number of videos per batch processing job"
    )
    WORKER_CONCURRENCY: int = Field(
        default=4,
        description="Number of worker processes to run concurrently"
    )
    JOB_RETRY_DELAY: int = Field(
        default=60,
        description="Delay between job retries in seconds"
    )
    JOB_MAX_RETRIES: int = Field(
        default=3,
        description="Maximum number of job retries"
    )

    # Monitoring Configuration
    METRICS_ENABLED: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    METRICS_PORT: int = Field(
        default=8001,
        description="Port for metrics endpoint"
    )
    FLOWER_BASIC_AUTH: str = Field(
        default="admin:admin",
        description="Basic authentication for Flower monitoring"
    )
    PROMETHEUS_METRICS_PATH: str = Field(
        default="/metrics",
        description="Path for Prometheus metrics endpoint"
    )

    # Performance Configuration
    DATABASE_QUERY_TIMEOUT: int = Field(
        default=30,
        description="Database query timeout in seconds"
    )
    SLOW_QUERY_THRESHOLD_MS: int = Field(
        default=1000,
        description="Threshold for slow query logging in milliseconds"
    )
    CACHE_DEFAULT_TTL: int = Field(
        default=3600,  # 1 hour
        description="Default cache TTL in seconds"
    )

    # Health Check Configuration
    HEALTH_CHECK_TIMEOUT: int = Field(
        default=30,
        description="Health check timeout in seconds"
    )
    HEALTH_CHECK_INTERVAL: int = Field(
        default=60,
        description="Health check interval in seconds"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()