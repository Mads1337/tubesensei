from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


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

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()