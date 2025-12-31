from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache
import secrets


class SecuritySettings(BaseSettings):
    """Security-specific settings"""
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for JWT encoding"
    )
    ALGORITHM: str = Field(
        default="HS256",
        description="JWT encoding algorithm"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=10080,  # 7 days
        description="Access token expiration time in minutes"
    )
    REFRESH_TOKEN_EXPIRE_MINUTES: int = Field(
        default=43200,  # 30 days
        description="Refresh token expiration time in minutes"
    )
    PASSWORD_MIN_LENGTH: int = Field(
        default=8,
        description="Minimum password length"
    )
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(
        default=True,
        description="Require uppercase letter in password"
    )
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(
        default=True,
        description="Require lowercase letter in password"
    )
    PASSWORD_REQUIRE_DIGIT: bool = Field(
        default=True,
        description="Require digit in password"
    )
    PASSWORD_REQUIRE_SPECIAL: bool = Field(
        default=False,
        description="Require special character in password"
    )
    BCRYPT_ROUNDS: int = Field(
        default=12,
        description="Number of bcrypt rounds for password hashing"
    )
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:8000", "http://localhost:3000", "http://127.0.0.1:8000"],
        description="Allowed CORS origins"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1", "0.0.0.0"],
        description="Allowed hosts for TrustedHost middleware"
    )
    ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    ALLOWED_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    ALLOWED_HEADERS: List[str] = Field(
        default=["*"],
        description="Allowed headers in CORS requests"
    )
    
    # Session settings
    SESSION_COOKIE_NAME: str = Field(
        default="tubesensei_session",
        description="Session cookie name"
    )
    SESSION_COOKIE_HTTPONLY: bool = Field(
        default=True,
        description="HTTPOnly flag for session cookie"
    )
    SESSION_COOKIE_SECURE: bool = Field(
        default=False,  # Set to True in production with HTTPS
        description="Secure flag for session cookie"
    )
    SESSION_COOKIE_SAMESITE: str = Field(
        default="lax",
        description="SameSite attribute for session cookie"
    )
    SESSION_EXPIRE_HOURS: int = Field(
        default=24,
        description="Session expiration time in hours"
    )
    
    # Rate limiting
    RATE_LIMIT_ENABLED: bool = Field(
        default=True,
        description="Enable rate limiting"
    )
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=60,
        description="Maximum requests per minute per IP"
    )
    LOGIN_ATTEMPTS_MAX: int = Field(
        default=5,
        description="Maximum login attempts before lockout"
    )
    LOGIN_LOCKOUT_MINUTES: int = Field(
        default=15,
        description="Login lockout duration in minutes"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "SECURITY_"
        case_sensitive = True


class AdminSettings(BaseSettings):
    """Admin interface settings"""
    ADMIN_PATH_PREFIX: str = Field(
        default="/admin",
        description="Admin interface path prefix"
    )
    ADMIN_TITLE: str = Field(
        default="TubeSensei Admin",
        description="Admin interface title"
    )
    ADMIN_DESCRIPTION: str = Field(
        default="YouTube Content Analysis Platform Administration",
        description="Admin interface description"
    )
    ADMIN_VERSION: str = Field(
        default="1.0.0",
        description="Admin interface version"
    )
    ADMIN_PAGINATION_DEFAULT: int = Field(
        default=20,
        description="Default pagination size"
    )
    ADMIN_PAGINATION_MAX: int = Field(
        default=100,
        description="Maximum pagination size"
    )
    ADMIN_ENABLE_DOCS: bool = Field(
        default=True,
        description="Enable admin API documentation"
    )
    ADMIN_ENABLE_REDOC: bool = Field(
        default=True,
        description="Enable ReDoc documentation"
    )
    
    # Template settings
    TEMPLATE_DIR: str = Field(
        default="templates",
        description="Template directory path"
    )
    STATIC_DIR: str = Field(
        default="static",
        description="Static files directory path"
    )
    STATIC_URL: str = Field(
        default="/static",
        description="Static files URL prefix"
    )
    
    # UI settings
    UI_THEME: str = Field(
        default="light",
        description="Default UI theme (light/dark)"
    )
    UI_ENABLE_ANIMATIONS: bool = Field(
        default=True,
        description="Enable UI animations"
    )
    UI_ENABLE_TOOLTIPS: bool = Field(
        default=True,
        description="Enable tooltips"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "ADMIN_"
        case_sensitive = True


class TopicDiscoverySettings(BaseSettings):
    """Topic Discovery campaign settings"""
    # Default limits
    DEFAULT_VIDEO_LIMIT: int = Field(
        default=3000,
        description="Default maximum total videos to discover per campaign"
    )
    DEFAULT_CHANNEL_LIMIT: int = Field(
        default=5,
        description="Default maximum videos from any single channel"
    )
    SEARCH_LIMIT: int = Field(
        default=50,
        description="Maximum results from initial YouTube search"
    )
    SIMILAR_DEPTH: int = Field(
        default=2,
        description="Default recursion depth for similar videos discovery"
    )

    # AI Filtering
    FILTER_THRESHOLD: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Default minimum relevance score for topic filter"
    )
    FILTER_BATCH_SIZE: int = Field(
        default=10,
        description="Number of videos to filter in a single LLM call"
    )
    FILTER_MODEL: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for topic filtering"
    )

    # Rate Limiting
    RATE_LIMIT: float = Field(
        default=0.5,
        description="Minimum seconds between YouTube API calls"
    )
    MAX_CONCURRENT_AGENTS: int = Field(
        default=3,
        description="Maximum number of agents running concurrently"
    )

    # Celery task settings
    TASK_SOFT_TIME_LIMIT: int = Field(
        default=3600,
        description="Soft time limit for campaign tasks (seconds)"
    )
    TASK_TIME_LIMIT: int = Field(
        default=7200,
        description="Hard time limit for campaign tasks (seconds)"
    )

    class Config:
        env_file = ".env"
        env_prefix = "TOPIC_"
        case_sensitive = True


class EnhancedSettings(BaseSettings):
    """Enhanced settings combining all configurations"""
    # Application settings
    APP_NAME: str = Field(
        default="TubeSensei",
        description="Application name"
    )
    APP_VERSION: str = Field(
        default="3.0.0",
        description="Application version"
    )
    ENVIRONMENT: str = Field(
        default="development",
        description="Environment (development/staging/production)"
    )
    DEBUG: bool = Field(
        default=True,
        description="Debug mode"
    )
    
    # Server settings
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host"
    )
    PORT: int = Field(
        default=8000,
        description="Server port"
    )
    WORKERS: int = Field(
        default=4,
        description="Number of worker processes"
    )
    RELOAD: bool = Field(
        default=True,
        description="Enable auto-reload in development"
    )
    
    # Database settings (from existing config)
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://bazaar_user:bazaar_pass@localhost:5433/tubesensei",
        description="PostgreSQL database connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(
        default=20,
        description="Database connection pool size"
    )
    DATABASE_POOL_MAX_OVERFLOW: int = Field(
        default=30,
        description="Maximum overflow for connection pool"
    )
    DATABASE_POOL_TIMEOUT: int = Field(
        default=30,
        description="Database connection timeout"
    )
    DATABASE_ECHO: bool = Field(
        default=False,
        description="Echo SQL queries (debug)"
    )
    
    # Redis settings
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    REDIS_MAX_CONNECTIONS: int = Field(
        default=50,
        description="Maximum Redis connections"
    )
    REDIS_DECODE_RESPONSES: bool = Field(
        default=True,
        description="Decode Redis responses"
    )
    REDIS_SOCKET_TIMEOUT: int = Field(
        default=5,
        description="Redis socket timeout in seconds"
    )
    REDIS_CONNECTION_TIMEOUT: int = Field(
        default=5,
        description="Redis connection timeout in seconds"
    )
    
    # Logging settings
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format (json/text)"
    )
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Log file path"
    )
    LOG_ROTATION: str = Field(
        default="1 day",
        description="Log rotation interval"
    )
    LOG_RETENTION: str = Field(
        default="30 days",
        description="Log retention period"
    )
    
    # Security settings
    security: SecuritySettings = Field(
        default_factory=SecuritySettings,
        description="Security configuration"
    )
    
    # Admin settings
    admin: AdminSettings = Field(
        default_factory=AdminSettings,
        description="Admin interface configuration"
    )

    # Topic Discovery settings
    topic_discovery: TopicDiscoverySettings = Field(
        default_factory=TopicDiscoverySettings,
        description="Topic discovery campaign configuration"
    )

    # YouTube API settings (from existing config)
    YOUTUBE_API_KEY: str = Field(
        default="",
        description="YouTube Data API v3 key"
    )
    YOUTUBE_QUOTA_PER_DAY: int = Field(
        default=10000,
        description="YouTube API daily quota"
    )
    
    # OpenAI/LLM settings (from existing config)
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    # Feature flags
    FEATURES_ENABLE_REGISTRATION: bool = Field(
        default=True,
        description="Enable user registration"
    )
    FEATURES_ENABLE_API_DOCS: bool = Field(
        default=True,
        description="Enable API documentation"
    )
    FEATURES_ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    FEATURES_ENABLE_HEALTH_CHECKS: bool = Field(
        default=True,
        description="Enable health check endpoints"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL for Alembic"""
        return self.DATABASE_URL.replace("+asyncpg", "")


@lru_cache()
def get_settings() -> EnhancedSettings:
    """Get cached settings instance"""
    return EnhancedSettings()


# Export settings instance
settings = get_settings()