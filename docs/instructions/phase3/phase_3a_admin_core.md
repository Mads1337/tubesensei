# TubeSensei Phase 3A: Admin Interface Core
## Week 8 - Days 1-2: Foundation and Authentication

### Version: 1.0
### Duration: 2 Days
### Dependencies: Phase 1 & 2 Complete

---

## Table of Contents
1. [Phase Overview](#phase-overview)
2. [Day 1: FastAPI Foundation](#day-1-fastapi-foundation)
3. [Day 2: Authentication & Authorization](#day-2-authentication--authorization)
4. [Implementation Checklist](#implementation-checklist)
5. [Testing Requirements](#testing-requirements)

---

## Phase Overview

### Objectives
Establish the core FastAPI application structure with authentication, routing, and base templates for the admin interface.

### Deliverables
- FastAPI application with proper structure
- Authentication system with JWT tokens
- Base template system with Jinja2
- Core middleware and error handling
- Database session management
- Basic routing structure

### Critical Path Items
1. FastAPI app initialization
2. Database connection setup
3. Authentication middleware
4. Base template structure
5. Static file serving

---

## Day 1: FastAPI Foundation

### 1.1 Project Structure Setup

```
tubesensei/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Settings management
│   │   ├── database.py         # Database connection
│   │   ├── dependencies.py     # Shared dependencies
│   │   └── exceptions.py       # Custom exceptions
│   ├── api/
│   │   ├── __init__.py
│   │   ├── admin/              # Admin endpoints
│   │   └── v1/                 # API v1 endpoints
│   ├── models/                 # SQLAlchemy models
│   ├── schemas/                # Pydantic schemas
│   ├── services/               # Business logic
│   └── utils/                  # Utility functions
├── static/
│   ├── css/
│   ├── js/
│   └── img/
├── templates/
│   ├── base.html
│   └── admin/
├── tests/
├── alembic/                    # Database migrations
├── .env
├── requirements.txt
└── docker-compose.yml
```

### 1.2 Core Application Setup

#### Main Application File
```python
# app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.exceptions import setup_exception_handlers
from app.api.admin import router as admin_router
from app.core.logging import setup_logging

# Setup logging
setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    await init_db()
    yield
    # Shutdown
    await close_db()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="YouTube Content Analysis Platform",
    lifespan=lifespan
)

# Configure middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
app.include_router(admin_router, prefix="/admin", tags=["admin"])

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request):
    """Root endpoint - redirect to admin"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "TubeSensei"}
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }
```

### 1.3 Configuration Management

#### Settings Configuration
```python
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "TubeSensei"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days
    
    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 30
    DATABASE_POOL_TIMEOUT: int = 30
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_MAX_CONNECTIONS: int = 50
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:8000"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    
    # YouTube API
    YOUTUBE_API_KEY: str
    YOUTUBE_QUOTA_PER_DAY: int = 10000
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

settings = get_settings()
```

### 1.4 Database Connection Setup

#### Async Database Manager
```python
# app/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from app.core.config import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_timeout=settings.DATABASE_POOL_TIMEOUT,
    echo=settings.DEBUG,
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

async def init_db():
    """Initialize database connection"""
    async with engine.begin() as conn:
        # Create tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)

async def close_db():
    """Close database connection"""
    await engine.dispose()

@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Database dependency for FastAPI"""
    async with get_session() as session:
        yield session
```

### 1.5 Base Templates Setup

#### Base HTML Template
```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}TubeSensei Admin{% endblock %}</title>
    
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    
    <!-- HTMX -->
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    
    <!-- Alpine.js for reactive components -->
    <script defer src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js"></script>
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <!-- Custom CSS -->
    <link rel="stylesheet" href="{{ url_for('static', path='/css/style.css') }}">
    
    {% block head %}{% endblock %}
</head>
<body class="bg-gray-50">
    <!-- Navigation -->
    <nav class="bg-white shadow-sm border-b">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16">
                <div class="flex">
                    <div class="flex-shrink-0 flex items-center">
                        <h1 class="text-xl font-bold text-gray-900">TubeSensei</h1>
                    </div>
                    <div class="hidden sm:ml-6 sm:flex sm:space-x-8">
                        <a href="/admin" class="nav-link">Dashboard</a>
                        <a href="/admin/channels" class="nav-link">Channels</a>
                        <a href="/admin/videos" class="nav-link">Videos</a>
                        <a href="/admin/ideas" class="nav-link">Ideas</a>
                        <a href="/admin/jobs" class="nav-link">Jobs</a>
                    </div>
                </div>
                <div class="flex items-center">
                    <div class="ml-3 relative">
                        <button id="user-menu" class="flex text-sm rounded-full">
                            <span class="sr-only">Open user menu</span>
                            <div class="h-8 w-8 rounded-full bg-gray-300"></div>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {% block content %}{% endblock %}
    </main>

    <!-- Modal Container -->
    <div id="modal-container"></div>

    <!-- Toast Notifications -->
    <div id="toast-container" class="fixed bottom-0 right-0 p-4 space-y-2"></div>

    <!-- Custom Scripts -->
    <script src="{{ url_for('static', path='/js/main.js') }}"></script>
    {% block scripts %}{% endblock %}
</body>
</html>
```

### 1.6 Error Handling Setup

#### Exception Handlers
```python
# app/core/exceptions.py
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
import logging

logger = logging.getLogger(__name__)

class TubeSenseiException(Exception):
    """Base exception for TubeSensei"""
    def __init__(self, message: str, status_code: int = 400, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

class NotFoundException(TubeSenseiException):
    """Resource not found exception"""
    def __init__(self, resource: str, resource_id: str = None):
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(message, status_code=404)

class AuthenticationException(TubeSenseiException):
    """Authentication failed exception"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)

class PermissionException(TubeSenseiException):
    """Permission denied exception"""
    def __init__(self, message: str = "Permission denied"):
        super().__init__(message, status_code=403)

class ValidationException(TubeSenseiException):
    """Validation error exception"""
    def __init__(self, errors: dict):
        super().__init__("Validation failed", status_code=422, details=errors)

async def tubesensei_exception_handler(request: Request, exc: TubeSenseiException):
    """Handle TubeSensei exceptions"""
    logger.error(f"TubeSensei exception: {exc.message}", extra={
        "status_code": exc.status_code,
        "details": exc.details,
        "path": str(request.url)
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "path": str(request.url)
        }
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error: {exc.errors()}", extra={
        "path": str(request.url)
    })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "details": exc.errors(),
            "path": str(request.url)
        }
    )

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP exception: {exc.detail}", extra={
        "status_code": exc.status_code,
        "path": str(request.url)
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "path": str(request.url)
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", extra={
        "traceback": traceback.format_exc(),
        "path": str(request.url)
    })
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "path": str(request.url)
        }
    )

def setup_exception_handlers(app):
    """Setup all exception handlers"""
    app.add_exception_handler(TubeSenseiException, tubesensei_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    if not app.debug:
        app.add_exception_handler(Exception, general_exception_handler)
```

---

## Day 2: Authentication & Authorization

### 2.1 JWT Authentication System

#### Authentication Handler
```python
# app/core/auth.py
from datetime import datetime, timedelta
from typing import Optional, Tuple
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.user import User
from app.core.exceptions import AuthenticationException

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

class AuthHandler:
    """Handle authentication operations"""
    
    def __init__(self):
        self.secret = settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        self.access_token_expire = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    def hash_password(self, password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def encode_token(self, user_id: str, email: str, role: str = "admin") -> str:
        """Generate JWT token"""
        payload = {
            "sub": user_id,
            "email": email,
            "role": role,
            "exp": datetime.utcnow() + self.access_token_expire,
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> dict:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            raise AuthenticationException(f"Invalid token: {str(e)}")
    
    async def get_current_user(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: AsyncSession = Depends(get_db)
    ) -> User:
        """Get current authenticated user"""
        token = credentials.credentials
        
        try:
            payload = self.decode_token(token)
            user_id = payload.get("sub")
            
            if not user_id:
                raise AuthenticationException("Invalid token payload")
            
            user = await db.get(User, user_id)
            if not user:
                raise AuthenticationException("User not found")
            
            if not user.is_active:
                raise AuthenticationException("User account is disabled")
            
            return user
            
        except JWTError:
            raise AuthenticationException("Could not validate credentials")

# Create singleton instance
auth_handler = AuthHandler()

# Dependency for protected routes
async def require_auth(user: User = Depends(auth_handler.get_current_user)) -> User:
    """Require authenticated user"""
    return user

async def require_admin(user: User = Depends(auth_handler.get_current_user)) -> User:
    """Require admin user"""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return user
```

### 2.2 User Model and Schema

#### User Model
```python
# app/models/user.py
from sqlalchemy import Column, String, Boolean, DateTime, Enum
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import enum

from app.core.database import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)
    
    def __repr__(self):
        return f"<User {self.email}>"
```

#### User Schemas
```python
# app/schemas/user.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from uuid import UUID

class UserBase(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    role: str = "user"

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(UserBase):
    id: UUID
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: Optional[datetime]
    
    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse
```

### 2.3 Authentication Endpoints

#### Auth Router
```python
# app/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from app.core.database import get_db
from app.core.auth import auth_handler
from app.schemas.user import UserCreate, UserLogin, UserResponse, TokenResponse
from app.models.user import User
from app.core.exceptions import AuthenticationException, ValidationException

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user"""
    # Check if user exists
    existing_user = await db.execute(
        select(User).where(
            (User.email == user_data.email) | 
            (User.username == user_data.username)
        )
    )
    if existing_user.scalar_one_or_none():
        raise ValidationException({
            "email": "User with this email or username already exists"
        })
    
    # Create new user
    hashed_password = auth_handler.hash_password(user_data.password)
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        role=user_data.role
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user

@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Login and get access token"""
    # Get user by email
    result = await db.execute(
        select(User).where(User.email == credentials.email)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise AuthenticationException("Invalid email or password")
    
    # Verify password
    if not auth_handler.verify_password(credentials.password, user.hashed_password):
        raise AuthenticationException("Invalid email or password")
    
    if not user.is_active:
        raise AuthenticationException("User account is disabled")
    
    # Update last login
    user.last_login_at = datetime.utcnow()
    await db.commit()
    
    # Generate token
    token = auth_handler.encode_token(
        str(user.id),
        user.email,
        user.role.value
    )
    
    return TokenResponse(
        access_token=token,
        user=UserResponse.from_orm(user)
    )

@router.post("/logout")
async def logout(response: Response):
    """Logout (client-side token removal)"""
    # In a JWT system, logout is typically handled client-side
    # We can add token blacklisting here if needed
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}

@router.get("/me", response_model=UserResponse)
async def get_current_user(
    user: User = Depends(auth_handler.get_current_user)
):
    """Get current authenticated user"""
    return user

@router.post("/refresh")
async def refresh_token(
    user: User = Depends(auth_handler.get_current_user)
):
    """Refresh access token"""
    token = auth_handler.encode_token(
        str(user.id),
        user.email,
        user.role.value
    )
    return TokenResponse(
        access_token=token,
        user=UserResponse.from_orm(user)
    )
```

### 2.4 Session Management

#### Session Store
```python
# app/core/session.py
import redis.asyncio as aioredis
import json
from typing import Optional, Any
from datetime import timedelta

from app.core.config import settings

class SessionStore:
    """Redis-based session storage"""
    
    def __init__(self):
        self.redis = None
        self.prefix = "session:"
        self.ttl = timedelta(days=7)
    
    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()
    
    async def set(self, session_id: str, data: dict, ttl: Optional[int] = None):
        """Store session data"""
        key = f"{self.prefix}{session_id}"
        ttl = ttl or int(self.ttl.total_seconds())
        
        await self.redis.setex(
            key,
            ttl,
            json.dumps(data)
        )
    
    async def get(self, session_id: str) -> Optional[dict]:
        """Retrieve session data"""
        key = f"{self.prefix}{session_id}"
        data = await self.redis.get(key)
        
        if data:
            return json.loads(data)
        return None
    
    async def delete(self, session_id: str):
        """Delete session"""
        key = f"{self.prefix}{session_id}"
        await self.redis.delete(key)
    
    async def exists(self, session_id: str) -> bool:
        """Check if session exists"""
        key = f"{self.prefix}{session_id}"
        return bool(await self.redis.exists(key))
    
    async def extend(self, session_id: str, ttl: Optional[int] = None):
        """Extend session TTL"""
        key = f"{self.prefix}{session_id}"
        ttl = ttl or int(self.ttl.total_seconds())
        await self.redis.expire(key, ttl)

# Global session store instance
session_store = SessionStore()
```

### 2.5 Role-Based Access Control

#### Permission System
```python
# app/core/permissions.py
from enum import Enum
from typing import List, Optional
from fastapi import Depends, HTTPException, status

from app.models.user import User, UserRole
from app.core.auth import auth_handler

class Permission(str, Enum):
    """System permissions"""
    # Channel permissions
    CHANNEL_VIEW = "channel:view"
    CHANNEL_CREATE = "channel:create"
    CHANNEL_UPDATE = "channel:update"
    CHANNEL_DELETE = "channel:delete"
    
    # Video permissions
    VIDEO_VIEW = "video:view"
    VIDEO_PROCESS = "video:process"
    VIDEO_UPDATE = "video:update"
    
    # Idea permissions
    IDEA_VIEW = "idea:view"
    IDEA_UPDATE = "idea:update"
    IDEA_EXPORT = "idea:export"
    
    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_SETTINGS = "admin:settings"
    ADMIN_JOBS = "admin:jobs"

# Role permission mappings
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [p.value for p in Permission],  # All permissions
    UserRole.USER: [
        Permission.CHANNEL_VIEW,
        Permission.CHANNEL_CREATE,
        Permission.CHANNEL_UPDATE,
        Permission.VIDEO_VIEW,
        Permission.VIDEO_PROCESS,
        Permission.IDEA_VIEW,
        Permission.IDEA_UPDATE,
        Permission.IDEA_EXPORT,
    ],
    UserRole.VIEWER: [
        Permission.CHANNEL_VIEW,
        Permission.VIDEO_VIEW,
        Permission.IDEA_VIEW,
    ]
}

class PermissionChecker:
    """Check user permissions"""
    
    def __init__(self, required_permissions: List[Permission]):
        self.required_permissions = required_permissions
    
    async def __call__(
        self,
        user: User = Depends(auth_handler.get_current_user)
    ) -> User:
        """Check if user has required permissions"""
        user_permissions = ROLE_PERMISSIONS.get(user.role, [])
        
        for permission in self.required_permissions:
            if permission.value not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission.value}"
                )
        
        return user

# Convenience functions
def require_permission(permission: Permission):
    """Require single permission"""
    return PermissionChecker([permission])

def require_any_permission(*permissions: Permission):
    """Require any of the given permissions"""
    async def checker(user: User = Depends(auth_handler.get_current_user)) -> User:
        user_permissions = ROLE_PERMISSIONS.get(user.role, [])
        
        if not any(p.value in user_permissions for p in permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        return user
    
    return checker
```

---

## Implementation Checklist

### Day 1 Tasks
- [ ] Create project directory structure
- [ ] Set up virtual environment and install dependencies
- [ ] Create FastAPI main application
- [ ] Configure settings and environment variables
- [ ] Set up database connection with async SQLAlchemy
- [ ] Create base templates with Tailwind CSS and HTMX
- [ ] Configure static file serving
- [ ] Implement error handling and logging
- [ ] Set up development server with hot reload
- [ ] Test basic application startup and health endpoint

### Day 2 Tasks
- [ ] Implement User model and migrations
- [ ] Create authentication handler with JWT
- [ ] Implement registration and login endpoints
- [ ] Set up session management with Redis
- [ ] Create role-based permission system
- [ ] Implement protected route decorators
- [ ] Add authentication middleware
- [ ] Create user management endpoints
- [ ] Test authentication flow
- [ ] Document authentication API

---

## Testing Requirements

### Unit Tests

#### Test Authentication
```python
# tests/test_auth.py
import pytest
from httpx import AsyncClient
from app.main import app
from app.core.auth import auth_handler

@pytest.mark.asyncio
async def test_register_user():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/auth/register", json={
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpass123"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert "id" in data

@pytest.mark.asyncio
async def test_login():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # First register
        await client.post("/auth/register", json={
            "email": "login@example.com",
            "username": "loginuser",
            "password": "loginpass123"
        })
        
        # Then login
        response = await client.post("/auth/login", json={
            "email": "login@example.com",
            "password": "loginpass123"
        })
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_protected_route():
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Try without token
        response = await client.get("/auth/me")
        assert response.status_code == 403
        
        # Login and get token
        login_response = await client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "testpass123"
        })
        token = login_response.json()["access_token"]
        
        # Try with token
        response = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
```

### Integration Tests

#### Test Database Connection
```python
# tests/test_database.py
import pytest
from app.core.database import get_session, engine
from app.models.user import User

@pytest.mark.asyncio
async def test_database_connection():
    async with get_session() as session:
        # Test connection
        result = await session.execute("SELECT 1")
        assert result.scalar() == 1

@pytest.mark.asyncio
async def test_user_crud():
    async with get_session() as session:
        # Create user
        user = User(
            email="db_test@example.com",
            username="dbtest",
            hashed_password="hashed"
        )
        session.add(user)
        await session.commit()
        
        # Read user
        saved_user = await session.get(User, user.id)
        assert saved_user.email == "db_test@example.com"
        
        # Update user
        saved_user.is_verified = True
        await session.commit()
        
        # Delete user
        await session.delete(saved_user)
        await session.commit()
```

---

## Success Criteria

### Day 1 Completion
- FastAPI application runs successfully
- Database connection established
- Static files served correctly
- Base templates render properly
- Error handling works as expected
- Health check endpoint responds

### Day 2 Completion
- User registration and login functional
- JWT tokens generated and validated
- Protected routes require authentication
- Role-based permissions enforced
- Session management operational
- All auth endpoints tested

### Quality Metrics
- Code coverage > 90% for auth module
- All endpoints documented
- No security vulnerabilities
- Response times < 100ms for auth operations
- Proper error messages returned