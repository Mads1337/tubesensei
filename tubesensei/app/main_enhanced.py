"""
TubeSensei Enhanced FastAPI Application with Admin Interface
"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
import uuid

from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.sessions import SessionMiddleware
import redis.asyncio as aioredis

from app.core.config import settings
from app.core.exceptions import setup_exception_handlers
from app.database import get_db as get_session, init_db, close_db

# Import existing routers
from app.api.admin import router as admin_router
# from app.api.auth import router as auth_router
# from app.api.v1 import router as api_v1_router

# Setup enhanced logging
from app.utils.logging import setup_logging
import logging

# Setup logger with fallback
try:
    logger = setup_logging()
except Exception:
    logger = None

def safe_log_info(message):
    if logger:
        logger.info(message)
    else:
        print(f"INFO: {message}")

def safe_log_warning(message):
    if logger:
        logger.warning(message)
    else:
        print(f"WARNING: {message}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    safe_log_info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    safe_log_info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize database
    await init_db()
    safe_log_info("Database initialized")
    
    # Initialize Redis connection for sessions
    app.state.redis = await aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=settings.REDIS_MAX_CONNECTIONS
    )
    safe_log_info("Redis connection established")
    
    # Setup template directories
    template_dir = Path(__file__).parent.parent.parent / settings.admin.TEMPLATE_DIR
    static_dir = Path(__file__).parent.parent.parent / settings.admin.STATIC_DIR
    
    # Ensure directories exist
    template_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    
    safe_log_info(f"Template directory: {template_dir}")
    safe_log_info(f"Static directory: {static_dir}")
    
    yield
    
    # Cleanup
    safe_log_info("Shutting down application...")
    await app.state.redis.close()
    await close_db()
    safe_log_info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="YouTube Content Analysis Platform with Admin Interface",
    lifespan=lifespan,
    debug=settings.DEBUG,
    docs_url="/api/docs" if settings.FEATURES_ENABLE_API_DOCS else None,
    redoc_url="/api/redoc" if settings.FEATURES_ENABLE_API_DOCS else None,
    openapi_url="/api/openapi.json" if settings.FEATURES_ENABLE_API_DOCS else None,
)

# Configure middleware in correct order (outermost to innermost)

# 1. Trusted Host Middleware (security)
if settings.security.ALLOWED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.security.ALLOWED_HOSTS
    )

# 2. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.ALLOWED_ORIGINS,
    allow_credentials=settings.security.ALLOW_CREDENTIALS,
    allow_methods=settings.security.ALLOWED_METHODS,
    allow_headers=settings.security.ALLOWED_HEADERS,
)

# 3. Session Middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.security.SECRET_KEY,
    session_cookie=settings.security.SESSION_COOKIE_NAME,
    max_age=settings.security.SESSION_EXPIRE_HOURS * 3600,
    same_site=settings.security.SESSION_COOKIE_SAMESITE,
    https_only=settings.security.SESSION_COOKIE_SECURE,
)

# 4. GZip Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 5. Custom Request ID Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request"""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# 6. Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses"""
    start_time = time.time()
    
    # Log request
    safe_log_info(f"Request started: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    safe_log_info(f"Request completed: {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 7. Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Only add HSTS in production
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# Mount static files
static_dir = Path(__file__).parent.parent.parent / settings.admin.STATIC_DIR
if static_dir.exists():
    app.mount(
        settings.admin.STATIC_URL,
        StaticFiles(directory=str(static_dir)),
        name="static"
    )
    safe_log_info(f"Static files mounted at {settings.admin.STATIC_URL}")

# Setup templates
template_dir = Path(__file__).parent.parent.parent / settings.admin.TEMPLATE_DIR
templates = Jinja2Templates(directory=str(template_dir))

# Setup exception handlers
setup_exception_handlers(app)

# Include routers
# Authentication routes
# if hasattr(app, 'state'):
#     try:
#         app.include_router(auth_router, prefix="/api/auth", tags=["authentication"])
#         logger.info("Auth router included")
#     except ImportError:
#         logger.warning("Auth router not yet implemented")

# Admin routes
if hasattr(app, 'state'):
    try:
        app.include_router(admin_router, tags=["admin"])
        safe_log_info("Admin router included")
    except ImportError:
        safe_log_warning("Admin router not yet implemented")

# API v1 routes
# if hasattr(app, 'state'):
#     try:
#         app.include_router(api_v1_router, prefix="/api/v1", tags=["api"])
#         logger.info("API v1 router included")
#     except ImportError:
#         logger.warning("API v1 router not yet implemented")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Root endpoint - redirect to admin or show landing page"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT
        }
    )

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

@app.get("/health/detailed")
async def detailed_health_check(
    session: AsyncSession = Depends(get_session),
    request: Request = None
):
    """Detailed health check with component status"""
    health_status = {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "components": {}
    }
    
    # Check database
    try:
        from sqlalchemy import text
        await session.execute(text("SELECT 1"))
        health_status["components"]["database"] = {
            "status": "healthy",
            "type": "postgresql"
        }
    except Exception as e:
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        if hasattr(app.state, 'redis'):
            await app.state.redis.ping()
            health_status["components"]["redis"] = {
                "status": "healthy",
                "type": "redis"
            }
    except Exception as e:
        health_status["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Celery workers
    try:
        from app.celery_app import celery_app
        stats = celery_app.control.inspect().stats()
        if stats:
            worker_count = len(stats)
            health_status["components"]["workers"] = {
                "status": "healthy",
                "count": worker_count
            }
        else:
            health_status["components"]["workers"] = {
                "status": "unhealthy",
                "count": 0
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["workers"] = {
            "status": "unknown",
            "error": str(e)
        }
    
    return health_status

# API info endpoint
@app.get("/api/info")
async def api_info():
    """Get API information"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "api_version": "v1",
        "documentation": {
            "openapi": "/api/docs",
            "redoc": "/api/redoc",
            "postman": "/api/postman"
        },
        "endpoints": {
            "auth": "/api/auth",
            "admin": settings.admin.ADMIN_PATH_PREFIX,
            "api": "/api/v1",
            "health": "/health",
            "metrics": "/metrics" if settings.FEATURES_ENABLE_METRICS else None
        }
    }

# Metrics endpoint (if enabled)
if settings.FEATURES_ENABLE_METRICS:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main_enhanced:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD and settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )