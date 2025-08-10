from fastapi import Request, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException
import traceback
import logging
from typing import Union, Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


class TubeSenseiException(Exception):
    """Base exception for TubeSensei application"""
    def __init__(
        self, 
        message: str, 
        status_code: int = 400, 
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.headers = headers
        super().__init__(self.message)


class NotFoundException(TubeSenseiException):
    """Resource not found exception"""
    def __init__(self, resource: str, resource_id: Optional[str] = None):
        message = f"{resource} not found"
        if resource_id:
            message += f": {resource_id}"
        super().__init__(message, status_code=404)


class AuthenticationException(TubeSenseiException):
    """Authentication failed exception"""
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(
            message, 
            status_code=401,
            details=details,
            headers={"WWW-Authenticate": "Bearer"}
        )


class PermissionException(TubeSenseiException):
    """Permission denied exception"""
    def __init__(self, message: str = "Permission denied", required_permission: Optional[str] = None):
        details = {}
        if required_permission:
            details["required_permission"] = required_permission
        super().__init__(message, status_code=403, details=details)


class ValidationException(TubeSenseiException):
    """Validation error exception"""
    def __init__(self, errors: Dict[str, Any], message: str = "Validation failed"):
        super().__init__(message, status_code=422, details={"errors": errors})


class RateLimitException(TubeSenseiException):
    """Rate limit exceeded exception"""
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        retry_after: Optional[int] = None
    ):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        super().__init__(
            message, 
            status_code=429, 
            headers=headers,
            details={"retry_after": retry_after} if retry_after else None
        )


class QuotaExceededException(TubeSenseiException):
    """Quota exceeded exception"""
    def __init__(self, resource: str, message: Optional[str] = None):
        msg = message or f"Quota exceeded for {resource}"
        super().__init__(msg, status_code=402, details={"resource": resource})


class ServiceUnavailableException(TubeSenseiException):
    """Service unavailable exception"""
    def __init__(self, service: str, message: Optional[str] = None):
        msg = message or f"Service {service} is temporarily unavailable"
        super().__init__(msg, status_code=503, details={"service": service})


class BadRequestException(TubeSenseiException):
    """Bad request exception"""
    def __init__(self, message: str = "Bad request", details: Optional[Dict] = None):
        super().__init__(message, status_code=400, details=details)


class ConflictException(TubeSenseiException):
    """Resource conflict exception"""
    def __init__(self, message: str = "Resource conflict", details: Optional[Dict] = None):
        super().__init__(message, status_code=409, details=details)


class UnprocessableEntityException(TubeSenseiException):
    """Unprocessable entity exception"""
    def __init__(self, message: str = "Unprocessable entity", details: Optional[Dict] = None):
        super().__init__(message, status_code=422, details=details)


async def tubesensei_exception_handler(
    request: Request, 
    exc: TubeSenseiException
) -> Union[JSONResponse, HTMLResponse]:
    """Handle TubeSensei exceptions"""
    logger.error(
        f"TubeSensei exception: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "details": exc.details,
            "path": str(request.url),
            "method": request.method,
            "client": request.client.host if request.client else None
        }
    )
    
    # Check if request expects JSON or HTML
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header or request.url.path.startswith("/api"):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details,
                "path": str(request.url)
            },
            headers=exc.headers
        )
    else:
        # For HTML responses, we'll need templates (will be set up later)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.message,
                "details": exc.details
            },
            headers=exc.headers
        )


async def validation_exception_handler(
    request: Request, 
    exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors"""
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"][1:]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.error(
        "Validation error",
        extra={
            "errors": errors,
            "path": str(request.url),
            "method": request.method
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation failed",
            "details": {"errors": errors},
            "path": str(request.url)
        }
    )


async def http_exception_handler(
    request: Request, 
    exc: StarletteHTTPException
) -> Union[JSONResponse, HTMLResponse]:
    """Handle HTTP exceptions"""
    logger.error(
        f"HTTP exception: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": str(request.url),
            "method": request.method
        }
    )
    
    # Check if request expects JSON or HTML
    accept_header = request.headers.get("accept", "")
    if "application/json" in accept_header or request.url.path.startswith("/api"):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "path": str(request.url)
            }
        )
    else:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail}
        )


async def general_exception_handler(
    request: Request, 
    exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions"""
    error_id = str(id(exc))
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={
            "error_id": error_id,
            "traceback": traceback.format_exc(),
            "path": str(request.url),
            "method": request.method,
            "client": request.client.host if request.client else None
        }
    )
    
    # In production, don't expose internal error details
    from app.core.config import settings
    if settings.is_production:
        message = "Internal server error"
        details = {"error_id": error_id}
    else:
        message = str(exc)
        details = {
            "error_id": error_id,
            "type": type(exc).__name__,
            "traceback": traceback.format_exc().split("\n")
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": message,
            "details": details,
            "path": str(request.url)
        }
    )


def setup_exception_handlers(app):
    """Setup all exception handlers for the application"""
    from fastapi import FastAPI
    
    # Custom exception handlers
    app.add_exception_handler(TubeSenseiException, tubesensei_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # Only add general exception handler in non-debug mode
    from app.core.config import settings
    if not settings.DEBUG:
        app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Exception handlers configured")