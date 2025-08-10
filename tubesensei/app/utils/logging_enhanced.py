"""
Enhanced logging configuration for TubeSensei
"""
import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import traceback
from pythonjsonlogger import jsonlogger

from app.core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add level name
        log_record['level'] = record.levelname
        
        # Add module and function info
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add app info
        log_record['app'] = settings.APP_NAME
        log_record['environment'] = settings.ENVIRONMENT
        log_record['version'] = settings.APP_VERSION
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
            log_record['exception_type'] = record.exc_info[0].__name__ if record.exc_info[0] else None
        
        # Add extra fields if present
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_record['user_id'] = record.user_id
        if hasattr(record, 'client_ip'):
            log_record['client_ip'] = record.client_ip


class ContextFilter(logging.Filter):
    """Add context information to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add default values if not present
        if not hasattr(record, 'request_id'):
            record.request_id = None
        if not hasattr(record, 'user_id'):
            record.user_id = None
        if not hasattr(record, 'client_ip'):
            record.client_ip = None
        
        return True


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging() -> logging.Logger:
    """Setup comprehensive logging configuration"""
    
    # Create logger
    logger = logging.getLogger("tubesensei")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Add context filter
    context_filter = ContextFilter()
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    if settings.LOG_FORMAT == "json":
        # JSON format for production
        json_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(json_formatter)
    else:
        # Colored text format for development
        if settings.DEBUG:
            text_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            text_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        console_handler.setFormatter(text_formatter)
    
    console_handler.addFilter(context_filter)
    logger.addHandler(console_handler)
    
    # File Handler (if LOG_FILE is set)
    if settings.LOG_FILE:
        log_file = Path(settings.LOG_FILE)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # Always use JSON format for file logs
        json_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(json_formatter)
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)
    
    # Error file handler for ERROR and above
    error_log_file = Path("logs/error.log")
    error_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(json_formatter if 'json_formatter' in locals() else CustomJsonFormatter())
    error_handler.addFilter(context_filter)
    logger.addHandler(error_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Also configure other loggers
    configure_library_loggers()
    
    logger.info(
        "Logging configured",
        extra={
            "log_level": settings.LOG_LEVEL,
            "log_format": settings.LOG_FORMAT,
            "log_file": settings.LOG_FILE,
            "environment": settings.ENVIRONMENT
        }
    )
    
    return logger


def configure_library_loggers():
    """Configure logging for third-party libraries"""
    
    # Reduce noise from libraries
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("aioredis").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Celery logging
    logging.getLogger("celery").setLevel(logging.INFO)
    logging.getLogger("celery.task").setLevel(logging.INFO)
    logging.getLogger("celery.worker").setLevel(logging.INFO)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter for adding context to log messages"""
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        # Add context to extra fields
        extra = kwargs.get('extra', {})
        extra.update(self.extra)
        kwargs['extra'] = extra
        return msg, kwargs


def get_logger(name: str, **context) -> LoggerAdapter:
    """Get a logger with optional context"""
    logger = logging.getLogger(f"tubesensei.{name}")
    return LoggerAdapter(logger, context)


def log_execution_time(func):
    """Decorator to log function execution time"""
    import functools
    import time
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                f"Function {func.__name__} executed successfully",
                extra={
                    "function": func.__name__,
                    "execution_time": f"{execution_time:.3f}s"
                }
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    "function": func.__name__,
                    "execution_time": f"{execution_time:.3f}s",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                f"Function {func.__name__} executed successfully",
                extra={
                    "function": func.__name__,
                    "execution_time": f"{execution_time:.3f}s"
                }
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed",
                extra={
                    "function": func.__name__,
                    "execution_time": f"{execution_time:.3f}s",
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Create default logger instance
default_logger = setup_logging()