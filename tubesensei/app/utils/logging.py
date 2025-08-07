"""
Structured logging configuration for TubeSensei
Provides JSON-formatted structured logging with proper processors
"""

import structlog
import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime

def setup_logging(
    log_level: str = "INFO",
    enable_json: bool = True,
    include_caller: bool = False
) -> None:
    """
    Configure structured logging with JSON formatting
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Enable JSON output format
        include_caller: Include caller information in logs
    """
    
    # Configure processors based on format preference
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if include_caller:
        processors.append(structlog.processors.CallsiteParameterAdder())
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )


def get_logger(name: str, **context: Any) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger with optional context
    
    Args:
        name: Logger name (usually module name)
        **context: Additional context to bind to logger
        
    Returns:
        Configured structured logger
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


class LoggerMixin:
    """Mixin to add structured logging to classes"""
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._logger = get_logger(cls.__module__ + "." + cls.__qualname__)
    
    @property
    def logger(self) -> structlog.stdlib.BoundLogger:
        """Get logger for this class"""
        return self._logger


class JobLogger:
    """Specialized logger for job processing"""
    
    def __init__(self, job_id: str, job_type: str):
        self.logger = get_logger(
            "tubesensei.jobs",
            job_id=job_id,
            job_type=job_type
        )
    
    def start(self, **context: Any) -> None:
        """Log job start"""
        self.logger.info("Job started", **context)
    
    def progress(self, current: int, total: int, **context: Any) -> None:
        """Log job progress"""
        percentage = (current / total) * 100 if total > 0 else 0
        self.logger.info(
            "Job progress",
            current=current,
            total=total,
            percentage=round(percentage, 2),
            **context
        )
    
    def success(self, duration_ms: Optional[int] = None, **context: Any) -> None:
        """Log job success"""
        log_data = {"status": "success", **context}
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms
        self.logger.info("Job completed", **log_data)
    
    def failure(self, error: Exception, retry_count: int = 0, **context: Any) -> None:
        """Log job failure"""
        self.logger.error(
            "Job failed",
            error=str(error),
            error_type=type(error).__name__,
            retry_count=retry_count,
            **context
        )
    
    def warning(self, message: str, **context: Any) -> None:
        """Log job warning"""
        self.logger.warning(message, **context)


class APILogger:
    """Specialized logger for API operations"""
    
    def __init__(self):
        self.logger = get_logger("tubesensei.api")
    
    def request(
        self, 
        method: str, 
        path: str, 
        user_id: Optional[str] = None,
        **context: Any
    ) -> None:
        """Log API request"""
        log_data = {
            "method": method,
            "path": path,
            "timestamp": datetime.utcnow().isoformat(),
            **context
        }
        if user_id:
            log_data["user_id"] = user_id
        self.logger.info("API request", **log_data)
    
    def response(
        self, 
        method: str, 
        path: str, 
        status_code: int,
        duration_ms: int,
        user_id: Optional[str] = None,
        **context: Any
    ) -> None:
        """Log API response"""
        log_data = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat(),
            **context
        }
        if user_id:
            log_data["user_id"] = user_id
        
        if status_code >= 500:
            self.logger.error("API response", **log_data)
        elif status_code >= 400:
            self.logger.warning("API response", **log_data)
        else:
            self.logger.info("API response", **log_data)


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = get_logger("tubesensei.performance")
    
    def query_performance(
        self, 
        query_type: str, 
        duration_ms: int,
        rows_affected: Optional[int] = None,
        **context: Any
    ) -> None:
        """Log database query performance"""
        log_data = {
            "query_type": query_type,
            "duration_ms": duration_ms,
            **context
        }
        if rows_affected is not None:
            log_data["rows_affected"] = rows_affected
        
        # Log slow queries as warnings
        if duration_ms > 1000:  # > 1 second
            self.logger.warning("Slow query detected", **log_data)
        else:
            self.logger.info("Query performance", **log_data)
    
    def cache_performance(
        self, 
        operation: str, 
        hit: bool, 
        key: str,
        duration_ms: Optional[int] = None,
        **context: Any
    ) -> None:
        """Log cache performance"""
        log_data = {
            "operation": operation,
            "cache_hit": hit,
            "cache_key": key,
            **context
        }
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms
        
        self.logger.info("Cache performance", **log_data)
    
    def worker_performance(
        self,
        worker_name: str,
        tasks_processed: int,
        processing_time_ms: int,
        memory_usage_mb: Optional[float] = None,
        **context: Any
    ) -> None:
        """Log worker performance metrics"""
        log_data = {
            "worker_name": worker_name,
            "tasks_processed": tasks_processed,
            "processing_time_ms": processing_time_ms,
            "tasks_per_second": round(tasks_processed / (processing_time_ms / 1000), 2),
            **context
        }
        if memory_usage_mb is not None:
            log_data["memory_usage_mb"] = memory_usage_mb
        
        self.logger.info("Worker performance", **log_data)


def log_function_call(func_name: str, **context: Any):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger("tubesensei.functions")
            start_time = datetime.utcnow()
            
            try:
                logger.info(f"Function {func_name} started", **context)
                result = func(*args, **kwargs)
                
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                logger.info(
                    f"Function {func_name} completed",
                    duration_ms=duration_ms,
                    **context
                )
                return result
                
            except Exception as e:
                duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                logger.error(
                    f"Function {func_name} failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    duration_ms=duration_ms,
                    **context
                )
                raise
        
        return wrapper
    return decorator


# Initialize default logger
logger = get_logger("tubesensei")