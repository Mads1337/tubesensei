"""
Retry Strategy module for TubeSensei - Phase 2A Core Module

This module provides intelligent retry mechanisms for LLM operations with:
- Configurable retry policies with exponential backoff and jitter
- Rate limit detection and handling
- Circuit breaker pattern for failing providers
- Comprehensive retry context tracking
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Type, Union, Set
from datetime import datetime, timedelta, timezone

from app.config import settings


logger = logging.getLogger(__name__)


class RetryableError(Enum):
    """Types of retryable errors."""
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    MODEL_OVERLOAD = "model_overload"
    TEMPORARY_FAILURE = "temporary_failure"


class NonRetryableError(Enum):
    """Types of non-retryable errors."""
    INVALID_API_KEY = "invalid_api_key"
    INVALID_REQUEST = "invalid_request"
    INSUFFICIENT_QUOTA = "insufficient_quota"
    MODEL_NOT_FOUND = "model_not_found"
    CONTENT_FILTER = "content_filter"


@dataclass
class RetryContext:
    """Context information for retry attempts."""
    attempt: int = 0
    total_delay: float = 0.0
    errors: list = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_attempt_at: Optional[datetime] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    
    def add_error(self, error: Exception, attempt: int) -> None:
        """Add an error to the context."""
        self.errors.append({
            'attempt': attempt,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now(timezone.utc)
        })
        self.last_attempt_at = datetime.now(timezone.utc)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a provider."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    is_open: bool = False
    consecutive_successes: int = 0
    
    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self.failure_count = 0
        self.last_failure_time = None
        self.opened_at = None
        self.is_open = False
        self.consecutive_successes = 0


class RetryStrategy:
    """
    Intelligent retry strategy with exponential backoff, jitter, and circuit breaker.
    """
    
    # Default configuration values
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_INITIAL_DELAY = 1.0
    DEFAULT_MAX_DELAY = 60.0
    DEFAULT_BACKOFF_MULTIPLIER = 2.0
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD = 5
    DEFAULT_CIRCUIT_BREAKER_TIMEOUT = 300  # 5 minutes
    
    # Rate limit indicators
    RATE_LIMIT_INDICATORS = {
        "rate limit exceeded",
        "too many requests",
        "quota exceeded",
        "rate_limit_exceeded",
        "429",
        "throttled",
        "rate limited"
    }
    
    # Timeout indicators
    TIMEOUT_INDICATORS = {
        "timeout",
        "timed out",
        "connection timeout",
        "read timeout",
        "request timeout"
    }
    
    # Connection error indicators
    CONNECTION_ERROR_INDICATORS = {
        "connection error",
        "connection failed",
        "network error",
        "connection refused",
        "connection reset",
        "dns resolution failed",
        "host unreachable"
    }
    
    # Model overload indicators
    MODEL_OVERLOAD_INDICATORS = {
        "model is overloaded",
        "service unavailable",
        "temporarily unavailable",
        "server overloaded",
        "capacity exceeded",
        "503"
    }
    
    # Non-retryable error indicators
    NON_RETRYABLE_INDICATORS = {
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "invalid request",
        "bad request",
        "model not found",
        "insufficient quota",
        "content filter",
        "content policy violation",
        "401",
        "403",
        "400",
        "404"
    }
    
    def __init__(
        self,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        backoff_multiplier: Optional[float] = None,
        circuit_breaker_threshold: Optional[int] = None,
        circuit_breaker_timeout: Optional[int] = None,
        jitter: bool = True
    ):
        """
        Initialize retry strategy with configurable parameters.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_multiplier: Exponential backoff multiplier
            circuit_breaker_threshold: Number of failures before opening circuit
            circuit_breaker_timeout: Time to wait before trying a failed provider again
            jitter: Whether to add randomization to delay calculations
        """
        # Load from settings with fallbacks
        self.max_retries = max_retries or getattr(settings, 'MAX_RETRIES', self.DEFAULT_MAX_RETRIES)
        self.initial_delay = initial_delay or getattr(settings, 'INITIAL_DELAY', self.DEFAULT_INITIAL_DELAY)
        self.max_delay = max_delay or getattr(settings, 'MAX_DELAY', self.DEFAULT_MAX_DELAY)
        self.backoff_multiplier = backoff_multiplier or getattr(settings, 'BACKOFF_MULTIPLIER', self.DEFAULT_BACKOFF_MULTIPLIER)
        self.circuit_breaker_threshold = circuit_breaker_threshold or getattr(settings, 'CIRCUIT_BREAKER_THRESHOLD', self.DEFAULT_CIRCUIT_BREAKER_THRESHOLD)
        self.circuit_breaker_timeout = circuit_breaker_timeout or getattr(settings, 'CIRCUIT_BREAKER_TIMEOUT', self.DEFAULT_CIRCUIT_BREAKER_TIMEOUT)
        self.jitter = jitter
        
        # Circuit breaker states by provider
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        logger.info(
            f"RetryStrategy initialized: max_retries={self.max_retries}, "
            f"initial_delay={self.initial_delay}s, max_delay={self.max_delay}s, "
            f"circuit_breaker_threshold={self.circuit_breaker_threshold}"
        )
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """
        Determine if we should retry based on the exception and attempt number.
        
        Args:
            exception: The exception that occurred
            attempt: Current attempt number (1-based)
            
        Returns:
            True if we should retry, False otherwise
        """
        # Check if we've exceeded max retries
        # For max_retries=3: allow retries after attempts 1, 2, 3 but not after attempt 4
        if attempt > self.max_retries:
            logger.debug(f"Max retries ({self.max_retries}) exceeded on attempt {attempt}")
            return False
        
        # Classify the error
        error_type = self._classify_error(exception)
        
        # Never retry non-retryable errors
        if error_type in NonRetryableError:
            logger.info(f"Non-retryable error detected: {error_type.value}")
            return False
        
        # Retry retryable errors
        if error_type in RetryableError:
            logger.info(f"Retryable error detected: {error_type.value}, attempt {attempt}")
            return True
        
        # Default to not retrying unknown errors after first attempt
        logger.warning(f"Unknown error type, not retrying: {type(exception).__name__}: {exception}")
        return False
    
    def get_delay(self, attempt: int, error_type: Optional[RetryableError] = None) -> float:
        """
        Calculate delay for the next retry attempt with exponential backoff and jitter.
        
        Args:
            attempt: Current attempt number (1-based)
            error_type: Type of error (may influence delay calculation)
            
        Returns:
            Delay in seconds before next retry
        """
        # Base delay with exponential backoff
        delay = self.initial_delay * (self.backoff_multiplier ** (attempt - 1))
        
        # Apply max delay cap
        delay = min(delay, self.max_delay)
        
        # Special handling for rate limits - use longer delays
        if error_type == RetryableError.RATE_LIMIT:
            delay = min(delay * 2, self.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.jitter:
            # Add up to 25% jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount
        
        logger.debug(f"Calculated delay for attempt {attempt}: {delay:.2f}s")
        return delay
    
    async def record_success(self, provider: str) -> None:
        """
        Record a successful operation for a provider, potentially closing its circuit.
        
        Args:
            provider: Provider name
        """
        if provider not in self.circuit_breakers:
            return
        
        circuit = self.circuit_breakers[provider]
        circuit.consecutive_successes += 1
        
        # Close circuit if we have enough consecutive successes
        if circuit.is_open and circuit.consecutive_successes >= 2:
            logger.info(f"Circuit breaker closed for provider {provider} after successful operations")
            circuit.reset()
        elif not circuit.is_open:
            # Reset failure count on success
            circuit.failure_count = max(0, circuit.failure_count - 1)
    
    async def record_failure(self, provider: str, exception: Exception) -> None:
        """
        Record a failure for a provider, potentially opening its circuit.
        
        Args:
            provider: Provider name
            exception: The exception that occurred
        """
        if provider not in self.circuit_breakers:
            self.circuit_breakers[provider] = CircuitBreakerState()
        
        circuit = self.circuit_breakers[provider]
        circuit.failure_count += 1
        circuit.last_failure_time = datetime.now(timezone.utc)
        circuit.consecutive_successes = 0
        
        # Open circuit if threshold exceeded
        if not circuit.is_open and circuit.failure_count >= self.circuit_breaker_threshold:
            circuit.is_open = True
            circuit.opened_at = datetime.now(timezone.utc)
            logger.warning(
                f"Circuit breaker opened for provider {provider} after {circuit.failure_count} failures"
            )
    
    def is_circuit_open(self, provider: str) -> bool:
        """
        Check if the circuit breaker is open for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            True if circuit is open (provider should be avoided)
        """
        if provider not in self.circuit_breakers:
            return False
        
        circuit = self.circuit_breakers[provider]
        
        if not circuit.is_open:
            return False
        
        # Check if circuit should be half-opened (timeout expired)
        if circuit.opened_at and datetime.now(timezone.utc) - circuit.opened_at > timedelta(seconds=self.circuit_breaker_timeout):
            logger.info(f"Circuit breaker timeout expired for provider {provider}, allowing retry")
            circuit.is_open = False
            circuit.failure_count = 0
            return False
        
        return True
    
    def get_available_providers(self, providers: list) -> list:
        """
        Filter out providers with open circuits.
        
        Args:
            providers: List of provider names
            
        Returns:
            List of available providers (circuits not open)
        """
        return [provider for provider in providers if not self.is_circuit_open(provider)]
    
    def create_retry_context(self, provider: str = None, model: str = None) -> RetryContext:
        """
        Create a new retry context for tracking retry attempts.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            New RetryContext instance
        """
        return RetryContext(provider=provider, model=model)
    
    async def execute_with_retry(
        self,
        operation,
        context: RetryContext,
        *args,
        **kwargs
    ):
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Async function to execute
            context: RetryContext for tracking
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation
            
        Returns:
            Result of successful operation
            
        Raises:
            Last exception if all retries failed
        """
        last_exception = None
        
        for attempt in range(1, self.max_retries + 2):  # +1 for initial attempt
            # Check circuit breaker for provider
            if context.provider and self.is_circuit_open(context.provider):
                logger.warning(f"Skipping provider {context.provider} - circuit breaker open")
                raise RuntimeError(f"Circuit breaker open for provider {context.provider}")
            
            try:
                context.attempt = attempt
                
                # Execute operation
                result = await operation(*args, **kwargs)
                
                # Record success
                if context.provider:
                    await self.record_success(context.provider)
                
                # Log successful retry if this wasn't the first attempt
                if attempt > 1:
                    logger.info(
                        f"Operation succeeded on attempt {attempt} after {context.total_delay:.2f}s total delay"
                    )
                
                return result
                
            except Exception as e:
                last_exception = e
                context.add_error(e, attempt)
                
                # Record failure for circuit breaker
                if context.provider:
                    await self.record_failure(context.provider, e)
                
                # Check if we should retry
                if not self.should_retry(e, attempt):
                    logger.info(f"Not retrying after attempt {attempt}: {e}")
                    break
                
                # Calculate delay
                error_type = self._classify_error(e)
                delay = self.get_delay(attempt, error_type)
                context.total_delay += delay
                
                logger.warning(
                    f"Attempt {attempt} failed: {e}. "
                    f"Retrying in {delay:.2f}s (total delay: {context.total_delay:.2f}s)"
                )
                
                # Wait before retry
                await asyncio.sleep(delay)
        
        # All retries failed
        total_time = (datetime.now(timezone.utc) - context.started_at).total_seconds()
        logger.error(
            f"Operation failed after {context.attempt} attempts in {total_time:.2f}s. "
            f"Total delay: {context.total_delay:.2f}s. Last error: {last_exception}"
        )
        
        raise last_exception
    
    def _classify_error(self, exception: Exception) -> Union[RetryableError, NonRetryableError, None]:
        """
        Classify an exception as retryable, non-retryable, or unknown.
        
        Args:
            exception: Exception to classify
            
        Returns:
            Error classification or None if unknown
        """
        error_message = str(exception).lower()
        error_type = type(exception).__name__.lower()
        
        # Check for non-retryable errors first
        for indicator in self.NON_RETRYABLE_INDICATORS:
            if indicator in error_message or indicator in error_type:
                if "401" in indicator or "unauthorized" in indicator or "api key" in indicator:
                    return NonRetryableError.INVALID_API_KEY
                elif "400" in indicator or "bad request" in indicator:
                    return NonRetryableError.INVALID_REQUEST
                elif "404" in indicator or "not found" in indicator:
                    return NonRetryableError.MODEL_NOT_FOUND
                elif "quota" in indicator:
                    return NonRetryableError.INSUFFICIENT_QUOTA
                elif "content" in indicator:
                    return NonRetryableError.CONTENT_FILTER
                else:
                    return NonRetryableError.INVALID_REQUEST
        
        # Check for retryable errors
        for indicator in self.RATE_LIMIT_INDICATORS:
            if indicator in error_message or indicator in error_type:
                return RetryableError.RATE_LIMIT
        
        for indicator in self.TIMEOUT_INDICATORS:
            if indicator in error_message or indicator in error_type:
                return RetryableError.TIMEOUT
        
        for indicator in self.CONNECTION_ERROR_INDICATORS:
            if indicator in error_message or indicator in error_type:
                return RetryableError.CONNECTION
        
        for indicator in self.MODEL_OVERLOAD_INDICATORS:
            if indicator in error_message or indicator in error_type:
                return RetryableError.MODEL_OVERLOAD
        
        # Check common exception types
        if isinstance(exception, (TimeoutError, asyncio.TimeoutError)):
            return RetryableError.TIMEOUT
        elif isinstance(exception, (ConnectionError, OSError)):
            return RetryableError.CONNECTION
        
        # Unknown error type
        return None
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict]:
        """
        Get status of all circuit breakers.
        
        Returns:
            Dictionary with circuit breaker status for each provider
        """
        status = {}
        for provider, circuit in self.circuit_breakers.items():
            status[provider] = {
                'is_open': circuit.is_open,
                'failure_count': circuit.failure_count,
                'consecutive_successes': circuit.consecutive_successes,
                'last_failure_time': circuit.last_failure_time.isoformat() if circuit.last_failure_time else None,
                'opened_at': circuit.opened_at.isoformat() if circuit.opened_at else None
            }
        return status
    
    def reset_circuit_breaker(self, provider: str) -> None:
        """
        Manually reset a circuit breaker for a provider.
        
        Args:
            provider: Provider name
        """
        if provider in self.circuit_breakers:
            self.circuit_breakers[provider].reset()
            logger.info(f"Circuit breaker manually reset for provider {provider}")
    
    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        for provider in self.circuit_breakers:
            self.circuit_breakers[provider].reset()
        logger.info("All circuit breakers manually reset")