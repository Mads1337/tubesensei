import asyncio
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import logging

from ..utils.exceptions import RateLimitError

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Token bucket implementation for rate limiting.
    Allows burst traffic while maintaining average rate limit.
    """
    
    def __init__(
        self, 
        rate: float,  # tokens per second
        capacity: Optional[int] = None,
        initial_tokens: Optional[float] = None
    ):
        self.rate = rate
        self.capacity = capacity or rate
        self.tokens = initial_tokens if initial_tokens is not None else self.capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait for tokens (seconds)
        
        Returns:
            True if tokens were acquired, False if timeout
        
        Raises:
            ValueError: If requested tokens exceed capacity
        """
        if tokens > self.capacity:
            raise ValueError(f"Requested {tokens} tokens exceeds capacity {self.capacity}")
        
        start_time = time.monotonic()
        
        while True:
            async with self._lock:
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.monotonic() - start_time
                    if elapsed + wait_time > timeout:
                        return False
            
            # Wait before retrying
            await asyncio.sleep(min(wait_time, 0.1))
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.monotonic()
        elapsed = now - self.last_update
        
        # Add tokens based on rate
        tokens_to_add = elapsed * self.rate
        self.tokens = min(self.tokens + tokens_to_add, self.capacity)
        self.last_update = now
    
    @property
    def available_tokens(self) -> float:
        """Get current available tokens (thread-safe snapshot)"""
        self._refill()
        return self.tokens


class RateLimiter:
    """
    Advanced rate limiter with multiple strategies and retry logic.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_capacity: Optional[int] = None,
        retry_attempts: int = 3,
        backoff_factor: float = 2.0
    ):
        self.requests_per_minute = requests_per_minute
        self.rate_per_second = requests_per_minute / 60.0
        self.burst_capacity = burst_capacity or min(requests_per_minute, 10)
        self.retry_attempts = retry_attempts
        self.backoff_factor = backoff_factor
        
        # Create token bucket
        self.bucket = TokenBucket(
            rate=self.rate_per_second,
            capacity=self.burst_capacity
        )
        
        # Track statistics
        self.stats: Dict[str, Any] = {
            "total_requests": 0,
            "total_throttled": 0,
            "total_rejected": 0,
            "last_request_time": None
        }
    
    @asynccontextmanager
    async def acquire(self, tokens: int = 1, timeout: Optional[float] = 30):
        """
        Context manager for rate-limited operations.
        
        Usage:
            async with rate_limiter.acquire():
                # Perform rate-limited operation
                await make_api_call()
        """
        acquired = False
        try:
            # Try to acquire tokens
            acquired = await self.bucket.acquire(tokens, timeout)
            
            if not acquired:
                self.stats["total_rejected"] += 1
                raise RateLimitError(
                    f"Could not acquire {tokens} tokens within {timeout} seconds"
                )
            
            self.stats["total_requests"] += 1
            self.stats["last_request_time"] = time.time()
            
            yield
            
        except Exception as e:
            if not acquired:
                self.stats["total_throttled"] += 1
            raise
    
    async def execute_with_retry(
        self,
        func,
        *args,
        tokens: int = 1,
        **kwargs
    ):
        """
        Execute a function with rate limiting and retry logic.
        
        Args:
            func: Async function to execute
            tokens: Number of tokens required
            *args, **kwargs: Arguments for the function
        
        Returns:
            Result of the function call
        """
        last_error = None
        wait_time = 1.0
        
        for attempt in range(self.retry_attempts):
            try:
                async with self.acquire(tokens=tokens):
                    return await func(*args, **kwargs)
                    
            except RateLimitError as e:
                last_error = e
                logger.warning(
                    f"Rate limit hit on attempt {attempt + 1}/{self.retry_attempts}. "
                    f"Waiting {wait_time:.1f}s before retry..."
                )
                
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(wait_time)
                    wait_time *= self.backoff_factor
                    
            except Exception as e:
                # Non-rate-limit errors should bubble up
                raise
        
        # All retries exhausted
        raise last_error or RateLimitError("Rate limit exceeded after all retries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            **self.stats,
            "available_tokens": self.bucket.available_tokens,
            "capacity": self.bucket.capacity,
            "rate_per_second": self.bucket.rate
        }
    
    async def wait_if_needed(self, tokens: int = 1) -> float:
        """
        Calculate and wait if necessary before making a request.
        
        Returns:
            Time waited in seconds
        """
        start_time = time.monotonic()
        
        # Check if we need to wait
        if self.bucket.available_tokens < tokens:
            tokens_needed = tokens - self.bucket.available_tokens
            wait_time = tokens_needed / self.bucket.rate
            
            logger.debug(f"Rate limiter: waiting {wait_time:.2f}s for {tokens} tokens")
            await asyncio.sleep(wait_time)
        
        return time.monotonic() - start_time


class MultiServiceRateLimiter:
    """
    Rate limiter that manages multiple services with different limits.
    """
    
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
    
    def add_service(
        self,
        service_name: str,
        requests_per_minute: int,
        burst_capacity: Optional[int] = None
    ):
        """Add a service with its rate limit configuration"""
        self.limiters[service_name] = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_capacity=burst_capacity
        )
    
    def get_limiter(self, service_name: str) -> RateLimiter:
        """Get rate limiter for a specific service"""
        if service_name not in self.limiters:
            raise ValueError(f"No rate limiter configured for service: {service_name}")
        return self.limiters[service_name]
    
    @asynccontextmanager
    async def acquire(self, service_name: str, tokens: int = 1):
        """Acquire tokens for a specific service"""
        limiter = self.get_limiter(service_name)
        async with limiter.acquire(tokens=tokens):
            yield
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all services"""
        return {
            service: limiter.get_stats()
            for service, limiter in self.limiters.items()
        }