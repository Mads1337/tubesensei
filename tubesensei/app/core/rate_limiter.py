"""
Rate limiting middleware for TubeSensei API.

Uses Redis sliding window algorithm for accurate rate limiting.
Returns 429 Too Many Requests with Retry-After header when limit exceeded.
"""
import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiting middleware.

    Per-IP rate limiting using Redis. Skips health check endpoints.
    Returns 429 with Retry-After and X-RateLimit-* headers.
    """

    def __init__(
        self,
        app: ASGIApp,
        redis_url: str,
        requests_per_minute: int = 60,
        enabled: bool = True,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self.enabled = enabled
        self._redis_url = redis_url
        self._redis = None

    def _get_redis(self):
        """Lazy Redis connection."""
        if self._redis is None:
            try:
                import redis
                self._redis = redis.from_url(self._redis_url, decode_responses=True)
            except Exception as e:
                logger.warning(f"Rate limiter: Redis connection failed: {e}")
                self._redis = None
        return self._redis

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request headers."""
        # Check common proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        if request.client:
            return request.client.host
        return "unknown"

    def _should_skip(self, path: str) -> bool:
        """Skip rate limiting for health checks and static files."""
        skip_prefixes = ["/health", "/static", "/favicon"]
        return any(path.startswith(p) for p in skip_prefixes)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self.enabled:
            return await call_next(request)

        if self._should_skip(request.url.path):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        r = self._get_redis()

        if r is None:
            # Fail open if Redis is unavailable
            return await call_next(request)

        # Sliding window using Redis sorted set
        key = f"ratelimit:{client_ip}"
        now = time.time()
        window_start = now - self.window_seconds

        try:
            pipe = r.pipeline()
            # Remove old entries outside window
            pipe.zremrangebyscore(key, 0, window_start)
            # Count requests in current window
            pipe.zcard(key)
            # Add current request
            pipe.zadd(key, {str(now): now})
            # Set expiry
            pipe.expire(key, self.window_seconds * 2)
            results = pipe.execute()

            current_count = results[1]  # count BEFORE adding current request
            remaining = max(0, self.requests_per_minute - current_count - 1)
            reset_time = int(now) + self.window_seconds

            # Set rate limit headers on all responses
            headers = {
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": str(remaining),
                "X-RateLimit-Reset": str(reset_time),
            }

            if current_count >= self.requests_per_minute:
                retry_after = self.window_seconds
                headers["Retry-After"] = str(retry_after)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too Many Requests",
                        "detail": f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute.",
                        "retry_after": retry_after,
                    },
                    headers=headers,
                )
        except Exception as e:
            logger.warning(f"Rate limiter error: {e}")
            return await call_next(request)

        response = await call_next(request)
        # Add rate limit headers to successful responses
        for header, value in headers.items():
            response.headers[header] = value
        return response
