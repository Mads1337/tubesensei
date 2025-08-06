import pytest
import asyncio
import time
from app.utils.rate_limiter import TokenBucket, RateLimiter, MultiServiceRateLimiter
from app.utils.exceptions import RateLimitError


class TestTokenBucket:
    """Test TokenBucket implementation"""
    
    @pytest.mark.asyncio
    async def test_token_bucket_basic(self):
        """Test basic token bucket functionality"""
        bucket = TokenBucket(rate=10, capacity=10)
        
        # Should be able to acquire initial tokens
        assert await bucket.acquire(5) is True
        assert bucket.available_tokens == pytest.approx(5, rel=0.1)
        
        # Should be able to acquire remaining tokens
        assert await bucket.acquire(5) is True
        assert bucket.available_tokens == pytest.approx(0, rel=0.1)
    
    @pytest.mark.asyncio
    async def test_token_bucket_refill(self):
        """Test token bucket refill over time"""
        bucket = TokenBucket(rate=10, capacity=10, initial_tokens=0)
        
        # Initially no tokens
        assert bucket.available_tokens == pytest.approx(0, rel=0.1)
        
        # Wait for refill
        await asyncio.sleep(0.5)
        
        # Should have refilled ~5 tokens (10 per second * 0.5 seconds)
        assert bucket.available_tokens == pytest.approx(5, rel=1)
    
    @pytest.mark.asyncio
    async def test_token_bucket_capacity_limit(self):
        """Test that tokens don't exceed capacity"""
        bucket = TokenBucket(rate=100, capacity=10, initial_tokens=10)
        
        # Wait for potential overfill
        await asyncio.sleep(1)
        
        # Should still be at capacity
        assert bucket.available_tokens <= 10
    
    @pytest.mark.asyncio
    async def test_token_bucket_timeout(self):
        """Test token acquisition timeout"""
        bucket = TokenBucket(rate=1, capacity=1, initial_tokens=0)
        
        # Should timeout when trying to acquire tokens
        result = await bucket.acquire(1, timeout=0.1)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_token_bucket_exceed_capacity_error(self):
        """Test error when requesting more than capacity"""
        bucket = TokenBucket(rate=10, capacity=10)
        
        with pytest.raises(ValueError):
            await bucket.acquire(20)


class TestRateLimiter:
    """Test RateLimiter functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Test basic rate limiting"""
        limiter = RateLimiter(requests_per_minute=60, burst_capacity=5)
        
        # Should allow initial burst
        for _ in range(5):
            async with limiter.acquire():
                pass
        
        assert limiter.stats['total_requests'] == 5
        assert limiter.stats['total_rejected'] == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_rejection(self):
        """Test rate limit rejection"""
        limiter = RateLimiter(
            requests_per_minute=60,
            burst_capacity=2,
            retry_attempts=1
        )
        
        # Use up burst capacity
        async with limiter.acquire():
            pass
        async with limiter.acquire():
            pass
        
        # Next request should be rejected quickly
        with pytest.raises(RateLimitError):
            async with limiter.acquire(timeout=0.1):
                pass
    
    @pytest.mark.asyncio
    async def test_rate_limiter_stats(self):
        """Test rate limiter statistics"""
        limiter = RateLimiter(requests_per_minute=120)
        
        # Make some requests
        async with limiter.acquire():
            pass
        async with limiter.acquire():
            pass
        
        stats = limiter.get_stats()
        assert stats['total_requests'] == 2
        assert stats['total_rejected'] == 0
        assert stats['total_throttled'] == 0
        assert 'available_tokens' in stats
        assert 'capacity' in stats
        assert 'rate_per_second' in stats
    
    @pytest.mark.asyncio
    async def test_rate_limiter_execute_with_retry(self):
        """Test execute with retry functionality"""
        limiter = RateLimiter(
            requests_per_minute=60,
            retry_attempts=3,
            backoff_factor=1.5
        )
        
        call_count = 0
        
        async def test_function():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await limiter.execute_with_retry(test_function)
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiter_wait_if_needed(self):
        """Test wait_if_needed functionality"""
        limiter = RateLimiter(requests_per_minute=60, burst_capacity=1)
        
        # Use up capacity
        async with limiter.acquire():
            pass
        
        # Should wait before next request
        start = time.monotonic()
        wait_time = await limiter.wait_if_needed()
        elapsed = time.monotonic() - start
        
        assert wait_time > 0
        assert elapsed >= wait_time


class TestMultiServiceRateLimiter:
    """Test MultiServiceRateLimiter functionality"""
    
    @pytest.mark.asyncio
    async def test_multi_service_basic(self):
        """Test basic multi-service rate limiting"""
        multi_limiter = MultiServiceRateLimiter()
        
        # Add services with different limits
        multi_limiter.add_service('service1', requests_per_minute=60)
        multi_limiter.add_service('service2', requests_per_minute=30)
        
        # Use service1
        async with multi_limiter.acquire('service1'):
            pass
        
        # Use service2
        async with multi_limiter.acquire('service2'):
            pass
        
        # Check stats
        all_stats = multi_limiter.get_all_stats()
        assert 'service1' in all_stats
        assert 'service2' in all_stats
        assert all_stats['service1']['total_requests'] == 1
        assert all_stats['service2']['total_requests'] == 1
    
    @pytest.mark.asyncio
    async def test_multi_service_independent_limits(self):
        """Test that services have independent rate limits"""
        multi_limiter = MultiServiceRateLimiter()
        
        # Service1 with high limit, service2 with low limit
        multi_limiter.add_service('fast', requests_per_minute=600, burst_capacity=10)
        multi_limiter.add_service('slow', requests_per_minute=60, burst_capacity=2)
        
        # Fast service should handle many requests
        for _ in range(5):
            async with multi_limiter.acquire('fast'):
                pass
        
        # Slow service should be limited after burst
        async with multi_limiter.acquire('slow'):
            pass
        async with multi_limiter.acquire('slow'):
            pass
        
        # Next slow request should fail quickly
        with pytest.raises(RateLimitError):
            async with multi_limiter.acquire('slow', tokens=1):
                await asyncio.wait_for(asyncio.sleep(0.1), timeout=0.05)
    
    def test_multi_service_invalid_service(self):
        """Test error for invalid service name"""
        multi_limiter = MultiServiceRateLimiter()
        
        with pytest.raises(ValueError):
            multi_limiter.get_limiter('nonexistent')