"""
Tests for the RetryStrategy module - Phase 2A
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from tubesensei.app.ai.retry_strategy import (
    RetryStrategy,
    RetryContext,
    RetryableError,
    NonRetryableError,
    CircuitBreakerState
)


class TestRetryStrategy:
    """Test suite for RetryStrategy class."""
    
    @pytest.fixture
    def retry_strategy(self):
        """Create a RetryStrategy instance for testing."""
        return RetryStrategy(
            max_retries=3,
            initial_delay=0.1,  # Short delay for tests
            max_delay=1.0,
            backoff_multiplier=2.0,
            circuit_breaker_threshold=5,  # Higher than max_retries to avoid interference
            circuit_breaker_timeout=60,
            jitter=False  # Disable jitter for predictable tests
        )
    
    def test_initialization(self, retry_strategy):
        """Test RetryStrategy initialization."""
        assert retry_strategy.max_retries == 3
        assert retry_strategy.initial_delay == 0.1
        assert retry_strategy.max_delay == 1.0
        assert retry_strategy.backoff_multiplier == 2.0
        assert retry_strategy.circuit_breaker_threshold == 5
        assert retry_strategy.circuit_breaker_timeout == 60
        assert retry_strategy.jitter is False
        assert retry_strategy.circuit_breakers == {}
    
    def test_error_classification_retryable(self, retry_strategy):
        """Test classification of retryable errors."""
        test_cases = [
            (Exception("rate limit exceeded"), RetryableError.RATE_LIMIT),
            (Exception("429 Too Many Requests"), RetryableError.RATE_LIMIT),
            (Exception("connection timeout"), RetryableError.TIMEOUT),
            (Exception("timed out after 30s"), RetryableError.TIMEOUT),
            (Exception("connection error"), RetryableError.CONNECTION),
            (Exception("connection refused"), RetryableError.CONNECTION),
            (Exception("service unavailable - 503"), RetryableError.MODEL_OVERLOAD),
            (Exception("model is overloaded"), RetryableError.MODEL_OVERLOAD),
        ]
        
        for exception, expected_type in test_cases:
            result = retry_strategy._classify_error(exception)
            assert result == expected_type, f"Failed for {exception}: got {result}, expected {expected_type}"
    
    def test_error_classification_non_retryable(self, retry_strategy):
        """Test classification of non-retryable errors."""
        test_cases = [
            (Exception("invalid api key"), NonRetryableError.INVALID_API_KEY),
            (Exception("401 unauthorized"), NonRetryableError.INVALID_API_KEY),
            (Exception("bad request - 400"), NonRetryableError.INVALID_REQUEST),
            (Exception("invalid request format"), NonRetryableError.INVALID_REQUEST),
            (Exception("model not found - 404"), NonRetryableError.MODEL_NOT_FOUND),
            (Exception("insufficient quota"), NonRetryableError.INSUFFICIENT_QUOTA),
            (Exception("content policy violation"), NonRetryableError.CONTENT_FILTER),
        ]
        
        for exception, expected_type in test_cases:
            result = retry_strategy._classify_error(exception)
            assert result == expected_type, f"Failed for {exception}: got {result}, expected {expected_type}"
    
    def test_should_retry_logic(self, retry_strategy):
        """Test should_retry decision logic."""
        # Retryable errors should retry within limits
        retryable_error = Exception("rate limit exceeded")
        assert retry_strategy.should_retry(retryable_error, 1) is True
        assert retry_strategy.should_retry(retryable_error, 2) is True
        assert retry_strategy.should_retry(retryable_error, 3) is True
        assert retry_strategy.should_retry(retryable_error, 4) is False  # Exceeds max_retries
        
        # Non-retryable errors should never retry
        non_retryable_error = Exception("invalid api key")
        assert retry_strategy.should_retry(non_retryable_error, 1) is False
        assert retry_strategy.should_retry(non_retryable_error, 2) is False
        
        # Unknown errors should not retry
        unknown_error = Exception("some unknown error")
        assert retry_strategy.should_retry(unknown_error, 1) is False
    
    def test_delay_calculation(self, retry_strategy):
        """Test exponential backoff delay calculation."""
        # Without jitter, delays should be predictable
        assert retry_strategy.get_delay(1) == 0.1  # initial_delay
        assert retry_strategy.get_delay(2) == 0.2  # initial_delay * multiplier^1
        assert retry_strategy.get_delay(3) == 0.4  # initial_delay * multiplier^2
        assert retry_strategy.get_delay(4) == 0.8  # initial_delay * multiplier^3
        
        # Should cap at max_delay
        assert retry_strategy.get_delay(10) == 1.0  # max_delay
        
        # Rate limit errors should have longer delays
        rate_limit_delay = retry_strategy.get_delay(1, RetryableError.RATE_LIMIT)
        normal_delay = retry_strategy.get_delay(1)
        assert rate_limit_delay > normal_delay
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, retry_strategy):
        """Test circuit breaker open/close functionality."""
        provider = "test_provider"
        
        # Initially circuit should be closed
        assert retry_strategy.is_circuit_open(provider) is False
        
        # Record failures up to threshold
        for i in range(5):  # threshold is 5
            await retry_strategy.record_failure(provider, Exception(f"failure {i}"))
        
        # Circuit should now be open
        assert retry_strategy.is_circuit_open(provider) is True
        
        # Available providers should exclude the failed one
        providers = ["openai", "test_provider", "anthropic"]
        available = retry_strategy.get_available_providers(providers)
        assert "test_provider" not in available
        assert len(available) == 2
        
        # Record success to start recovery
        await retry_strategy.record_success(provider)
        await retry_strategy.record_success(provider)  # Need 2 consecutive successes
        
        # Circuit should be closed again
        assert retry_strategy.is_circuit_open(provider) is False
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_timeout(self, retry_strategy):
        """Test circuit breaker timeout functionality."""
        provider = "test_provider"
        
        # Open circuit by exceeding threshold
        for i in range(5):
            await retry_strategy.record_failure(provider, Exception(f"failure {i}"))
        
        assert retry_strategy.is_circuit_open(provider) is True
        
        # Manually set opened_at to past to simulate timeout
        circuit = retry_strategy.circuit_breakers[provider]
        circuit.opened_at = datetime.now(timezone.utc) - timedelta(seconds=65)  # Past timeout
        
        # Circuit should be closed due to timeout
        assert retry_strategy.is_circuit_open(provider) is False
    
    def test_retry_context_creation(self, retry_strategy):
        """Test RetryContext creation and management."""
        context = retry_strategy.create_retry_context("openai", "gpt-4")
        
        assert context.attempt == 0
        assert context.total_delay == 0.0
        assert context.errors == []
        assert context.provider == "openai"
        assert context.model == "gpt-4"
        assert isinstance(context.started_at, datetime)
        
        # Test adding errors
        error = Exception("test error")
        context.add_error(error, 1)
        
        assert len(context.errors) == 1
        assert context.errors[0]['attempt'] == 1
        assert context.errors[0]['error_type'] == 'Exception'
        assert context.errors[0]['error_message'] == 'test error'
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, retry_strategy):
        """Test execute_with_retry with eventual success."""
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("rate limit exceeded")
            return "success"
        
        context = retry_strategy.create_retry_context("openai", "gpt-4")
        result = await retry_strategy.execute_with_retry(mock_operation, context)
        
        assert result == "success"
        assert call_count == 3
        assert context.attempt == 3
        assert len(context.errors) == 2  # Two failures before success
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure(self, retry_strategy):
        """Test execute_with_retry with all attempts failing."""
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            raise Exception("rate limit exceeded")
        
        context = retry_strategy.create_retry_context("openai", "gpt-4")
        
        with pytest.raises(Exception, match="rate limit exceeded"):
            await retry_strategy.execute_with_retry(mock_operation, context)
        
        assert call_count == 4  # Initial attempt + 3 retries
        assert context.attempt == 4
        assert len(context.errors) == 4
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retryable(self, retry_strategy):
        """Test execute_with_retry with non-retryable error."""
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            raise Exception("invalid api key")
        
        context = retry_strategy.create_retry_context("openai", "gpt-4")
        
        with pytest.raises(Exception, match="invalid api key"):
            await retry_strategy.execute_with_retry(mock_operation, context)
        
        assert call_count == 1  # Should not retry
        assert context.attempt == 1
        assert len(context.errors) == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_circuit_open(self, retry_strategy):
        """Test execute_with_retry with open circuit."""
        # Open circuit for provider
        provider = "openai"
        for i in range(5):
            await retry_strategy.record_failure(provider, Exception(f"failure {i}"))
        
        async def mock_operation():
            return "should not be called"
        
        context = retry_strategy.create_retry_context(provider, "gpt-4")
        
        with pytest.raises(RuntimeError, match="Circuit breaker open"):
            await retry_strategy.execute_with_retry(mock_operation, context)
    
    def test_circuit_breaker_status(self, retry_strategy):
        """Test circuit breaker status reporting."""
        # Initially no circuit breakers
        status = retry_strategy.get_circuit_breaker_status()
        assert status == {}
        
        # Add some failures
        asyncio.run(retry_strategy.record_failure("provider1", Exception("error")))
        asyncio.run(retry_strategy.record_failure("provider2", Exception("error")))
        
        status = retry_strategy.get_circuit_breaker_status()
        assert len(status) == 2
        assert "provider1" in status
        assert "provider2" in status
        assert status["provider1"]["failure_count"] == 1
        assert status["provider1"]["is_open"] is False
    
    def test_manual_circuit_breaker_reset(self, retry_strategy):
        """Test manual circuit breaker reset functionality."""
        provider = "test_provider"
        
        # Open circuit
        for i in range(5):
            asyncio.run(retry_strategy.record_failure(provider, Exception(f"failure {i}")))
        
        assert retry_strategy.is_circuit_open(provider) is True
        
        # Reset circuit breaker
        retry_strategy.reset_circuit_breaker(provider)
        assert retry_strategy.is_circuit_open(provider) is False
        
        # Reset all circuit breakers
        asyncio.run(retry_strategy.record_failure(provider, Exception("new failure")))
        assert retry_strategy.circuit_breakers[provider].failure_count == 1
        
        retry_strategy.reset_all_circuit_breakers()
        assert retry_strategy.circuit_breakers[provider].failure_count == 0


@pytest.mark.asyncio
async def test_integration_with_timeout_errors():
    """Test retry strategy handles timeout errors correctly."""
    strategy = RetryStrategy(max_retries=2, initial_delay=0.01)
    
    call_count = 0
    
    async def timeout_operation():
        nonlocal call_count
        call_count += 1
        raise asyncio.TimeoutError("Request timeout")
    
    context = strategy.create_retry_context("openai", "gpt-4")
    
    with pytest.raises(asyncio.TimeoutError):
        await strategy.execute_with_retry(timeout_operation, context)
    
    assert call_count == 3  # Initial + 2 retries
    assert context.attempt == 3
    
    # Check that timeout errors are classified correctly
    assert strategy._classify_error(asyncio.TimeoutError()) == RetryableError.TIMEOUT


@pytest.mark.asyncio 
async def test_integration_with_connection_errors():
    """Test retry strategy handles connection errors correctly."""
    strategy = RetryStrategy(max_retries=2, initial_delay=0.01)
    
    call_count = 0
    
    async def connection_operation():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Connection failed")
    
    context = strategy.create_retry_context("anthropic", "claude-3")
    
    with pytest.raises(ConnectionError):
        await strategy.execute_with_retry(connection_operation, context)
    
    assert call_count == 3  # Initial + 2 retries
    
    # Check that connection errors are classified correctly
    assert strategy._classify_error(ConnectionError()) == RetryableError.CONNECTION