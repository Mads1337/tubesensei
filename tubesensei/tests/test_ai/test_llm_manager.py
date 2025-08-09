"""
Tests for llm_manager.py module.

Tests the ModelType enum, LLMResponse dataclass, LLMManager initialization,
completion methods, cost tracking, and cache functionality.
"""

import pytest
import json
import time
import asyncio
from decimal import Decimal
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import asdict

from tubesensei.app.ai.llm_manager import (
    ModelType,
    LLMResponse, 
    LLMManager,
    MODEL_COSTS
)


class TestModelType:
    """Test ModelType enum values."""
    
    def test_model_type_enum_values(self):
        """Test that all expected model types exist."""
        expected_types = ["fast", "balanced", "quality"]
        actual_types = [mt.value for mt in ModelType]
        
        assert set(expected_types) == set(actual_types)
        assert len(ModelType) == 3
    
    def test_model_type_accessibility(self):
        """Test that all model types are accessible."""
        assert ModelType.FAST.value == "fast"
        assert ModelType.BALANCED.value == "balanced" 
        assert ModelType.QUALITY.value == "quality"


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_llm_response_instantiation(self):
        """Test LLMResponse can be created with required fields."""
        response = LLMResponse(
            content="Test response content",
            model="gpt-4",
            provider="openai",
            tokens_used=150,
            cost=0.045,
            processing_time=2.5
        )
        
        assert response.content == "Test response content"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.tokens_used == 150
        assert response.cost == 0.045
        assert response.processing_time == 2.5
        assert response.cached is False  # Default value
    
    def test_llm_response_with_cached_flag(self):
        """Test LLMResponse with cached flag set to True."""
        response = LLMResponse(
            content="Cached content",
            model="claude-3-sonnet",
            provider="anthropic",
            tokens_used=200,
            cost=0.0,
            processing_time=0.1,
            cached=True
        )
        
        assert response.cached is True
    
    def test_llm_response_serialization(self):
        """Test LLMResponse can be serialized to dict."""
        response = LLMResponse(
            content="Test",
            model="gpt-3.5-turbo",
            provider="openai", 
            tokens_used=100,
            cost=0.002,
            processing_time=1.0
        )
        
        response_dict = asdict(response)
        
        assert response_dict["content"] == "Test"
        assert response_dict["model"] == "gpt-3.5-turbo"
        assert response_dict["provider"] == "openai"
        assert response_dict["cached"] is False


class TestLLMManagerInitialization:
    """Test LLMManager initialization and setup."""
    
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_llm_manager_initialization(self, mock_settings):
        """Test LLMManager initializes with default values."""
        mock_settings.DEBUG = False
        mock_settings.OPENAI_API_KEY = "test_key"
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.GOOGLE_API_KEY = None
        mock_settings.DEEPSEEK_API_KEY = None
        
        with patch('tubesensei.app.ai.llm_manager.Router') as mock_router:
            manager = LLMManager()
            
            assert manager.router is not None
            assert manager.redis_client is None
            assert manager.total_cost == Decimal('0.00')
            assert manager.request_count == 0
            assert manager.cost_by_model == {}
            assert manager.cost_by_provider == {}
    
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_model_config_structure(self, mock_settings):
        """Test that MODEL_CONFIG contains expected model lists."""
        mock_settings.DEBUG = False
        mock_settings.OPENAI_API_KEY = "test_key"
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.GOOGLE_API_KEY = None
        mock_settings.DEEPSEEK_API_KEY = None
        
        with patch('tubesensei.app.ai.llm_manager.Router'):
            manager = LLMManager()
            
            # Check that all model types have corresponding configurations
            for model_type in ModelType:
                assert model_type in manager.MODEL_CONFIG
                assert isinstance(manager.MODEL_CONFIG[model_type], list)
                assert len(manager.MODEL_CONFIG[model_type]) > 0
    
    @patch('tubesensei.app.ai.llm_manager.settings')
    @patch('tubesensei.app.ai.llm_manager.logger')
    def test_router_setup_no_api_keys(self, mock_logger, mock_settings):
        """Test router setup behavior when no API keys are configured."""
        mock_settings.DEBUG = False
        mock_settings.OPENAI_API_KEY = None
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.GOOGLE_API_KEY = None
        mock_settings.DEEPSEEK_API_KEY = None
        
        with patch('tubesensei.app.ai.llm_manager.Router') as mock_router:
            manager = LLMManager()
            
            # Should log warning about no API keys
            mock_logger.warning.assert_called_with("No LLM API keys configured")
    
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_router_setup_with_multiple_providers(self, mock_settings):
        """Test router setup with multiple API keys."""
        mock_settings.DEBUG = False
        mock_settings.OPENAI_API_KEY = "openai_key"
        mock_settings.ANTHROPIC_API_KEY = "anthropic_key"
        mock_settings.GOOGLE_API_KEY = "google_key"
        mock_settings.DEEPSEEK_API_KEY = "deepseek_key"
        
        with patch('tubesensei.app.ai.llm_manager.Router') as mock_router:
            manager = LLMManager()
            
            # Router should be called with model list
            mock_router.assert_called_once()
            call_args = mock_router.call_args
            model_list = call_args[1]['model_list']
            
            # Should have models from all providers
            model_names = [model['model_name'] for model in model_list]
            assert any('gpt-' in name for name in model_names)  # OpenAI
            assert any('claude-' in name for name in model_names)  # Anthropic
            assert any('gemini-' in name for name in model_names)  # Google
            assert 'deepseek-chat' in model_names  # DeepSeek


class TestRedisInitialization:
    """Test Redis client initialization."""
    
    @patch('tubesensei.app.ai.llm_manager.aioredis')
    @patch('tubesensei.app.ai.llm_manager.settings')
    async def test_redis_initialization_success(self, mock_settings, mock_aioredis):
        """Test successful Redis client initialization."""
        mock_settings.REDIS_URL = "redis://localhost:6379"
        mock_redis_client = AsyncMock()
        mock_aioredis.from_url.return_value = mock_redis_client
        
        with patch('tubesensei.app.ai.llm_manager.Router'):
            manager = LLMManager()
            await manager.initialize()
            
            assert manager.redis_client == mock_redis_client
            mock_redis_client.ping.assert_called_once()
    
    @patch('tubesensei.app.ai.llm_manager.aioredis', None)  # Simulate missing aioredis
    @patch('tubesensei.app.ai.llm_manager.logger')
    async def test_redis_unavailable(self, mock_logger):
        """Test behavior when aioredis is not available."""
        with patch('tubesensei.app.ai.llm_manager.Router'):
            manager = LLMManager()
            await manager.initialize()
            
            assert manager.redis_client is None
            mock_logger.warning.assert_called_with("Redis not available - caching disabled")
    
    @patch('tubesensei.app.ai.llm_manager.aioredis')
    @patch('tubesensei.app.ai.llm_manager.settings')
    @patch('tubesensei.app.ai.llm_manager.logger')
    async def test_redis_connection_failure(self, mock_logger, mock_settings, mock_aioredis):
        """Test Redis connection failure handling."""
        mock_settings.REDIS_URL = "redis://invalid:6379"
        mock_redis_client = AsyncMock()
        mock_redis_client.ping.side_effect = Exception("Connection failed")
        mock_aioredis.from_url.return_value = mock_redis_client
        
        with patch('tubesensei.app.ai.llm_manager.Router'):
            manager = LLMManager()
            await manager.initialize()  # Should not raise exception
            
            mock_logger.error.assert_called()


class TestCacheKeyGeneration:
    """Test cache key generation."""
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    def test_cache_key_generation_consistency(self, mock_router):
        """Test that identical inputs generate identical cache keys."""
        manager = LLMManager()
        
        messages = [{"role": "user", "content": "Test message"}]
        model_type = "gpt-4"
        temperature = 0.7
        max_tokens = 150
        
        key1 = manager._generate_cache_key(
            messages, model_type, temperature, max_tokens
        )
        key2 = manager._generate_cache_key(
            messages, model_type, temperature, max_tokens
        )
        
        assert key1 == key2
        assert key1.startswith("llm_cache:")
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    def test_cache_key_generation_different_inputs(self, mock_router):
        """Test that different inputs generate different cache keys."""
        manager = LLMManager()
        
        messages1 = [{"role": "user", "content": "Message 1"}]
        messages2 = [{"role": "user", "content": "Message 2"}]
        
        key1 = manager._generate_cache_key(messages1, "gpt-4", 0.7)
        key2 = manager._generate_cache_key(messages2, "gpt-4", 0.7)
        
        assert key1 != key2
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    def test_cache_key_generation_with_kwargs(self, mock_router):
        """Test cache key generation includes kwargs."""
        manager = LLMManager()
        
        messages = [{"role": "user", "content": "Test"}]
        
        key1 = manager._generate_cache_key(
            messages, "gpt-4", 0.7, extra_param="value1"
        )
        key2 = manager._generate_cache_key(
            messages, "gpt-4", 0.7, extra_param="value2"
        )
        
        assert key1 != key2


class TestCacheOperations:
    """Test cache get and set operations."""
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    async def test_get_cached_response_not_found(self, mock_router):
        """Test cache get when response is not cached."""
        manager = LLMManager()
        manager.redis_client = AsyncMock()
        manager.redis_client.get.return_value = None
        
        result = await manager._get_cached_response("test_key")
        
        assert result is None
        manager.redis_client.get.assert_called_once_with("test_key")
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    async def test_get_cached_response_found(self, mock_router):
        """Test successful cache retrieval."""
        manager = LLMManager()
        manager.redis_client = AsyncMock()
        
        cached_data = {
            "content": "Cached content",
            "model": "gpt-4",
            "provider": "openai",
            "tokens_used": 100,
            "cost": 0.03,
            "processing_time": 1.5
        }
        manager.redis_client.get.return_value = json.dumps(cached_data)
        
        result = await manager._get_cached_response("test_key")
        
        assert isinstance(result, LLMResponse)
        assert result.content == "Cached content"
        assert result.model == "gpt-4"
        assert result.cached is True
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.logger')
    async def test_get_cached_response_malformed_json(self, mock_logger, mock_router):
        """Test cache retrieval with malformed JSON."""
        manager = LLMManager()
        manager.redis_client = AsyncMock()
        manager.redis_client.get.return_value = "invalid json"
        
        result = await manager._get_cached_response("test_key")
        
        assert result is None
        mock_logger.warning.assert_called()
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    async def test_cache_response_success(self, mock_settings, mock_router):
        """Test successful cache storage."""
        mock_settings.LLM_CACHE_TTL = 3600
        
        manager = LLMManager()
        manager.redis_client = AsyncMock()
        
        response = LLMResponse(
            content="Test content",
            model="gpt-4",
            provider="openai",
            tokens_used=100,
            cost=0.03,
            processing_time=1.0
        )
        
        await manager._cache_response("test_key", response)
        
        manager.redis_client.setex.assert_called_once()
        call_args = manager.redis_client.setex.call_args
        assert call_args[0][0] == "test_key"
        assert call_args[0][1] == 3600  # TTL
        
        # Verify JSON data
        json_data = json.loads(call_args[0][2])
        assert json_data["content"] == "Test content"
        assert json_data["model"] == "gpt-4"


class TestCompletionMethod:
    """Test the complete() method with mocked dependencies."""
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.completion')
    @patch('tubesensei.app.ai.llm_manager.settings')
    async def test_complete_success(self, mock_settings, mock_completion, mock_router_class):
        """Test successful completion."""
        # Setup settings
        mock_settings.DEBUG = False
        mock_settings.OPENAI_API_KEY = "test_key"
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.GOOGLE_API_KEY = None
        mock_settings.DEEPSEEK_API_KEY = None
        mock_settings.LLM_DEFAULT_TEMPERATURE = 0.7
        mock_settings.LLM_COST_TRACKING_ENABLED = True
        
        # Setup mock completion response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test response content"
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        
        # Use asyncio.to_thread to handle the async completion call
        async def mock_to_thread(func, *args, **kwargs):
            return mock_response
        
        with patch('asyncio.to_thread', side_effect=mock_to_thread):
            mock_completion.return_value = mock_response
            
            manager = LLMManager()
            
            messages = [{"role": "user", "content": "Test message"}]
            result = await manager.complete(messages, ModelType.FAST)
            
            assert isinstance(result, LLMResponse)
            assert result.content == "Test response content"
            assert result.tokens_used == 150
            assert result.cached is False
            assert result.cost > 0  # Should have calculated cost
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    async def test_complete_no_router_raises_error(self, mock_settings, mock_router):
        """Test completion raises error when no router is configured."""
        mock_settings.DEBUG = False
        mock_settings.OPENAI_API_KEY = None
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.GOOGLE_API_KEY = None
        mock_settings.DEEPSEEK_API_KEY = None
        
        manager = LLMManager()
        manager.router = None
        
        with pytest.raises(RuntimeError) as exc_info:
            await manager.complete([{"role": "user", "content": "test"}])
        
        assert "LLM router not initialized" in str(exc_info.value)
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    async def test_complete_with_cache_hit(self, mock_settings, mock_router):
        """Test completion with cache hit."""
        mock_settings.DEBUG = False
        mock_settings.OPENAI_API_KEY = "test_key"
        mock_settings.ANTHROPIC_API_KEY = None
        mock_settings.GOOGLE_API_KEY = None
        mock_settings.DEEPSEEK_API_KEY = None
        mock_settings.LLM_DEFAULT_TEMPERATURE = 0.7
        
        manager = LLMManager()
        manager.redis_client = AsyncMock()
        
        # Setup cached response
        cached_response = LLMResponse(
            content="Cached content",
            model="gpt-4",
            provider="openai", 
            tokens_used=100,
            cost=0.03,
            processing_time=1.0,
            cached=True
        )
        
        with patch.object(manager, '_get_cached_response', return_value=cached_response):
            messages = [{"role": "user", "content": "Test"}]
            result = await manager.complete(messages)
            
            assert result == cached_response
            assert result.cached is True


class TestCostCalculation:
    """Test cost calculation functionality."""
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_calculate_cost_gpt4(self, mock_settings, mock_router):
        """Test cost calculation for GPT-4."""
        mock_settings.LLM_COST_TRACKING_ENABLED = True
        manager = LLMManager()
        
        # Mock usage object
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 1000  # 1K tokens
        mock_usage.completion_tokens = 500  # 0.5K tokens
        
        cost = manager._calculate_cost("gpt-4", mock_usage)
        
        # Expected: (1000/1000 * 0.03) + (500/1000 * 0.06) = 0.03 + 0.03 = 0.06
        expected_cost = 0.06
        assert abs(cost - expected_cost) < 0.001  # Allow for floating point precision
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_calculate_cost_unknown_model(self, mock_settings, mock_router):
        """Test cost calculation for unknown model uses defaults."""
        mock_settings.LLM_COST_TRACKING_ENABLED = True
        manager = LLMManager()
        
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 1000
        mock_usage.completion_tokens = 1000
        
        cost = manager._calculate_cost("unknown-model", mock_usage)
        
        # Should use default rates: input=0.001, output=0.002
        expected_cost = (1000/1000 * 0.001) + (1000/1000 * 0.002)  # 0.003
        assert abs(cost - expected_cost) < 0.001
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_calculate_cost_disabled(self, mock_settings, mock_router):
        """Test cost calculation when disabled."""
        mock_settings.LLM_COST_TRACKING_ENABLED = False
        manager = LLMManager()
        
        mock_usage = MagicMock()
        cost = manager._calculate_cost("gpt-4", mock_usage)
        
        assert cost == 0.0
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_calculate_cost_no_usage(self, mock_settings, mock_router):
        """Test cost calculation with no usage data."""
        mock_settings.LLM_COST_TRACKING_ENABLED = True
        manager = LLMManager()
        
        cost = manager._calculate_cost("gpt-4", None)
        
        assert cost == 0.0


class TestProviderExtraction:
    """Test provider name extraction from model names."""
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    def test_get_provider_openai(self, mock_router):
        """Test provider extraction for OpenAI models."""
        manager = LLMManager()
        
        assert manager._get_provider("gpt-4") == "openai"
        assert manager._get_provider("gpt-4-turbo") == "openai"
        assert manager._get_provider("gpt-3.5-turbo") == "openai"
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    def test_get_provider_anthropic(self, mock_router):
        """Test provider extraction for Anthropic models."""
        manager = LLMManager()
        
        assert manager._get_provider("claude-3-opus-20240229") == "anthropic"
        assert manager._get_provider("claude-3-sonnet-20240229") == "anthropic"
        assert manager._get_provider("claude-3-haiku-20240307") == "anthropic"
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    def test_get_provider_google(self, mock_router):
        """Test provider extraction for Google models."""
        manager = LLMManager()
        
        assert manager._get_provider("gemini-pro") == "google"
        assert manager._get_provider("gemini-pro-vision") == "google"
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    def test_get_provider_deepseek(self, mock_router):
        """Test provider extraction for DeepSeek models."""
        manager = LLMManager()
        
        assert manager._get_provider("deepseek-chat") == "deepseek"
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    def test_get_provider_unknown(self, mock_router):
        """Test provider extraction for unknown models."""
        manager = LLMManager()
        
        assert manager._get_provider("unknown-model") == "unknown"


class TestCostTracking:
    """Test cost tracking functionality."""
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_track_cost_enabled(self, mock_settings, mock_router):
        """Test cost tracking when enabled."""
        mock_settings.LLM_COST_TRACKING_ENABLED = True
        manager = LLMManager()
        
        manager._track_cost("gpt-4", "openai", 0.05)
        
        assert manager.total_cost == Decimal('0.05')
        assert manager.request_count == 1
        assert manager.cost_by_model["gpt-4"] == Decimal('0.05')
        assert manager.cost_by_provider["openai"] == Decimal('0.05')
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_track_cost_multiple_requests(self, mock_settings, mock_router):
        """Test cost tracking across multiple requests."""
        mock_settings.LLM_COST_TRACKING_ENABLED = True
        manager = LLMManager()
        
        manager._track_cost("gpt-4", "openai", 0.03)
        manager._track_cost("gpt-4", "openai", 0.02)
        manager._track_cost("claude-3-sonnet", "anthropic", 0.01)
        
        assert manager.total_cost == Decimal('0.06')
        assert manager.request_count == 3
        assert manager.cost_by_model["gpt-4"] == Decimal('0.05')
        assert manager.cost_by_model["claude-3-sonnet"] == Decimal('0.01')
        assert manager.cost_by_provider["openai"] == Decimal('0.05')
        assert manager.cost_by_provider["anthropic"] == Decimal('0.01')
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_track_cost_disabled(self, mock_settings, mock_router):
        """Test cost tracking when disabled."""
        mock_settings.LLM_COST_TRACKING_ENABLED = False
        manager = LLMManager()
        
        manager._track_cost("gpt-4", "openai", 0.05)
        
        assert manager.total_cost == Decimal('0.00')
        assert manager.request_count == 0
        assert manager.cost_by_model == {}
        assert manager.cost_by_provider == {}
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_track_cost_zero_cost(self, mock_settings, mock_router):
        """Test cost tracking with zero cost."""
        mock_settings.LLM_COST_TRACKING_ENABLED = True
        manager = LLMManager()
        
        manager._track_cost("gpt-4", "openai", 0.0)
        
        assert manager.total_cost == Decimal('0.00')
        assert manager.request_count == 0
        assert manager.cost_by_model == {}


class TestCostReport:
    """Test cost reporting functionality."""
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_get_cost_report_empty(self, mock_settings, mock_router):
        """Test cost report with no requests."""
        mock_settings.LLM_COST_TRACKING_ENABLED = True
        manager = LLMManager()
        
        report = manager.get_cost_report()
        
        assert report["total_cost"] == 0.0
        assert report["total_requests"] == 0
        assert report["average_cost_per_request"] == 0.0
        assert report["cost_by_model"] == {}
        assert report["cost_by_provider"] == {}
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    @patch('tubesensei.app.ai.llm_manager.settings')
    def test_get_cost_report_with_data(self, mock_settings, mock_router):
        """Test cost report with tracked data."""
        mock_settings.LLM_COST_TRACKING_ENABLED = True
        manager = LLMManager()
        
        manager._track_cost("gpt-4", "openai", 0.06)
        manager._track_cost("claude-3-sonnet", "anthropic", 0.03)
        
        report = manager.get_cost_report()
        
        assert report["total_cost"] == 0.09
        assert report["total_requests"] == 2
        assert report["average_cost_per_request"] == 0.045
        assert report["cost_by_model"]["gpt-4"] == 0.06
        assert report["cost_by_model"]["claude-3-sonnet"] == 0.03
        assert report["cost_by_provider"]["openai"] == 0.06
        assert report["cost_by_provider"]["anthropic"] == 0.03


class TestCleanupAndClose:
    """Test cleanup and resource management."""
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    async def test_close_with_redis_client(self, mock_router):
        """Test close method with Redis client."""
        manager = LLMManager()
        manager.redis_client = AsyncMock()
        
        await manager.close()
        
        manager.redis_client.close.assert_called_once()
    
    @patch('tubesensei.app.ai.llm_manager.Router')
    async def test_close_without_redis_client(self, mock_router):
        """Test close method without Redis client."""
        manager = LLMManager()
        manager.redis_client = None
        
        # Should not raise any exceptions
        await manager.close()


class TestModelCostConstants:
    """Test MODEL_COSTS constant values."""
    
    def test_model_costs_structure(self):
        """Test that MODEL_COSTS has expected structure."""
        for model, costs in MODEL_COSTS.items():
            assert isinstance(costs, dict)
            assert "input" in costs
            assert "output" in costs
            assert isinstance(costs["input"], (int, float))
            assert isinstance(costs["output"], (int, float))
            assert costs["input"] > 0
            assert costs["output"] > 0
    
    def test_model_costs_includes_major_models(self):
        """Test that MODEL_COSTS includes major model families."""
        expected_models = [
            "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo",
            "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
            "gemini-pro", "deepseek-chat"
        ]
        
        for model in expected_models:
            assert model in MODEL_COSTS