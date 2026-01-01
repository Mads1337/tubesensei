"""
LLM Manager for TubeSensei - Phase 2A Core Module

This module provides a unified interface for managing multiple LLM providers
through LiteLLM, with automatic fallback, cost tracking, and model selection.
"""

import asyncio
import time
import logging
import json
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
from decimal import Decimal

try:
    import redis.asyncio as aioredis
except ImportError:
    try:
        import aioredis
    except ImportError:
        aioredis = None

from litellm import Router, completion
import litellm

from app.config import settings
from app.ai.retry_strategy import RetryStrategy


logger = logging.getLogger(__name__)

# Cost tracking per 1M tokens (current pricing - Dec 2025)
MODEL_COSTS = {
    # DeepSeek Models (primary - very cost effective)
    "deepseek-chat": {"input": 0.14, "output": 0.28},      # V3.2 non-thinking mode
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},  # V3.2 thinking/reasoning mode

    # OpenAI Models (fallback)
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "o3-mini": {"input": 1.10, "output": 4.40},

    # Anthropic Models (fallback)
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},

    # Google Gemini Models (fallback)
    "gemini-2.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
}


class ModelType(Enum):
    """Model types for different use cases."""
    FAST = "fast"           # Quick, cheap operations
    BALANCED = "balanced"   # Moderate quality/cost
    QUALITY = "quality"     # High-quality responses


@dataclass
class LLMResponse:
    """Response from an LLM completion request."""
    content: str
    model: str
    provider: str
    tokens_used: int
    cost: float
    processing_time: float
    cached: bool = False


class LLMManager:
    """
    Unified LLM manager with multi-provider support, fallback, and cost tracking.
    """
    
    # Model configuration - DeepSeek primary, others as fallback (Dec 2025)
    # To use deepseek-reasoner for quality tasks, change the first entry in QUALITY tier
    MODEL_CONFIG: Dict[ModelType, List[str]] = {
        ModelType.FAST: [
            "deepseek-chat",             # Primary - fast and cheap
            "gemini-2.5-flash",          # Fast fallback
            "gpt-4.1-mini",              # OpenAI fallback
            "claude-haiku-4-5",          # Anthropic fallback
        ],
        ModelType.BALANCED: [
            "deepseek-chat",             # Primary - can switch to deepseek-reasoner if needed
            "gemini-2.5-flash",          # Google fallback
            "gpt-4.1-mini",              # OpenAI fallback
            "claude-sonnet-4-5-20250929", # Anthropic fallback
        ],
        ModelType.QUALITY: [
            "deepseek-chat",             # Primary - switch to "deepseek-reasoner" for complex reasoning
            "gpt-4.1",                   # OpenAI quality fallback
            "claude-sonnet-4-5-20250929", # Anthropic quality fallback
            "gemini-2.5-pro",            # Google quality fallback
        ]
    }
    
    def __init__(self):
        """Initialize the LLM Manager."""
        self.router: Optional[Router] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.total_cost = Decimal('0.00')
        self.request_count = 0
        self.cost_by_model: Dict[str, Decimal] = {}
        self.cost_by_provider: Dict[str, Decimal] = {}
        
        # Initialize retry strategy
        self.retry_strategy = RetryStrategy()
        
        # Configure litellm logging
        litellm.set_verbose = settings.DEBUG
        
        self._setup_router()
    
    def _setup_router(self) -> None:
        """Configure LiteLLM router with multiple providers."""
        try:
            # Build model list for router
            model_list = []
            
            # DeepSeek models (primary - Dec 2025)
            if settings.DEEPSEEK_API_KEY:
                deepseek_models = ["deepseek-chat", "deepseek-reasoner"]
                for model in deepseek_models:
                    model_list.append({
                        "model_name": model,
                        "litellm_params": {
                            "model": f"deepseek/{model}",
                            "api_key": settings.DEEPSEEK_API_KEY
                        }
                    })

            # OpenAI models (fallback)
            if settings.OPENAI_API_KEY:
                openai_models = ["gpt-4.1", "gpt-4.1-mini", "o3-mini"]
                for model in openai_models:
                    model_list.append({
                        "model_name": model,
                        "litellm_params": {
                            "model": f"openai/{model}",
                            "api_key": settings.OPENAI_API_KEY
                        }
                    })

            # Anthropic models (fallback)
            if settings.ANTHROPIC_API_KEY:
                anthropic_models = ["claude-sonnet-4-5-20250929", "claude-haiku-4-5"]
                for model in anthropic_models:
                    model_list.append({
                        "model_name": model,
                        "litellm_params": {
                            "model": f"anthropic/{model}",
                            "api_key": settings.ANTHROPIC_API_KEY
                        }
                    })

            # Google Gemini models (fallback)
            if settings.GOOGLE_API_KEY:
                google_models = ["gemini-2.5-pro", "gemini-2.5-flash"]
                for model in google_models:
                    model_list.append({
                        "model_name": model,
                        "litellm_params": {
                            "model": f"gemini/{model}",
                            "api_key": settings.GOOGLE_API_KEY
                        }
                    })
            
            if not model_list:
                logger.warning("No LLM API keys configured")
                return
            
            # Initialize router with DeepSeek-primary fallbacks (Dec 2025)
            self.router = Router(
                model_list=model_list,
                fallbacks=[
                    # DeepSeek primary fallbacks
                    {"deepseek-chat": ["gpt-4.1-mini", "gemini-2.5-flash"]},
                    {"deepseek-reasoner": ["deepseek-chat", "gpt-4.1"]},

                    # OpenAI fallbacks
                    {"gpt-4.1": ["deepseek-chat", "claude-sonnet-4-5-20250929"]},
                    {"gpt-4.1-mini": ["deepseek-chat", "gemini-2.5-flash"]},
                    {"o3-mini": ["deepseek-reasoner", "gpt-4.1"]},

                    # Anthropic fallbacks
                    {"claude-sonnet-4-5-20250929": ["deepseek-chat", "gpt-4.1"]},
                    {"claude-haiku-4-5": ["deepseek-chat", "gpt-4.1-mini"]},

                    # Google fallbacks
                    {"gemini-2.5-pro": ["deepseek-chat", "gpt-4.1"]},
                    {"gemini-2.5-flash": ["deepseek-chat", "gpt-4.1-mini"]},
                ]
            )
            
            logger.info(f"LLM router configured with {len(model_list)} models")
            
        except Exception as e:
            logger.error(f"Failed to setup LLM router: {e}")
            raise
    
    async def initialize(self) -> None:
        """Initialize Redis connection for caching."""
        if aioredis is None:
            logger.warning("Redis not available - caching disabled")
            return
            
        try:
            self.redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis client initialized for LLM caching")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            # Don't raise - caching is optional
    
    def _generate_cache_key(
        self, 
        messages: List[Dict[str, str]], 
        model_type: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate unique cache key using hashlib for prompt+model+system_prompt.
        
        Args:
            messages: List of message dicts
            model_type: Model type string
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            SHA256 hash as cache key
        """
        # Create deterministic string from all parameters
        cache_data = {
            "messages": messages,
            "model_type": model_type,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "kwargs": sorted(kwargs.items())  # Sort for consistency
        }
        
        # Convert to JSON string for consistent hashing
        cache_string = json.dumps(cache_data, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA256 hash
        hash_obj = hashlib.sha256(cache_string.encode('utf-8'))
        return f"llm_cache:{hash_obj.hexdigest()}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """
        Retrieve cached response from Redis.
        
        Args:
            cache_key: Redis cache key
            
        Returns:
            Cached LLMResponse or None if not found
        """
        if not self.redis_client:
            return None
            
        try:
            cached_data = await self.redis_client.get(cache_key)
            if not cached_data:
                return None
            
            # Deserialize cached response
            response_data = json.loads(cached_data)
            
            # Reconstruct LLMResponse with cached=True
            return LLMResponse(
                content=response_data['content'],
                model=response_data['model'],
                provider=response_data['provider'],
                tokens_used=response_data['tokens_used'],
                cost=response_data['cost'],
                processing_time=response_data['processing_time'],
                cached=True
            )
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached response: {e}")
            return None
    
    async def _cache_response(self, cache_key: str, response: LLMResponse) -> None:
        """
        Store response in Redis with TTL.
        
        Args:
            cache_key: Redis cache key
            response: LLMResponse to cache
        """
        if not self.redis_client:
            return
            
        try:
            # Serialize response data (excluding cached flag)
            response_data = {
                'content': response.content,
                'model': response.model,
                'provider': response.provider,
                'tokens_used': response.tokens_used,
                'cost': response.cost,
                'processing_time': response.processing_time
            }
            
            # Store in Redis with TTL
            await self.redis_client.setex(
                cache_key,
                settings.LLM_CACHE_TTL,
                json.dumps(response_data)
            )
            
            logger.debug(f"Cached LLM response with key: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
            # Don't raise - caching failures shouldn't break the flow
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model_type: ModelType = ModelType.BALANCED,
        temperature: float = None,
        max_tokens: int = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Get LLM completion with automatic model selection and fallback.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_type: Type of model to use (fast/balanced/quality)
            temperature: Generation temperature (default from config)
            max_tokens: Maximum tokens to generate
            use_cache: Whether to use Redis caching (default: True)
            **kwargs: Additional LiteLLM parameters
            
        Returns:
            LLMResponse with completion data
        """
        if not self.router:
            raise RuntimeError("LLM router not initialized - no API keys configured")
        
        # Set defaults
        if temperature is None:
            temperature = settings.LLM_DEFAULT_TEMPERATURE
        
        # Check cache first if enabled
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(
                messages=messages,
                model_type=model_type.value,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            cached_response = await self._get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Cache hit for LLM request: {cached_response.model}")
                return cached_response
        
        # Get model list for the requested type
        available_models = self.MODEL_CONFIG.get(model_type, [])
        if not available_models:
            raise ValueError(f"No models configured for type: {model_type}")
        
        # Filter out models with open circuit breakers
        providers = [self._get_provider(model) for model in available_models]
        available_providers = self.retry_strategy.get_available_providers(providers)
        
        # Filter models by available providers
        filtered_models = [
            model for model in available_models 
            if self._get_provider(model) in available_providers
        ]
        
        if not filtered_models:
            # If all circuits are open, try one model anyway (circuit may have timeout expired)
            logger.warning("All provider circuits are open, trying first model anyway")
            filtered_models = [available_models[0]]
        
        # Prepare completion parameters
        completion_params = {
            "messages": messages,
            "temperature": temperature,
            **kwargs
        }
        
        if max_tokens:
            completion_params["max_tokens"] = max_tokens
        
        start_time = time.time()
        last_exception = None
        
        # Try models in order with retry strategy
        for model in filtered_models:
            provider = self._get_provider(model)
            
            # Skip if circuit is still open for this provider
            if self.retry_strategy.is_circuit_open(provider):
                logger.warning(f"Skipping model {model} - circuit breaker open for provider {provider}")
                continue
            
            try:
                logger.debug(f"Attempting completion with model: {model}")

                # Create retry context for this model/provider
                retry_context = self.retry_strategy.create_retry_context(provider=provider, model=model)

                # Define the operation to retry using router's acompletion
                async def completion_operation():
                    return await self.router.acompletion(
                        model=model,
                        **completion_params
                    )

                # Execute with retry strategy
                response = await self.retry_strategy.execute_with_retry(
                    completion_operation,
                    retry_context
                )
                
                processing_time = time.time() - start_time
                
                # Extract response data
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if response.usage else 0
                
                # Calculate cost
                cost = self._calculate_cost(model, response.usage) if response.usage else 0.0
                
                # Track cost
                self._track_cost(model, provider, cost)
                
                llm_response = LLMResponse(
                    content=content,
                    model=model,
                    provider=provider,
                    tokens_used=tokens_used,
                    cost=cost,
                    processing_time=processing_time,
                    cached=False
                )
                
                logger.info(
                    f"LLM completion successful: {model} | "
                    f"tokens={tokens_used} | cost=${cost:.4f} | "
                    f"time={processing_time:.2f}s"
                )
                
                # Cache successful response if caching is enabled
                if use_cache and cache_key:
                    await self._cache_response(cache_key, llm_response)
                
                return llm_response
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Model {model} failed after retries: {e}")
                continue
        
        # All models failed
        raise RuntimeError(
            f"All models failed for type {model_type}. "
            f"Last error: {last_exception}"
        )

    async def generate(
        self,
        prompt: str,
        model_type: ModelType = ModelType.BALANCED,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simple text generation interface that wraps complete().

        Args:
            prompt: The prompt text to send
            model_type: Type of model to use (fast/balanced/quality)
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            use_cache: Whether to use Redis caching
            **kwargs: Additional LiteLLM parameters

        Returns:
            Dict with 'content', 'usage', and 'cost' keys
        """
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call complete()
        response = await self.complete(
            messages=messages,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            use_cache=use_cache,
            **kwargs
        )

        # Return dict format expected by agents
        return {
            "content": response.content,
            "usage": {
                "total_tokens": response.tokens_used,
            },
            "cost": response.cost,
            "model": response.model,
            "provider": response.provider,
            "cached": response.cached,
        }

    def _calculate_cost(self, model: str, usage: Any) -> float:
        """Calculate cost for the completion."""
        if not usage or not settings.LLM_COST_TRACKING_ENABLED:
            return 0.0
        
        try:
            model_cost = MODEL_COSTS.get(model, {"input": 0.001, "output": 0.002})
            
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)
            
            input_cost = (input_tokens / 1000) * model_cost["input"]
            output_cost = (output_tokens / 1000) * model_cost["output"]
            
            return round(input_cost + output_cost, 6)
            
        except Exception as e:
            logger.warning(f"Failed to calculate cost for {model}: {e}")
            return 0.0
    
    def _get_provider(self, model: str) -> str:
        """Extract provider name from model."""
        if model.startswith("gpt-") or model.startswith("o3") or model.startswith("o4"):
            return "openai"
        elif model.startswith("claude-"):
            return "anthropic"
        elif model.startswith("gemini-"):
            return "google"
        elif model.startswith("deepseek"):
            return "deepseek"
        elif model.startswith("qwen"):
            return "qwen"
        else:
            return "unknown"
    
    def _track_cost(self, model: str, provider: str, cost: float) -> None:
        """Track cumulative costs by model and provider."""
        if not settings.LLM_COST_TRACKING_ENABLED or cost == 0:
            return
        
        cost_decimal = Decimal(str(cost))
        
        # Track total cost
        self.total_cost += cost_decimal
        self.request_count += 1
        
        # Track by model
        if model not in self.cost_by_model:
            self.cost_by_model[model] = Decimal('0.00')
        self.cost_by_model[model] += cost_decimal
        
        # Track by provider
        if provider not in self.cost_by_provider:
            self.cost_by_provider[provider] = Decimal('0.00')
        self.cost_by_provider[provider] += cost_decimal
    
    def get_cost_report(self) -> Dict[str, Any]:
        """Get comprehensive cost report."""
        return {
            "total_cost": float(self.total_cost),
            "total_requests": self.request_count,
            "average_cost_per_request": float(
                self.total_cost / self.request_count if self.request_count > 0 else 0
            ),
            "cost_by_model": {
                model: float(cost) for model, cost in self.cost_by_model.items()
            },
            "cost_by_provider": {
                provider: float(cost) for provider, cost in self.cost_by_provider.items()
            },
            "circuit_breaker_status": self.retry_strategy.get_circuit_breaker_status()
        }
    
    def get_retry_strategy_status(self) -> Dict[str, Any]:
        """Get retry strategy and circuit breaker status."""
        return {
            "circuit_breakers": self.retry_strategy.get_circuit_breaker_status(),
            "config": {
                "max_retries": self.retry_strategy.max_retries,
                "initial_delay": self.retry_strategy.initial_delay,
                "max_delay": self.retry_strategy.max_delay,
                "backoff_multiplier": self.retry_strategy.backoff_multiplier,
                "circuit_breaker_threshold": self.retry_strategy.circuit_breaker_threshold,
                "circuit_breaker_timeout": self.retry_strategy.circuit_breaker_timeout
            }
        }
    
    def reset_circuit_breaker(self, provider: str) -> None:
        """Reset circuit breaker for a specific provider."""
        self.retry_strategy.reset_circuit_breaker(provider)
    
    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        self.retry_strategy.reset_all_circuit_breakers()
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis client closed")