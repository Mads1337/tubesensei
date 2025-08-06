import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List, Any
from enum import Enum
import json
import logging
from pathlib import Path

from ..utils.exceptions import QuotaExceededError

logger = logging.getLogger(__name__)


class YouTubeAPIOperation(str, Enum):
    """YouTube API v3 operations and their quota costs"""
    
    # Search operations
    SEARCH_LIST = "search.list"  # 100 units
    
    # Channel operations
    CHANNELS_LIST = "channels.list"  # 1 unit
    
    # Video operations
    VIDEOS_LIST = "videos.list"  # 1 unit
    
    # Playlist operations
    PLAYLISTS_LIST = "playlists.list"  # 1 unit
    PLAYLIST_ITEMS_LIST = "playlistItems.list"  # 1 unit
    
    # Comments operations
    COMMENTS_LIST = "comments.list"  # 1 unit
    COMMENT_THREADS_LIST = "commentThreads.list"  # 1 unit
    
    # Captions operations
    CAPTIONS_LIST = "captions.list"  # 50 units
    CAPTIONS_DOWNLOAD = "captions.download"  # 200 units
    
    # Activities operations
    ACTIVITIES_LIST = "activities.list"  # 1 unit
    
    # Subscriptions operations
    SUBSCRIPTIONS_LIST = "subscriptions.list"  # 1 unit


class QuotaCost:
    """Quota costs for YouTube API v3 operations"""
    
    COSTS = {
        YouTubeAPIOperation.SEARCH_LIST: 100,
        YouTubeAPIOperation.CHANNELS_LIST: 1,
        YouTubeAPIOperation.VIDEOS_LIST: 1,
        YouTubeAPIOperation.PLAYLISTS_LIST: 1,
        YouTubeAPIOperation.PLAYLIST_ITEMS_LIST: 1,
        YouTubeAPIOperation.COMMENTS_LIST: 1,
        YouTubeAPIOperation.COMMENT_THREADS_LIST: 1,
        YouTubeAPIOperation.CAPTIONS_LIST: 50,
        YouTubeAPIOperation.CAPTIONS_DOWNLOAD: 200,
        YouTubeAPIOperation.ACTIVITIES_LIST: 1,
        YouTubeAPIOperation.SUBSCRIPTIONS_LIST: 1,
    }
    
    @classmethod
    def get_cost(cls, operation: YouTubeAPIOperation) -> int:
        """Get quota cost for an operation"""
        return cls.COSTS.get(operation, 1)


class QuotaManager:
    """
    Manages YouTube API quota usage and tracking.
    YouTube API v3 has a default quota of 10,000 units per day.
    """
    
    def __init__(
        self,
        daily_quota: int = 10000,
        warning_threshold: float = 0.8,
        storage_path: Optional[Path] = None
    ):
        self.daily_quota = daily_quota
        self.warning_threshold = warning_threshold
        self.storage_path = storage_path or Path("/tmp/youtube_quota.json")
        
        # Current usage tracking
        self.current_usage = 0
        self.usage_history: List[Dict[str, Any]] = []
        self.reset_time: Optional[datetime] = None
        
        # Operation statistics
        self.operation_counts: Dict[str, int] = {}
        self.operation_costs: Dict[str, int] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Load existing quota data
        asyncio.create_task(self._load_quota_data())
    
    async def _load_quota_data(self):
        """Load quota data from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                    # Check if data is from today
                    saved_date = datetime.fromisoformat(data.get('date', ''))
                    if saved_date.date() == datetime.now(timezone.utc).date():
                        self.current_usage = data.get('usage', 0)
                        self.usage_history = data.get('history', [])
                        self.operation_counts = data.get('operation_counts', {})
                        self.operation_costs = data.get('operation_costs', {})
                        logger.info(f"Loaded quota data: {self.current_usage}/{self.daily_quota} units used")
                    else:
                        await self.reset_quota()
            except Exception as e:
                logger.error(f"Error loading quota data: {e}")
                await self.reset_quota()
        else:
            await self.reset_quota()
    
    async def _save_quota_data(self):
        """Save quota data to storage"""
        try:
            data = {
                'date': datetime.now(timezone.utc).isoformat(),
                'usage': self.current_usage,
                'history': self.usage_history[-1000:],  # Keep last 1000 operations
                'operation_counts': self.operation_counts,
                'operation_costs': self.operation_costs,
                'reset_time': self.reset_time.isoformat() if self.reset_time else None
            }
            
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving quota data: {e}")
    
    async def reset_quota(self):
        """Reset quota for a new day"""
        async with self._lock:
            self.current_usage = 0
            self.usage_history = []
            self.operation_counts = {}
            self.operation_costs = {}
            self.reset_time = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            
            logger.info(f"Quota reset. Next reset at {self.reset_time}")
            await self._save_quota_data()
    
    async def check_and_reset_if_needed(self):
        """Check if quota needs to be reset (new day)"""
        now = datetime.now(timezone.utc)
        
        if self.reset_time is None:
            await self.reset_quota()
        elif now >= self.reset_time:
            await self.reset_quota()
    
    async def reserve_quota(
        self,
        operation: YouTubeAPIOperation,
        units: Optional[int] = None
    ) -> bool:
        """
        Reserve quota for an operation.
        
        Args:
            operation: The API operation to perform
            units: Override units (for batch operations)
            
        Returns:
            True if quota was reserved, False otherwise
            
        Raises:
            QuotaExceededError: If quota would be exceeded
        """
        await self.check_and_reset_if_needed()
        
        async with self._lock:
            cost = units or QuotaCost.get_cost(operation)
            
            # Check if we have enough quota
            if self.current_usage + cost > self.daily_quota:
                raise QuotaExceededError(
                    message=f"Insufficient quota for {operation.value}",
                    quota_used=self.current_usage,
                    quota_limit=self.daily_quota
                )
            
            # Reserve the quota
            self.current_usage += cost
            
            # Track operation
            operation_name = operation.value
            self.operation_counts[operation_name] = self.operation_counts.get(operation_name, 0) + 1
            self.operation_costs[operation_name] = self.operation_costs.get(operation_name, 0) + cost
            
            # Add to history
            self.usage_history.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'operation': operation_name,
                'cost': cost,
                'total_usage': self.current_usage
            })
            
            # Check warning threshold
            usage_percentage = self.current_usage / self.daily_quota
            if usage_percentage >= self.warning_threshold:
                logger.warning(
                    f"Quota usage at {usage_percentage:.1%} "
                    f"({self.current_usage}/{self.daily_quota} units)"
                )
            
            # Save data
            await self._save_quota_data()
            
            logger.debug(
                f"Reserved {cost} units for {operation_name}. "
                f"Total usage: {self.current_usage}/{self.daily_quota}"
            )
            
            return True
    
    async def release_quota(self, operation: YouTubeAPIOperation, units: Optional[int] = None):
        """
        Release previously reserved quota (e.g., if operation failed).
        
        Args:
            operation: The API operation that failed
            units: Override units to release
        """
        async with self._lock:
            cost = units or QuotaCost.get_cost(operation)
            self.current_usage = max(0, self.current_usage - cost)
            
            # Update operation counts
            operation_name = operation.value
            if operation_name in self.operation_counts:
                self.operation_counts[operation_name] = max(
                    0, self.operation_counts[operation_name] - 1
                )
            if operation_name in self.operation_costs:
                self.operation_costs[operation_name] = max(
                    0, self.operation_costs[operation_name] - cost
                )
            
            await self._save_quota_data()
            
            logger.debug(
                f"Released {cost} units for {operation_name}. "
                f"Total usage: {self.current_usage}/{self.daily_quota}"
            )
    
    def get_remaining_quota(self) -> int:
        """Get remaining quota units"""
        return max(0, self.daily_quota - self.current_usage)
    
    def get_usage_percentage(self) -> float:
        """Get quota usage as percentage"""
        return (self.current_usage / self.daily_quota) * 100 if self.daily_quota > 0 else 100
    
    def can_perform_operation(self, operation: YouTubeAPIOperation, units: Optional[int] = None) -> bool:
        """Check if an operation can be performed with current quota"""
        cost = units or QuotaCost.get_cost(operation)
        return self.current_usage + cost <= self.daily_quota
    
    async def estimate_operations_remaining(self) -> Dict[str, int]:
        """Estimate how many of each operation type can still be performed"""
        remaining = self.get_remaining_quota()
        estimates = {}
        
        for operation in YouTubeAPIOperation:
            cost = QuotaCost.get_cost(operation)
            estimates[operation.value] = remaining // cost if cost > 0 else 0
        
        return estimates
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get detailed usage statistics"""
        await self.check_and_reset_if_needed()
        
        return {
            'current_usage': self.current_usage,
            'daily_quota': self.daily_quota,
            'remaining_quota': self.get_remaining_quota(),
            'usage_percentage': self.get_usage_percentage(),
            'reset_time': self.reset_time.isoformat() if self.reset_time else None,
            'operation_counts': self.operation_counts,
            'operation_costs': self.operation_costs,
            'warning_threshold': self.warning_threshold,
            'is_warning': self.get_usage_percentage() >= self.warning_threshold * 100,
            'recent_operations': self.usage_history[-10:] if self.usage_history else []
        }
    
    async def optimize_batch_size(self, operation: YouTubeAPIOperation, total_items: int) -> int:
        """
        Calculate optimal batch size based on remaining quota.
        
        Args:
            operation: The operation to perform
            total_items: Total number of items to process
            
        Returns:
            Optimal batch size
        """
        remaining_quota = self.get_remaining_quota()
        operation_cost = QuotaCost.get_cost(operation)
        
        # Calculate how many operations we can perform
        max_operations = remaining_quota // operation_cost if operation_cost > 0 else total_items
        
        # YouTube API typically allows 50 items per request for most operations
        api_batch_limit = 50
        
        # Return the minimum of what we can afford and what's needed
        return min(max_operations * api_batch_limit, total_items, api_batch_limit)
    
    def __str__(self) -> str:
        """String representation of quota status"""
        return (
            f"QuotaManager("
            f"usage={self.current_usage}/{self.daily_quota}, "
            f"remaining={self.get_remaining_quota()}, "
            f"percentage={self.get_usage_percentage():.1f}%)"
        )