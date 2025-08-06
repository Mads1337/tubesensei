import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import json

from app.integrations.quota_manager import (
    QuotaManager,
    QuotaCost,
    YouTubeAPIOperation
)
from app.utils.exceptions import QuotaExceededError


class TestQuotaCost:
    """Test QuotaCost functionality"""
    
    def test_quota_costs(self):
        """Test that quota costs are correctly defined"""
        # High cost operations
        assert QuotaCost.get_cost(YouTubeAPIOperation.SEARCH_LIST) == 100
        assert QuotaCost.get_cost(YouTubeAPIOperation.CAPTIONS_DOWNLOAD) == 200
        assert QuotaCost.get_cost(YouTubeAPIOperation.CAPTIONS_LIST) == 50
        
        # Low cost operations
        assert QuotaCost.get_cost(YouTubeAPIOperation.CHANNELS_LIST) == 1
        assert QuotaCost.get_cost(YouTubeAPIOperation.VIDEOS_LIST) == 1
        assert QuotaCost.get_cost(YouTubeAPIOperation.PLAYLIST_ITEMS_LIST) == 1
        assert QuotaCost.get_cost(YouTubeAPIOperation.COMMENTS_LIST) == 1


class TestQuotaManager:
    """Test QuotaManager functionality"""
    
    @pytest.fixture
    async def temp_quota_file(self):
        """Create a temporary file for quota storage"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        yield temp_path
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()
    
    @pytest.fixture
    async def quota_manager(self, temp_quota_file):
        """Create a QuotaManager instance with temporary storage"""
        manager = QuotaManager(
            daily_quota=1000,
            warning_threshold=0.8,
            storage_path=temp_quota_file
        )
        await manager.reset_quota()
        return manager
    
    @pytest.mark.asyncio
    async def test_quota_reservation(self, quota_manager):
        """Test basic quota reservation"""
        # Reserve quota for a simple operation
        result = await quota_manager.reserve_quota(YouTubeAPIOperation.VIDEOS_LIST)
        assert result is True
        assert quota_manager.current_usage == 1
        assert quota_manager.get_remaining_quota() == 999
    
    @pytest.mark.asyncio
    async def test_quota_reservation_batch(self, quota_manager):
        """Test quota reservation with custom units"""
        # Reserve quota for batch operation
        result = await quota_manager.reserve_quota(
            YouTubeAPIOperation.VIDEOS_LIST,
            units=50
        )
        assert result is True
        assert quota_manager.current_usage == 50
        assert quota_manager.get_remaining_quota() == 950
    
    @pytest.mark.asyncio
    async def test_quota_exceeded_error(self, quota_manager):
        """Test quota exceeded error"""
        # Use up most of the quota
        await quota_manager.reserve_quota(
            YouTubeAPIOperation.SEARCH_LIST,
            units=900
        )
        
        # This should exceed quota
        with pytest.raises(QuotaExceededError) as exc_info:
            await quota_manager.reserve_quota(
                YouTubeAPIOperation.CAPTIONS_DOWNLOAD  # 200 units
            )
        
        assert exc_info.value.details['quota_used'] == 900
        assert exc_info.value.details['quota_limit'] == 1000
    
    @pytest.mark.asyncio
    async def test_quota_release(self, quota_manager):
        """Test quota release functionality"""
        # Reserve quota
        await quota_manager.reserve_quota(YouTubeAPIOperation.SEARCH_LIST)
        assert quota_manager.current_usage == 100
        
        # Release quota
        await quota_manager.release_quota(YouTubeAPIOperation.SEARCH_LIST)
        assert quota_manager.current_usage == 0
    
    @pytest.mark.asyncio
    async def test_quota_persistence(self, temp_quota_file):
        """Test quota data persistence"""
        # Create manager and use some quota
        manager1 = QuotaManager(
            daily_quota=1000,
            storage_path=temp_quota_file
        )
        await manager1.reset_quota()
        await manager1.reserve_quota(YouTubeAPIOperation.SEARCH_LIST)
        await manager1.reserve_quota(YouTubeAPIOperation.VIDEOS_LIST, units=50)
        
        # Create new manager with same storage
        manager2 = QuotaManager(
            daily_quota=1000,
            storage_path=temp_quota_file
        )
        await manager2._load_quota_data()
        
        # Should load previous usage
        assert manager2.current_usage == 150
        assert manager2.get_remaining_quota() == 850
    
    @pytest.mark.asyncio
    async def test_quota_reset(self, quota_manager):
        """Test quota reset functionality"""
        # Use some quota
        await quota_manager.reserve_quota(YouTubeAPIOperation.SEARCH_LIST)
        assert quota_manager.current_usage == 100
        
        # Reset quota
        await quota_manager.reset_quota()
        assert quota_manager.current_usage == 0
        assert quota_manager.usage_history == []
        assert quota_manager.operation_counts == {}
        assert quota_manager.reset_time is not None
    
    @pytest.mark.asyncio
    async def test_operation_tracking(self, quota_manager):
        """Test operation count and cost tracking"""
        # Perform various operations
        await quota_manager.reserve_quota(YouTubeAPIOperation.VIDEOS_LIST)
        await quota_manager.reserve_quota(YouTubeAPIOperation.VIDEOS_LIST)
        await quota_manager.reserve_quota(YouTubeAPIOperation.SEARCH_LIST)
        
        # Check operation counts
        assert quota_manager.operation_counts['videos.list'] == 2
        assert quota_manager.operation_counts['search.list'] == 1
        
        # Check operation costs
        assert quota_manager.operation_costs['videos.list'] == 2
        assert quota_manager.operation_costs['search.list'] == 100
    
    @pytest.mark.asyncio
    async def test_usage_history(self, quota_manager):
        """Test usage history tracking"""
        # Perform operations
        await quota_manager.reserve_quota(YouTubeAPIOperation.VIDEOS_LIST)
        await quota_manager.reserve_quota(YouTubeAPIOperation.CHANNELS_LIST)
        
        # Check history
        assert len(quota_manager.usage_history) == 2
        assert quota_manager.usage_history[0]['operation'] == 'videos.list'
        assert quota_manager.usage_history[0]['cost'] == 1
        assert quota_manager.usage_history[1]['operation'] == 'channels.list'
        assert quota_manager.usage_history[1]['cost'] == 1
    
    @pytest.mark.asyncio
    async def test_warning_threshold(self, quota_manager):
        """Test warning threshold detection"""
        # Use 80% of quota (warning threshold)
        await quota_manager.reserve_quota(
            YouTubeAPIOperation.SEARCH_LIST,
            units=800
        )
        
        # Check usage percentage
        assert quota_manager.get_usage_percentage() == 80.0
        
        # Get stats should show warning
        stats = await quota_manager.get_usage_stats()
        assert stats['is_warning'] is True
    
    @pytest.mark.asyncio
    async def test_can_perform_operation(self, quota_manager):
        """Test checking if operation can be performed"""
        # Initially should be able to perform any operation
        assert quota_manager.can_perform_operation(YouTubeAPIOperation.SEARCH_LIST) is True
        assert quota_manager.can_perform_operation(YouTubeAPIOperation.CAPTIONS_DOWNLOAD) is True
        
        # Use most of quota
        await quota_manager.reserve_quota(
            YouTubeAPIOperation.SEARCH_LIST,
            units=900
        )
        
        # Should not be able to perform expensive operations
        assert quota_manager.can_perform_operation(YouTubeAPIOperation.CAPTIONS_DOWNLOAD) is False
        assert quota_manager.can_perform_operation(YouTubeAPIOperation.SEARCH_LIST) is False
        
        # But can still perform cheap operations
        assert quota_manager.can_perform_operation(YouTubeAPIOperation.VIDEOS_LIST) is True
    
    @pytest.mark.asyncio
    async def test_estimate_operations_remaining(self, quota_manager):
        """Test estimating remaining operations"""
        # Use half the quota
        await quota_manager.reserve_quota(
            YouTubeAPIOperation.SEARCH_LIST,
            units=500
        )
        
        estimates = await quota_manager.estimate_operations_remaining()
        
        # Should be able to do 5 more searches (500 remaining / 100 per search)
        assert estimates['search.list'] == 5
        
        # Should be able to do 500 simple operations
        assert estimates['videos.list'] == 500
        assert estimates['channels.list'] == 500
        
        # Should be able to do 2 caption downloads (500 / 200)
        assert estimates['captions.download'] == 2
    
    @pytest.mark.asyncio
    async def test_optimize_batch_size(self, quota_manager):
        """Test batch size optimization"""
        # With full quota, should return API limit
        batch_size = await quota_manager.optimize_batch_size(
            YouTubeAPIOperation.VIDEOS_LIST,
            total_items=1000
        )
        assert batch_size == 50  # API limit
        
        # Use most of quota
        await quota_manager.reserve_quota(
            YouTubeAPIOperation.SEARCH_LIST,
            units=990
        )
        
        # Should limit based on remaining quota
        batch_size = await quota_manager.optimize_batch_size(
            YouTubeAPIOperation.VIDEOS_LIST,
            total_items=1000
        )
        assert batch_size == 50  # Still within limits (10 operations * 50 items)
        
        # For expensive operations
        batch_size = await quota_manager.optimize_batch_size(
            YouTubeAPIOperation.SEARCH_LIST,
            total_items=1000
        )
        assert batch_size == 0  # Can't afford any searches
    
    @pytest.mark.asyncio
    async def test_get_usage_stats(self, quota_manager):
        """Test getting comprehensive usage statistics"""
        # Perform some operations
        await quota_manager.reserve_quota(YouTubeAPIOperation.VIDEOS_LIST)
        await quota_manager.reserve_quota(YouTubeAPIOperation.SEARCH_LIST)
        
        stats = await quota_manager.get_usage_stats()
        
        assert stats['current_usage'] == 101
        assert stats['daily_quota'] == 1000
        assert stats['remaining_quota'] == 899
        assert stats['usage_percentage'] == 10.1
        assert stats['is_warning'] is False
        assert len(stats['recent_operations']) == 2
        assert 'videos.list' in stats['operation_counts']
        assert 'search.list' in stats['operation_counts']