import pytest
from datetime import datetime, timedelta, timezone
from app.models.filters import VideoFilters, ChannelFilters, VideoType


class TestVideoFilters:
    """Test VideoFilters functionality"""
    
    def test_default_filters(self):
        """Test default filter values"""
        filters = VideoFilters()
        
        assert filters.min_duration_seconds == 60
        assert filters.max_duration_seconds == 7200
        assert filters.exclude_shorts is True
        assert filters.exclude_live is False
        assert filters.exclude_premieres is True
        assert VideoType.REGULAR in filters.include_types
    
    def test_duration_validation(self):
        """Test duration range validation"""
        # Valid range
        filters = VideoFilters(
            min_duration_seconds=60,
            max_duration_seconds=3600
        )
        assert filters.min_duration_seconds == 60
        assert filters.max_duration_seconds == 3600
        
        # Invalid range should raise error
        with pytest.raises(ValueError):
            VideoFilters(
                min_duration_seconds=3600,
                max_duration_seconds=60
            )
    
    def test_view_count_validation(self):
        """Test view count range validation"""
        # Valid range
        filters = VideoFilters(
            min_views=100,
            max_views=10000
        )
        assert filters.min_views == 100
        assert filters.max_views == 10000
        
        # Invalid range should raise error
        with pytest.raises(ValueError):
            VideoFilters(
                min_views=10000,
                max_views=100
            )
    
    def test_date_range_validation(self):
        """Test date range validation"""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)
        
        # Valid range
        filters = VideoFilters(
            published_after=yesterday,
            published_before=tomorrow
        )
        assert filters.published_after == yesterday
        assert filters.published_before == tomorrow
        
        # Invalid range should raise error
        with pytest.raises(ValueError):
            VideoFilters(
                published_after=tomorrow,
                published_before=yesterday
            )
    
    def test_is_short_duration(self):
        """Test YouTube Shorts detection"""
        filters = VideoFilters()
        
        assert filters.is_short_duration(30) is True
        assert filters.is_short_duration(60) is True
        assert filters.is_short_duration(61) is False
        assert filters.is_short_duration(120) is False
    
    def test_apply_to_video_duration_filter(self):
        """Test applying duration filters to video data"""
        filters = VideoFilters(
            min_duration_seconds=120,
            max_duration_seconds=600
        )
        
        # Video within range
        video_data = {'duration_seconds': 300}
        assert filters.apply_to_video(video_data) is True
        
        # Video too short
        video_data = {'duration_seconds': 60}
        assert filters.apply_to_video(video_data) is False
        
        # Video too long
        video_data = {'duration_seconds': 700}
        assert filters.apply_to_video(video_data) is False
    
    def test_apply_to_video_shorts_filter(self):
        """Test filtering YouTube Shorts"""
        filters = VideoFilters(exclude_shorts=True)
        
        # Short video should be excluded
        video_data = {'duration_seconds': 45}
        assert filters.apply_to_video(video_data) is False
        
        # Regular video should pass
        video_data = {'duration_seconds': 300}
        assert filters.apply_to_video(video_data) is True
        
        # With shorts allowed
        filters = VideoFilters(exclude_shorts=False, min_duration_seconds=0)
        video_data = {'duration_seconds': 45}
        assert filters.apply_to_video(video_data) is True
    
    def test_apply_to_video_view_filter(self):
        """Test applying view count filters"""
        filters = VideoFilters(
            min_views=1000,
            max_views=100000
        )
        
        # Within range
        video_data = {'view_count': 5000}
        assert filters.apply_to_video(video_data) is True
        
        # Too few views
        video_data = {'view_count': 500}
        assert filters.apply_to_video(video_data) is False
        
        # Too many views
        video_data = {'view_count': 200000}
        assert filters.apply_to_video(video_data) is False
    
    def test_apply_to_video_date_filter(self):
        """Test applying date filters"""
        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        filters = VideoFilters(
            published_after=week_ago,
            published_before=now
        )
        
        # Video within date range
        video_data = {
            'published_at': (now - timedelta(days=3)).isoformat(),
            'duration_seconds': 300
        }
        assert filters.apply_to_video(video_data) is True
        
        # Video too old
        video_data = {
            'published_at': month_ago.isoformat(),
            'duration_seconds': 300
        }
        assert filters.apply_to_video(video_data) is False
    
    def test_apply_to_video_language_filter(self):
        """Test applying language filters"""
        filters = VideoFilters(language='en')
        
        # Matching language
        video_data = {
            'language': 'en',
            'duration_seconds': 300
        }
        assert filters.apply_to_video(video_data) is True
        
        # Non-matching language
        video_data = {
            'language': 'es',
            'duration_seconds': 300
        }
        assert filters.apply_to_video(video_data) is False
        
        # Multiple languages allowed
        filters = VideoFilters(languages=['en', 'es', 'fr'])
        video_data = {
            'language': 'es',
            'duration_seconds': 300
        }
        assert filters.apply_to_video(video_data) is True
    
    def test_apply_to_video_title_filter(self):
        """Test applying title filters"""
        filters = VideoFilters(
            title_contains=['tutorial', 'guide'],
            title_excludes=['clickbait', 'prank']
        )
        
        # Title contains required word
        video_data = {
            'title': 'Python Tutorial for Beginners',
            'duration_seconds': 300
        }
        assert filters.apply_to_video(video_data) is True
        
        # Title contains excluded word
        video_data = {
            'title': 'Epic Prank Tutorial',
            'duration_seconds': 300
        }
        assert filters.apply_to_video(video_data) is False
        
        # Title doesn't contain required words
        video_data = {
            'title': 'Random Video Title',
            'duration_seconds': 300
        }
        assert filters.apply_to_video(video_data) is False
    
    def test_complex_filter_combination(self):
        """Test complex combination of filters"""
        filters = VideoFilters(
            min_duration_seconds=300,  # 5 minutes
            max_duration_seconds=1800,  # 30 minutes
            min_views=10000,
            exclude_shorts=True,
            language='en',
            title_contains=['python'],
            published_after=datetime(2023, 1, 1, tzinfo=timezone.utc)
        )
        
        # Video that passes all filters
        good_video = {
            'duration_seconds': 600,
            'view_count': 50000,
            'language': 'en',
            'title': 'Advanced Python Programming',
            'published_at': '2023-06-15T00:00:00+00:00'
        }
        assert filters.apply_to_video(good_video) is True
        
        # Video that fails duration filter
        bad_duration = good_video.copy()
        bad_duration['duration_seconds'] = 60
        assert filters.apply_to_video(bad_duration) is False
        
        # Video that fails view filter
        bad_views = good_video.copy()
        bad_views['view_count'] = 5000
        assert filters.apply_to_video(bad_views) is False
        
        # Video that fails title filter
        bad_title = good_video.copy()
        bad_title['title'] = 'JavaScript Tutorial'
        assert filters.apply_to_video(bad_title) is False


class TestChannelFilters:
    """Test ChannelFilters functionality"""
    
    def test_default_channel_filters(self):
        """Test default channel filter values"""
        filters = ChannelFilters()
        
        assert filters.min_subscribers is None
        assert filters.max_subscribers is None
        assert filters.min_video_count is None
        assert filters.max_video_count is None
    
    def test_subscriber_count_filters(self):
        """Test subscriber count filtering"""
        filters = ChannelFilters(
            min_subscribers=1000,
            max_subscribers=1000000
        )
        
        assert filters.min_subscribers == 1000
        assert filters.max_subscribers == 1000000
    
    def test_video_count_filters(self):
        """Test video count filtering"""
        filters = ChannelFilters(
            min_video_count=10,
            max_video_count=1000
        )
        
        assert filters.min_video_count == 10
        assert filters.max_video_count == 1000
    
    def test_channel_date_filters(self):
        """Test channel creation date filters"""
        now = datetime.now(timezone.utc)
        year_ago = now - timedelta(days=365)
        
        filters = ChannelFilters(
            created_after=year_ago,
            created_before=now
        )
        
        assert filters.created_after == year_ago
        assert filters.created_before == now
    
    def test_channel_metadata_filters(self):
        """Test channel metadata filters"""
        filters = ChannelFilters(
            country='US',
            language='en',
            is_verified=True
        )
        
        assert filters.country == 'US'
        assert filters.language == 'en'
        assert filters.is_verified is True