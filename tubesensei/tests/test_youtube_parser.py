import pytest
from app.utils.youtube_parser import YouTubeParser
from app.utils.exceptions import InvalidURLError


class TestYouTubeParser:
    """Test YouTube URL parsing functionality"""
    
    def test_parse_channel_id_url(self):
        """Test parsing direct channel ID URL"""
        url = "https://www.youtube.com/channel/UCuAXFkgsw1L7xaCfnd5JJOw"
        result = YouTubeParser.parse_url(url)
        
        assert result['type'] == 'channel'
        assert result['channel_id'] == 'UCuAXFkgsw1L7xaCfnd5JJOw'
        assert result['id'] == 'UCuAXFkgsw1L7xaCfnd5JJOw'
    
    def test_parse_channel_handle_url(self):
        """Test parsing channel handle URL"""
        url = "https://www.youtube.com/@mkbhd"
        result = YouTubeParser.parse_url(url)
        
        assert result['type'] == 'channel'
        assert result['channel_handle'] == 'mkbhd'
        assert result['id'] == 'mkbhd'
    
    def test_parse_custom_channel_url(self):
        """Test parsing custom channel URL"""
        url = "https://www.youtube.com/c/MrBeast6000"
        result = YouTubeParser.parse_url(url)
        
        assert result['type'] == 'channel'
        assert result['channel_handle'] == 'MrBeast6000'
    
    def test_parse_user_channel_url(self):
        """Test parsing user channel URL"""
        url = "https://www.youtube.com/user/PewDiePie"
        result = YouTubeParser.parse_url(url)
        
        assert result['type'] == 'channel'
        assert result['channel_handle'] == 'PewDiePie'
    
    def test_parse_video_watch_url(self):
        """Test parsing standard video watch URL"""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = YouTubeParser.parse_url(url)
        
        assert result['type'] == 'video'
        assert result['video_id'] == 'dQw4w9WgXcQ'
        assert result['id'] == 'dQw4w9WgXcQ'
    
    def test_parse_video_short_url(self):
        """Test parsing short video URL"""
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = YouTubeParser.parse_url(url)
        
        assert result['type'] == 'video'
        assert result['video_id'] == 'dQw4w9WgXcQ'
    
    def test_parse_video_embed_url(self):
        """Test parsing embed video URL"""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        result = YouTubeParser.parse_url(url)
        
        assert result['type'] == 'video'
        assert result['video_id'] == 'dQw4w9WgXcQ'
    
    def test_parse_playlist_url(self):
        """Test parsing playlist URL"""
        url = "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        result = YouTubeParser.parse_url(url)
        
        assert result['type'] == 'playlist'
        assert result['playlist_id'] == 'PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf'
        assert result['id'] == 'PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf'
    
    def test_parse_video_with_playlist(self):
        """Test parsing video URL with playlist"""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"
        result = YouTubeParser.parse_url(url)
        
        assert result['type'] == 'video'
        assert result['video_id'] == 'dQw4w9WgXcQ'
        assert result['playlist_id'] == 'PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf'
    
    def test_validate_channel_id(self):
        """Test channel ID validation"""
        assert YouTubeParser.validate_channel_id('UCuAXFkgsw1L7xaCfnd5JJOw') is True
        assert YouTubeParser.validate_channel_id('invalid') is False
        assert YouTubeParser.validate_channel_id('UC' + 'x' * 20) is False  # Wrong length
        assert YouTubeParser.validate_channel_id('XX' + 'x' * 22) is False  # Wrong prefix
    
    def test_validate_video_id(self):
        """Test video ID validation"""
        assert YouTubeParser.validate_video_id('dQw4w9WgXcQ') is True
        assert YouTubeParser.validate_video_id('12345678901') is True
        assert YouTubeParser.validate_video_id('short') is False
        assert YouTubeParser.validate_video_id('toolongvideoid') is False
    
    def test_validate_playlist_id(self):
        """Test playlist ID validation"""
        assert YouTubeParser.validate_playlist_id('PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf') is True
        assert YouTubeParser.validate_playlist_id('UUuAXFkgsw1L7xaCfnd5JJOw') is True
        assert YouTubeParser.validate_playlist_id('invalid') is False
        assert YouTubeParser.validate_playlist_id('XXinvalid') is False
    
    def test_build_channel_url(self):
        """Test building channel URL"""
        channel_id = 'UCuAXFkgsw1L7xaCfnd5JJOw'
        url = YouTubeParser.build_channel_url(channel_id)
        assert url == f'https://www.youtube.com/channel/{channel_id}'
    
    def test_build_video_url(self):
        """Test building video URL"""
        video_id = 'dQw4w9WgXcQ'
        url = YouTubeParser.build_video_url(video_id)
        assert url == f'https://www.youtube.com/watch?v={video_id}'
    
    def test_build_playlist_url(self):
        """Test building playlist URL"""
        playlist_id = 'PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf'
        url = YouTubeParser.build_playlist_url(playlist_id)
        assert url == f'https://www.youtube.com/playlist?list={playlist_id}'
    
    def test_extract_video_timestamp(self):
        """Test extracting timestamp from video URL"""
        # Test with 't' parameter in seconds
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s"
        timestamp = YouTubeParser.extract_video_timestamp(url)
        assert timestamp == 120
        
        # Test with 't' parameter without 's'
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=90"
        timestamp = YouTubeParser.extract_video_timestamp(url)
        assert timestamp == 90
        
        # Test with complex time format
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=2m30s"
        timestamp = YouTubeParser.extract_video_timestamp(url)
        assert timestamp == 150
        
        # Test with hour format
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=1h30m45s"
        timestamp = YouTubeParser.extract_video_timestamp(url)
        assert timestamp == 5445
        
        # Test with no timestamp
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        timestamp = YouTubeParser.extract_video_timestamp(url)
        assert timestamp is None
    
    def test_invalid_url_raises_error(self):
        """Test that invalid URLs raise InvalidURLError"""
        with pytest.raises(InvalidURLError):
            YouTubeParser.parse_url("https://www.google.com")
        
        with pytest.raises(InvalidURLError):
            YouTubeParser.parse_url("not_a_url")
        
        with pytest.raises(InvalidURLError):
            YouTubeParser.parse_url("")
    
    def test_url_normalization(self):
        """Test URL normalization"""
        # Test without protocol
        url = "youtube.com/watch?v=dQw4w9WgXcQ"
        result = YouTubeParser.parse_url(url)
        assert result['video_id'] == 'dQw4w9WgXcQ'
        
        # Test with mobile URL
        url = "m.youtube.com/watch?v=dQw4w9WgXcQ"
        result = YouTubeParser.parse_url(url)
        assert result['video_id'] == 'dQw4w9WgXcQ'