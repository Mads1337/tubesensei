# Phase 1B: YouTube Integration & Discovery

## Overview
This phase implements YouTube API integration for channel management and video discovery. Expected duration: 1 week.

## Prerequisites
- Phase 1A completed (database models and setup)
- Google Cloud Console account
- YouTube Data API v3 enabled
- API key generated

## Goals
- Integrate YouTube Data API v3 with quota management
- Build channel management service
- Implement video discovery functionality
- Handle API errors and rate limiting

## Task 1: YouTube API Setup

### 1.1 Install Dependencies
Add to `requirements.txt`:
```
google-api-python-client==2.108.0
google-auth==2.25.2
google-auth-httplib2==0.1.1
httpx==0.25.2
tenacity==8.2.3
```

### 1.2 API Configuration
Update `.env`:
```env
YOUTUBE_API_KEY=your_youtube_api_key_here
YOUTUBE_QUOTA_PER_DAY=10000
YOUTUBE_MAX_RESULTS_PER_PAGE=50
YOUTUBE_REQUEST_TIMEOUT=30
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### 1.3 Configuration Class
`app/config.py`:
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    youtube_api_key: str
    youtube_quota_per_day: int = 10000
    youtube_max_results_per_page: int = 50
    youtube_request_timeout: int = 30
    rate_limit_requests_per_minute: int = 60
    
    class Config:
        env_file = ".env"
```

## Task 2: YouTube API Integration

### 2.1 Base YouTube Client
`app/integrations/youtube_api.py`:

#### Core Components
- YouTube API client initialization
- Quota tracking and management
- Rate limiting implementation
- Error handling wrapper

#### Key Methods
```python
class YouTubeAPIClient:
    def __init__(self):
        # Initialize Google API client
        # Set up quota tracking
        # Configure rate limiter
    
    async def get_channel_info(self, channel_id: str) -> dict:
        # Fetch channel metadata
        # Handle quota consumption
        # Return channel details
    
    async def list_channel_videos(
        self, 
        channel_id: str, 
        max_results: int = 500
    ) -> List[dict]:
        # Paginated video fetching
        # Handle page tokens
        # Return video list
    
    async def get_video_details(
        self, 
        video_ids: List[str]
    ) -> List[dict]:
        # Batch fetch video metadata
        # Handle 50 video limit per request
        # Return video details
```

### 2.2 Quota Management
`app/integrations/quota_manager.py`:

#### Features
- Track daily quota usage
- Calculate cost per operation
- Implement quota reservation
- Auto-pause when near limit

#### Quota Costs (YouTube API v3)
- Channel info: 1 unit
- Video list: 100 units per page
- Video details: 1 unit per video
- Search: 100 units per request

### 2.3 Rate Limiting
`app/utils/rate_limiter.py`:

#### Implementation
- Token bucket algorithm
- Async-compatible
- Per-minute limits
- Automatic retry with backoff

## Task 3: Channel Management Service

### 3.1 Channel Manager
`app/services/channel_manager.py`:

#### Core Functions
```python
class ChannelManager:
    async def add_channel(self, channel_url: str) -> Channel:
        # Parse channel URL/ID
        # Check if exists in DB
        # Fetch metadata from YouTube
        # Store in database
        # Return channel object
    
    async def sync_channel_metadata(self, channel_id: UUID) -> Channel:
        # Get channel from DB
        # Fetch latest from YouTube
        # Update subscriber count, video count
        # Update last_checked_at
        # Return updated channel
    
    async def discover_channel_videos(
        self, 
        channel_id: UUID,
        fetch_all: bool = True
    ) -> List[Video]:
        # Get channel from DB
        # Fetch videos from YouTube
        # Filter existing videos
        # Store new videos
        # Queue for processing
        # Return video list
    
    async def get_channel_status(self, channel_id: UUID) -> dict:
        # Check channel health
        # Video discovery progress
        # Last sync time
        # Error status
```

### 3.2 Channel URL Parser
`app/utils/youtube_parser.py`:

#### Supported Formats
- `https://www.youtube.com/channel/UC...`
- `https://www.youtube.com/@username`
- `https://www.youtube.com/c/channelname`
- `https://youtube.com/user/username`

#### Functionality
- Extract channel ID from URL
- Handle vanity URLs
- Validate channel format
- Return standardized ID

## Task 4: Video Discovery Service

### 4.1 Video Discovery
`app/services/video_discovery.py`:

#### Core Functions
```python
class VideoDiscovery:
    async def discover_videos(
        self,
        channel_id: UUID,
        filters: Optional[VideoFilters] = None
    ) -> List[Video]:
        # Fetch videos from YouTube API
        # Apply filtering rules
        # Check for duplicates
        # Store in database
        # Return discovered videos
    
    async def batch_discover(
        self,
        channel_ids: List[UUID]
    ) -> dict:
        # Process multiple channels
        # Track progress
        # Handle failures
        # Return results summary
    
    async def update_video_metadata(
        self,
        video_ids: List[UUID]
    ) -> List[Video]:
        # Fetch latest stats
        # Update view counts
        # Update engagement metrics
        # Return updated videos
```

### 4.2 Video Filters
`app/models/filters.py`:

#### Filter Criteria
```python
class VideoFilters(BaseModel):
    min_duration_seconds: Optional[int] = 60
    max_duration_seconds: Optional[int] = 7200  # 2 hours
    min_views: Optional[int] = 1000
    published_after: Optional[datetime] = None
    published_before: Optional[datetime] = None
    exclude_shorts: bool = True
    language: Optional[str] = "en"
```

### 4.3 Duplicate Detection
- Check by youtube_video_id
- Handle re-uploads
- Track video versions
- Merge duplicate metadata

## Task 5: Error Handling

### 5.1 Custom Exceptions
`app/utils/exceptions.py`:
```python
class YouTubeAPIError(Exception): pass
class QuotaExceededError(YouTubeAPIError): pass
class ChannelNotFoundError(YouTubeAPIError): pass
class VideoNotFoundError(YouTubeAPIError): pass
class RateLimitError(YouTubeAPIError): pass
```

### 5.2 Retry Logic
- Exponential backoff for rate limits
- Immediate retry for network errors
- Skip and log for permanent failures
- Queue for later retry

## Testing Requirements

### Unit Tests
- [ ] YouTube URL parser handles all formats
- [ ] Quota manager tracks usage correctly
- [ ] Rate limiter enforces limits
- [ ] Filter logic works as expected

### Integration Tests
- [ ] Can fetch real channel data (with test API key)
- [ ] Video discovery returns expected results
- [ ] Pagination works correctly
- [ ] Error handling functions properly

### Mock Tests
- [ ] Mock YouTube API responses
- [ ] Test quota exhaustion handling
- [ ] Test network failure recovery
- [ ] Test malformed response handling

## API Endpoints (Optional)

### FastAPI Routes
```python
# app/main.py
@app.post("/channels")
async def add_channel(channel_url: str)

@app.get("/channels/{channel_id}")
async def get_channel(channel_id: UUID)

@app.post("/channels/{channel_id}/sync")
async def sync_channel(channel_id: UUID)

@app.post("/channels/{channel_id}/discover")
async def discover_videos(channel_id: UUID)
```

## Performance Considerations

### Optimization Tips
1. Cache channel metadata for 24 hours
2. Batch video detail requests (50 max)
3. Use async for all I/O operations
4. Implement connection pooling
5. Store API responses for debugging

### Monitoring
- Log all API calls with quota cost
- Track response times
- Monitor error rates
- Alert on quota threshold (80%)

## Deliverables
1. Working YouTube API integration with quota management
2. Channel management service with all CRUD operations
3. Video discovery with filtering and duplicate detection
4. Error handling and retry logic
5. Comprehensive test suite
6. API documentation

## Success Criteria
- [ ] Successfully add channels by URL
- [ ] Fetch and store channel metadata
- [ ] Discover all videos from a channel
- [ ] Handle API quota limits gracefully
- [ ] Process 100+ videos per minute
- [ ] Retry transient failures automatically
- [ ] All tests pass with >80% coverage

## Common Issues & Solutions

### Issue: API Key Invalid
**Solution**: Verify key in Google Console, check restrictions

### Issue: Quota Exceeded Quickly
**Solution**: Optimize requests, use batch operations, implement caching

### Issue: Channel URL Not Recognized
**Solution**: Update parser regex, handle new URL formats

### Issue: Rate Limit Errors
**Solution**: Implement exponential backoff, reduce concurrent requests

## Next Steps
After completing Phase 1B:
1. Test with real YouTube channels
2. Verify quota consumption is optimal
3. Document actual API costs
4. Proceed to Phase 1C: Transcript Processing