# Phase 1C: Transcript Processing

## Overview
This phase implements transcript extraction and processing from YouTube videos. Expected duration: 1 week.

## Prerequisites
- Phase 1A completed (database setup)
- Phase 1B completed (YouTube integration)
- Videos discovered and stored in database

## Goals
- Extract transcripts from YouTube videos
- Process and clean transcript content
- Store transcripts with metadata
- Implement quality metrics and caching

## Task 1: Transcript API Integration

### 1.1 Install Dependencies
Add to `requirements.txt`:
```
youtube-transcript-api==0.6.1
langdetect==1.0.9
ftfy==6.1.3
beautifulsoup4==4.12.2
html2text==2020.1.16
regex==2023.10.3
```

### 1.2 Transcript Configuration
Update `.env`:
```env
TRANSCRIPT_TIMEOUT_SECONDS=300
TRANSCRIPT_MAX_RETRIES=3
TRANSCRIPT_PREFERRED_LANGUAGES=en,en-US,en-GB
TRANSCRIPT_CACHE_TTL_HOURS=168
MIN_TRANSCRIPT_WORD_COUNT=100
MAX_TRANSCRIPT_LENGTH=500000
```

### 1.3 Update Configuration
`app/config.py`:
```python
class Settings(BaseSettings):
    # ... existing settings ...
    transcript_timeout_seconds: int = 300
    transcript_max_retries: int = 3
    transcript_preferred_languages: List[str] = ["en", "en-US", "en-GB"]
    transcript_cache_ttl_hours: int = 168
    min_transcript_word_count: int = 100
    max_transcript_length: int = 500000
```

## Task 2: Transcript Extraction

### 2.1 Transcript API Client
`app/integrations/transcript_api.py`:

#### Core Components
```python
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

class TranscriptAPIClient:
    def __init__(self):
        # Configure timeout
        # Set up language preferences
        # Initialize cache
    
    async def get_transcript(
        self,
        youtube_video_id: str,
        languages: Optional[List[str]] = None
    ) -> TranscriptData:
        # Try to fetch transcript
        # Handle multiple language options
        # Return transcript with metadata
    
    async def list_available_transcripts(
        self,
        youtube_video_id: str
    ) -> List[TranscriptInfo]:
        # Get all available transcripts
        # Include language info
        # Mark auto-generated vs manual
    
    async def get_transcript_with_timestamps(
        self,
        youtube_video_id: str
    ) -> List[TranscriptSegment]:
        # Fetch with timing data
        # Preserve timestamps
        # Return segmented transcript
```

### 2.2 Transcript Data Models
`app/models/transcript_data.py`:

```python
class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float
    end: float

class TranscriptData(BaseModel):
    content: str
    segments: List[TranscriptSegment]
    language: str
    is_auto_generated: bool
    confidence_score: Optional[float]

class TranscriptInfo(BaseModel):
    language: str
    language_code: str
    is_auto_generated: bool
    is_translatable: bool
    is_available: bool
```

### 2.3 Error Handling
`app/integrations/transcript_errors.py`:

```python
class TranscriptError(Exception):
    pass

class TranscriptNotAvailableError(TranscriptError):
    pass

class TranscriptDisabledError(TranscriptError):
    pass

class TranscriptTimeoutError(TranscriptError):
    pass

async def handle_transcript_error(error: Exception) -> Optional[str]:
    # Map API errors to custom exceptions
    # Log error details
    # Determine if retry is appropriate
    # Return error message for storage
```

## Task 3: Transcript Processing Service

### 3.1 Main Processing Service
`app/services/transcript_processor.py`:

#### Core Functions
```python
class TranscriptProcessor:
    def __init__(self):
        self.api_client = TranscriptAPIClient()
        self.cleaner = TranscriptCleaner()
        self.analyzer = TranscriptAnalyzer()
    
    async def extract_transcript(
        self,
        video_id: UUID
    ) -> Optional[Transcript]:
        # Get video from database
        # Check if transcript exists
        # Fetch from YouTube
        # Process and clean
        # Store in database
        # Return transcript object
    
    async def batch_process_transcripts(
        self,
        video_ids: List[UUID],
        concurrent_limit: int = 5
    ) -> ProcessingResult:
        # Process multiple videos
        # Handle rate limiting
        # Track success/failure
        # Return summary
    
    async def reprocess_failed_transcripts(
        self,
        session_id: Optional[UUID] = None
    ) -> ProcessingResult:
        # Find failed extractions
        # Retry with backoff
        # Update status
        # Return results
    
    async def update_transcript_quality(
        self,
        transcript_id: UUID
    ) -> Transcript:
        # Recalculate metrics
        # Update confidence score
        # Store updated data
        # Return transcript
```

### 3.2 Processing Result Model
```python
class ProcessingResult(BaseModel):
    total_processed: int
    successful: int
    failed: int
    skipped: int
    errors: List[ProcessingError]
    processing_time_seconds: float
    
class ProcessingError(BaseModel):
    video_id: UUID
    error_type: str
    error_message: str
    timestamp: datetime
```

## Task 4: Content Processing

### 4.1 Transcript Cleaner
`app/services/transcript_cleaner.py`:

#### Cleaning Operations
```python
class TranscriptCleaner:
    def clean_transcript(self, content: str) -> str:
        # Remove YouTube artifacts
        # Fix encoding issues
        # Normalize whitespace
        # Remove music/sound notations
        # Fix common OCR errors
        # Return cleaned text
    
    def remove_artifacts(self, text: str) -> str:
        # Remove [Music], [Applause]
        # Remove timestamps
        # Remove speaker labels
        # Remove URLs if needed
    
    def normalize_text(self, text: str) -> str:
        # Fix spacing issues
        # Correct punctuation
        # Handle line breaks
        # Standardize quotes
    
    def fix_encoding(self, text: str) -> str:
        # Use ftfy library
        # Handle unicode issues
        # Fix mojibake
        # Return clean UTF-8
```

### 4.2 Transcript Analyzer
`app/services/transcript_analyzer.py`:

#### Analysis Functions
```python
class TranscriptAnalyzer:
    def analyze_transcript(
        self,
        content: str
    ) -> TranscriptMetrics:
        # Calculate word count
        # Detect language
        # Assess quality
        # Extract key metrics
    
    def calculate_quality_score(
        self,
        content: str,
        is_auto_generated: bool
    ) -> float:
        # Check coherence
        # Assess completeness
        # Evaluate formatting
        # Return 0.0 to 1.0 score
    
    def detect_language(self, text: str) -> str:
        # Use langdetect
        # Handle mixed languages
        # Return ISO code
    
    def extract_metrics(self, content: str) -> dict:
        # Word count
        # Sentence count
        # Average sentence length
        # Unique words
        # Reading level
```

### 4.3 Transcript Metrics Model
```python
class TranscriptMetrics(BaseModel):
    word_count: int
    sentence_count: int
    unique_words: int
    avg_sentence_length: float
    detected_language: str
    quality_score: float
    has_timestamps: bool
    is_complete: bool
```

## Task 5: Caching System

### 5.1 Redis Cache Implementation
`app/services/transcript_cache.py`:

```python
class TranscriptCache:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = settings.transcript_cache_ttl_hours * 3600
    
    async def get(
        self,
        youtube_video_id: str
    ) -> Optional[TranscriptData]:
        # Check Redis cache
        # Deserialize if found
        # Return transcript data
    
    async def set(
        self,
        youtube_video_id: str,
        transcript_data: TranscriptData
    ):
        # Serialize transcript
        # Store with TTL
        # Log cache operation
    
    async def invalidate(self, youtube_video_id: str):
        # Remove from cache
        # Log invalidation
```

### 5.2 Cache Strategy
- Cache successful extractions for 7 days
- Don't cache failures
- Invalidate on reprocessing
- Use compression for large transcripts

## Task 6: Database Operations

### 6.1 Transcript Repository
`app/repositories/transcript_repository.py`:

```python
class TranscriptRepository:
    async def create(
        self,
        video_id: UUID,
        content: str,
        metadata: dict
    ) -> Transcript:
        # Create transcript record
        # Store with metadata
        # Return created object
    
    async def get_by_video(
        self,
        video_id: UUID
    ) -> Optional[Transcript]:
        # Query by video ID
        # Include relationships
        # Return transcript
    
    async def update_quality_metrics(
        self,
        transcript_id: UUID,
        metrics: TranscriptMetrics
    ) -> Transcript:
        # Update metrics
        # Store in metadata
        # Return updated object
    
    async def get_failed_extractions(
        self,
        limit: int = 100
    ) -> List[Video]:
        # Find videos without transcripts
        # Filter by status
        # Return for retry
```

## Testing Requirements

### Unit Tests
- [ ] Transcript extraction with mock API
- [ ] Text cleaning functions
- [ ] Quality score calculation
- [ ] Language detection accuracy
- [ ] Cache operations

### Integration Tests
- [ ] Extract real YouTube transcripts
- [ ] Handle various error conditions
- [ ] Batch processing performance
- [ ] Cache hit/miss scenarios
- [ ] Database storage and retrieval

### Performance Tests
- [ ] Process 100 transcripts/hour
- [ ] Handle large transcripts (>100k words)
- [ ] Concurrent processing limits
- [ ] Memory usage monitoring

## Monitoring & Logging

### Key Metrics
```python
# Log these for each transcript
{
    "video_id": "...",
    "extraction_time_ms": 1234,
    "word_count": 5678,
    "quality_score": 0.85,
    "source": "youtube_auto",
    "language": "en",
    "cache_hit": false,
    "error": null
}
```

### Alerts
- Extraction success rate < 80%
- Average extraction time > 10 seconds
- Cache hit rate < 50%
- Failed extraction backlog > 100

## Deliverables
1. Working transcript extraction from YouTube
2. Text cleaning and normalization
3. Quality metrics and scoring
4. Redis caching implementation
5. Batch processing capability
6. Comprehensive test coverage

## Success Criteria
- [ ] Extract transcripts from 90%+ videos with captions
- [ ] Process 100+ transcripts per hour
- [ ] Clean text properly (no artifacts)
- [ ] Accurate language detection
- [ ] Cache working with <100ms retrieval
- [ ] Handle all error cases gracefully
- [ ] Tests pass with >80% coverage

## Common Issues & Solutions

### Issue: Transcript Not Available
**Solution**: Check video availability, try different languages, mark as unavailable

### Issue: Rate Limiting from YouTube
**Solution**: Implement backoff, use caching, reduce concurrent requests

### Issue: Memory Issues with Large Transcripts
**Solution**: Stream processing, chunk large texts, implement size limits

### Issue: Poor Quality Auto-Generated Transcripts
**Solution**: Implement quality scoring, flag for manual review, use confidence thresholds

## Next Steps
After completing Phase 1C:
1. Test with variety of video types
2. Verify quality metrics accuracy
3. Optimize processing speed
4. Proceed to Phase 1D: Job Queue & Testing