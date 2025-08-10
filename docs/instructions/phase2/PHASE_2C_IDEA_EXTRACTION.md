# Phase 2C: Idea Extraction Pipeline (Week 3)

## Objectives
- Develop transcript analysis pipeline
- Implement business idea extraction
- Add categorization and confidence scoring
- Create deduplication system

## Implementation Steps

### Step 1: Create Idea Extractor Module

**File:** `tubesensei/app/ai/idea_extractor.py`
```python
"""
Extract and structure business ideas from video transcripts.

This module provides:
- AI-powered idea extraction from transcripts
- Idea structuring and categorization
- Confidence scoring and quality assessment
- Deduplication and similarity detection
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import hashlib
import re
from dataclasses import dataclass

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .llm_manager import LLMManager, ModelType
from .prompt_templates import PromptTemplates, PromptType
from .response_parser import ResponseParser, ParsedIdea
from ..models.video import Video
from ..models.transcript import Transcript
from ..models.idea import Idea, IdeaStatus
from ..database import get_db_session
from ..utils.logger import get_logger
from ..config import settings

logger = get_logger(__name__)

@dataclass
class ExtractedIdea:
    """Enriched idea with all metadata."""
    title: str
    description: str
    category: str
    target_market: str
    value_proposition: str
    complexity_score: int
    confidence_score: float
    quality_score: float
    source_context: str
    video_id: str
    timestamp: Optional[int] = None
    tags: List[str] = None
    market_size_estimate: Optional[str] = None
    similar_ideas: List[str] = None

class IdeaExtractor:
    """Extract business ideas from video transcripts."""
    
    def __init__(self):
        """Initialize idea extractor."""
        self.llm_manager = LLMManager()
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.idea_cache = []  # For deduplication
        
    async def initialize(self):
        """Initialize the extractor."""
        await self.llm_manager.initialize()
    
    async def extract_ideas_from_transcript(
        self,
        transcript: Transcript,
        video: Video,
        use_quality_model: bool = False
    ) -> List[ExtractedIdea]:
        """
        Extract business ideas from a transcript.
        
        Args:
            transcript: Transcript object
            video: Associated video object
            use_quality_model: Use higher quality model for extraction
            
        Returns:
            List of extracted ideas
        """
        try:
            # Split transcript if too long
            chunks = self._split_transcript(transcript.content)
            all_ideas = []
            
            for chunk_idx, chunk in enumerate(chunks):
                # Prepare prompt variables
                variables = {
                    "transcript": chunk,
                    "title": video.title,
                    "channel_name": video.channel.name if video.channel else "Unknown",
                    "duration_minutes": video.duration_seconds // 60 if video.duration_seconds else 0
                }
                
                # Get prompt
                system_prompt, user_prompt = PromptTemplates.get_prompt(
                    PromptType.IDEA_EXTRACTION,
                    variables
                )
                
                # Get LLM response
                model_type = ModelType.QUALITY if use_quality_model else ModelType.BALANCED
                response = await self.llm_manager.complete(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model_type=model_type,
                    temperature=0.5,
                    max_tokens=2000
                )
                
                # Parse ideas
                parsed_ideas = ResponseParser.parse_idea_extraction_response(response.content)
                
                # Convert to ExtractedIdea objects
                for idea in parsed_ideas:
                    extracted = ExtractedIdea(
                        title=idea.title,
                        description=idea.description,
                        category=idea.category,
                        target_market=idea.target_market,
                        value_proposition=idea.value_proposition,
                        complexity_score=idea.complexity_score,
                        confidence_score=idea.confidence,
                        quality_score=0.0,  # Will be assessed separately
                        source_context=idea.source_context,
                        video_id=str(video.id),
                        timestamp=self._estimate_timestamp(chunk_idx, len(chunks), video.duration_seconds)
                    )
                    
                    all_ideas.append(extracted)
            
            # Deduplicate ideas
            unique_ideas = await self._deduplicate_ideas(all_ideas)
            
            # Assess quality of each idea
            for idea in unique_ideas:
                idea.quality_score = await self._assess_idea_quality(idea)
            
            # Categorize and enrich ideas
            for idea in unique_ideas:
                enrichment = await self._enrich_idea(idea)
                idea.tags = enrichment.get("tags", [])
                idea.market_size_estimate = enrichment.get("market_size")
            
            logger.info(f"Extracted {len(unique_ideas)} unique ideas from video {video.id}")
            
            return unique_ideas
            
        except Exception as e:
            logger.error(f"Error extracting ideas from transcript: {str(e)}")
            return []
    
    def _split_transcript(self, content: str, max_tokens: int = 3000) -> List[str]:
        """
        Split transcript into chunks for processing.
        
        Args:
            content: Full transcript text
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of transcript chunks
        """
        # Rough estimation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(content) <= max_chars:
            return [content]
        
        # Split by paragraphs or sentences
        paragraphs = content.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_chars:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _estimate_timestamp(self, chunk_idx: int, total_chunks: int, duration: int) -> int:
        """Estimate timestamp for idea based on chunk position."""
        if total_chunks <= 1 or not duration:
            return 0
        
        return int((chunk_idx / total_chunks) * duration)
    
    async def _deduplicate_ideas(self, ideas: List[ExtractedIdea]) -> List[ExtractedIdea]:
        """
        Remove duplicate or very similar ideas.
        
        Args:
            ideas: List of extracted ideas
            
        Returns:
            List of unique ideas
        """
        if len(ideas) <= 1:
            return ideas
        
        # Create text representations for similarity comparison
        idea_texts = [f"{idea.title} {idea.description}" for idea in ideas]
        
        # Calculate similarity matrix
        try:
            tfidf_matrix = self.vectorizer.fit_transform(idea_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find unique ideas (similarity threshold: 0.8)
            unique_indices = []
            seen = set()
            
            for i in range(len(ideas)):
                if i in seen:
                    continue
                    
                unique_indices.append(i)
                
                # Mark similar ideas as seen
                for j in range(i + 1, len(ideas)):
                    if similarity_matrix[i][j] > 0.8:
                        seen.add(j)
                        # Merge confidence scores
                        ideas[i].confidence_score = max(
                            ideas[i].confidence_score,
                            ideas[j].confidence_score
                        )
            
            return [ideas[i] for i in unique_indices]
            
        except Exception as e:
            logger.warning(f"Deduplication failed, returning all ideas: {e}")
            return ideas
    
    async def _assess_idea_quality(self, idea: ExtractedIdea) -> float:
        """
        Assess the quality of an extracted idea.
        
        Args:
            idea: Extracted idea to assess
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            # Prepare prompt variables
            variables = {
                "title": idea.title,
                "description": idea.description,
                "category": idea.category,
                "source_context": idea.source_context
            }
            
            # Get prompt
            system_prompt, user_prompt = PromptTemplates.get_prompt(
                PromptType.QUALITY_ASSESSMENT,
                variables
            )
            
            # Get LLM response (use fast model for assessment)
            response = await self.llm_manager.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model_type=ModelType.FAST,
                temperature=0.3,
                max_tokens=800
            )
            
            # Parse response
            result = ResponseParser.parse_quality_assessment_response(response.content)
            
            return result.get("quality_score", 0.5)
            
        except Exception as e:
            logger.warning(f"Quality assessment failed for idea: {e}")
            return idea.confidence_score  # Fallback to confidence score
    
    async def _enrich_idea(self, idea: ExtractedIdea) -> Dict[str, Any]:
        """
        Enrich idea with additional metadata.
        
        Args:
            idea: Idea to enrich
            
        Returns:
            Enrichment data
        """
        try:
            # Prepare prompt variables
            variables = {
                "title": idea.title,
                "description": idea.description,
                "video_title": "From transcript"
            }
            
            # Get prompt
            system_prompt, user_prompt = PromptTemplates.get_prompt(
                PromptType.IDEA_CATEGORIZATION,
                variables
            )
            
            # Get LLM response
            response = await self.llm_manager.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model_type=ModelType.FAST,
                temperature=0.5,
                max_tokens=600
            )
            
            # Parse response
            enrichment = ResponseParser.parse_json_response(response.content)
            
            # Extract tags from various fields
            tags = set()
            if enrichment.get("industry"):
                tags.add(enrichment["industry"].lower())
            if enrichment.get("business_model"):
                tags.add(enrichment["business_model"].lower())
            if enrichment.get("required_skills"):
                tags.update([s.lower() for s in enrichment["required_skills"]])
            
            enrichment["tags"] = list(tags)
            
            return enrichment
            
        except Exception as e:
            logger.warning(f"Enrichment failed for idea: {e}")
            return {"tags": [], "market_size": "Unknown"}
    
    async def process_video(
        self,
        video_id: str,
        min_quality_score: float = 0.5
    ) -> List[Idea]:
        """
        Complete idea extraction pipeline for a video.
        
        Args:
            video_id: Video ID to process
            min_quality_score: Minimum quality score to save idea
            
        Returns:
            List of saved Idea objects
        """
        async with get_db_session() as session:
            # Get video and transcript
            video = await session.get(Video, video_id)
            if not video:
                logger.error(f"Video {video_id} not found")
                return []
            
            # Get transcript
            result = await session.execute(
                select(Transcript).where(Transcript.video_id == video_id)
            )
            transcript = result.scalar_one_or_none()
            
            if not transcript:
                logger.error(f"No transcript found for video {video_id}")
                return []
            
            # Extract ideas
            extracted_ideas = await self.extract_ideas_from_transcript(
                transcript,
                video
            )
            
            # Filter by quality
            quality_ideas = [
                idea for idea in extracted_ideas
                if idea.quality_score >= min_quality_score
            ]
            
            # Save to database
            saved_ideas = []
            for extracted in quality_ideas:
                idea = Idea(
                    video_id=video_id,
                    title=extracted.title,
                    description=extracted.description,
                    category=extracted.category,
                    market_size_estimate=extracted.market_size_estimate,
                    complexity_score=extracted.complexity_score,
                    confidence_score=extracted.confidence_score,
                    tags=extracted.tags or [],
                    status=IdeaStatus.EXTRACTED,
                    metadata={
                        "quality_score": extracted.quality_score,
                        "target_market": extracted.target_market,
                        "value_proposition": extracted.value_proposition,
                        "source_context": extracted.source_context,
                        "timestamp": extracted.timestamp
                    }
                )
                
                session.add(idea)
                saved_ideas.append(idea)
            
            await session.commit()
            
            logger.info(f"Saved {len(saved_ideas)} ideas from video {video_id}")
            
            return saved_ideas
    
    async def find_similar_ideas(
        self,
        idea: Idea,
        threshold: float = 0.7
    ) -> List[Idea]:
        """
        Find ideas similar to a given idea.
        
        Args:
            idea: Reference idea
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar ideas
        """
        async with get_db_session() as session:
            # Get all ideas
            result = await session.execute(
                select(Idea).where(Idea.id != idea.id)
            )
            all_ideas = result.scalars().all()
            
            if not all_ideas:
                return []
            
            # Create text representations
            reference_text = f"{idea.title} {idea.description}"
            idea_texts = [f"{i.title} {i.description}" for i in all_ideas]
            
            # Calculate similarities
            all_texts = [reference_text] + idea_texts
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            
            # Find similar ideas
            similar_ideas = []
            for i, similarity in enumerate(similarities):
                if similarity >= threshold:
                    all_ideas[i].similarity_score = similarity
                    similar_ideas.append(all_ideas[i])
            
            # Sort by similarity
            similar_ideas.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return similar_ideas
    
    async def close(self):
        """Cleanup resources."""
        await self.llm_manager.close()
```

## Testing Phase 2C

**File:** `tubesensei/tests/test_ai/test_idea_extractor.py`
```python
"""Tests for idea extraction."""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from tubesensei.app.ai.idea_extractor import IdeaExtractor, ExtractedIdea
from tubesensei.app.models.transcript import Transcript
from tubesensei.app.models.video import Video

@pytest.mark.asyncio
async def test_idea_extraction():
    """Test idea extraction from transcript."""
    extractor = IdeaExtractor()
    
    # Mock transcript and video
    transcript = Mock(spec=Transcript)
    transcript.content = "This video discusses building a SaaS business..."
    
    video = Mock(spec=Video)
    video.id = "test-video"
    video.title = "SaaS Business Ideas"
    video.duration_seconds = 600
    video.channel = Mock(name="Tech Channel")
    
    with patch.object(extractor.llm_manager, 'complete', new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = Mock(
            content='''
            {
                "ideas": [
                    {
                        "title": "AI-powered CRM",
                        "description": "CRM with AI features",
                        "category": "SaaS",
                        "target_market": "SMBs",
                        "value_proposition": "Automate sales",
                        "complexity_score": 7,
                        "confidence": 0.85,
                        "source_context": "From transcript"
                    }
                ],
                "total_ideas": 1
            }
            '''
        )
        
        ideas = await extractor.extract_ideas_from_transcript(transcript, video)
        
        assert len(ideas) == 1
        assert ideas[0].title == "AI-powered CRM"
        assert ideas[0].category == "SaaS"
        assert ideas[0].confidence_score == 0.85

def test_transcript_splitting():
    """Test transcript splitting logic."""
    extractor = IdeaExtractor()
    
    # Short transcript
    short_text = "Short content"
    chunks = extractor._split_transcript(short_text)
    assert len(chunks) == 1
    
    # Long transcript
    long_text = "Very long content " * 1000
    chunks = extractor._split_transcript(long_text, max_tokens=100)
    assert len(chunks) > 1

@pytest.mark.asyncio
async def test_idea_deduplication():
    """Test idea deduplication."""
    extractor = IdeaExtractor()
    
    ideas = [
        ExtractedIdea(
            title="AI CRM System",
            description="CRM with artificial intelligence",
            category="SaaS",
            target_market="SMB",
            value_proposition="Automate",
            complexity_score=7,
            confidence_score=0.8,
            quality_score=0.7,
            source_context="",
            video_id="1"
        ),
        ExtractedIdea(
            title="AI-powered CRM",  # Similar to first
            description="CRM using AI technology",
            category="SaaS",
            target_market="SMB",
            value_proposition="Automate",
            complexity_score=7,
            confidence_score=0.85,
            quality_score=0.75,
            source_context="",
            video_id="1"
        ),
        ExtractedIdea(
            title="Mobile Game",  # Different
            description="Casual mobile game",
            category="Gaming",
            target_market="Gamers",
            value_proposition="Entertainment",
            complexity_score=5,
            confidence_score=0.7,
            quality_score=0.6,
            source_context="",
            video_id="1"
        )
    ]
    
    unique = await extractor._deduplicate_ideas(ideas)
    
    # Should keep the first similar idea and the different one
    assert len(unique) == 2
    assert any("Game" in idea.title for idea in unique)
```

## Key Features

### 1. Idea Extraction
- AI-powered extraction from transcripts
- Handles long transcripts via chunking
- Structured idea output with metadata

### 2. Quality Assessment
- Confidence scoring for each idea
- Quality score calculation
- Context preservation from source

### 3. Deduplication
- TF-IDF based similarity detection
- Configurable similarity threshold
- Merges confidence scores from duplicates

### 4. Enrichment
- Automatic categorization
- Tag extraction
- Market size estimation
- Target market identification

### 5. Database Integration
- Saves extracted ideas to database
- Links ideas to source videos
- Maintains extraction metadata

## Configuration

The idea extraction system can be configured via environment variables:

```bash
# Extraction settings
IDEA_EXTRACTION_MIN_QUALITY=0.5
IDEA_EXTRACTION_MAX_TOKENS=3000
IDEA_EXTRACTION_SIMILARITY_THRESHOLD=0.8

# Model selection
IDEA_EXTRACTION_USE_QUALITY_MODEL=false
```

## Usage Example

```python
# Initialize extractor
extractor = IdeaExtractor()
await extractor.initialize()

# Process a video
ideas = await extractor.process_video(
    video_id="video-123",
    min_quality_score=0.6
)

# Find similar ideas
similar = await extractor.find_similar_ideas(
    idea=ideas[0],
    threshold=0.7
)

# Cleanup
await extractor.close()
```

## Validation Checklist

- [ ] Ideas extracted successfully from transcripts
- [ ] Quality scoring produces meaningful results
- [ ] Deduplication removes similar ideas
- [ ] Categorization assigns appropriate categories
- [ ] Tags are relevant and useful
- [ ] Similar idea detection works accurately
- [ ] Database storage maintains data integrity
- [ ] Error handling prevents data loss
- [ ] Performance meets requirements (< 30s per video)
- [ ] Cost tracking accurate for AI usage