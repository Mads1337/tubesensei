import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter
import statistics
from langdetect import detect, detect_langs, LangDetectException
import math

from ..config import settings
from ..models.transcript_data import TranscriptMetrics

logger = logging.getLogger(__name__)


class TranscriptAnalyzer:
    """
    Service for analyzing transcript content and quality.
    Calculates metrics, detects language, and assesses quality.
    """
    
    def __init__(self):
        # Common English stop words for analysis
        self.stop_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
            'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
            'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
            'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work',
            'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
            'give', 'day', 'most', 'us'
        }
        
        # Sentence ending patterns
        self.sentence_pattern = re.compile(r'[.!?]+')
        
        # Word pattern (alphanumeric sequences)
        self.word_pattern = re.compile(r'\b\w+\b')
        
        # Quality indicators
        self.quality_indicators = {
            'incomplete_sentences': re.compile(r'\b(um|uh|er|ah|like|you know|I mean)\b', re.IGNORECASE),
            'filler_words': re.compile(r'\b(basically|actually|literally|obviously|seriously)\b', re.IGNORECASE),
            'repetition_pattern': re.compile(r'\b(\w+)\s+\1\b', re.IGNORECASE),  # Repeated words
            'all_caps': re.compile(r'\b[A-Z]{2,}\b'),  # All caps words (might indicate issues)
            'numbers_only': re.compile(r'^\d+$'),  # Lines with only numbers
        }
        
        logger.info("Initialized TranscriptAnalyzer")
    
    def analyze_transcript(
        self,
        content: str,
        is_auto_generated: bool = True,
        detect_language_flag: bool = True
    ) -> TranscriptMetrics:
        """
        Perform comprehensive analysis of transcript content.
        
        Args:
            content: Transcript text to analyze
            is_auto_generated: Whether transcript is auto-generated
            detect_language_flag: Whether to detect language
            
        Returns:
            TranscriptMetrics object with analysis results
        """
        if not content:
            return self._empty_metrics()
        
        logger.debug(f"Analyzing transcript (length: {len(content)})")
        
        # Basic text statistics
        word_count = self.count_words(content)
        sentence_count = self.count_sentences(content)
        unique_words = self.count_unique_words(content)
        avg_sentence_length = self.calculate_avg_sentence_length(content)
        
        # Language detection
        detected_language = "en"  # Default
        language_confidence = 1.0
        if detect_language_flag:
            detected_language, language_confidence = self.detect_language(content)
        
        # Quality assessment
        quality_score = self.calculate_quality_score(
            content=content,
            is_auto_generated=is_auto_generated,
            word_count=word_count,
            sentence_count=sentence_count,
            unique_words=unique_words
        )
        
        # Check completeness
        is_complete = self.check_completeness(content, word_count, sentence_count)
        
        # Estimate reading level
        reading_level = self.estimate_reading_level(
            avg_sentence_length=avg_sentence_length,
            unique_word_ratio=unique_words / word_count if word_count > 0 else 0
        )
        
        metrics = TranscriptMetrics(
            word_count=word_count,
            sentence_count=sentence_count,
            unique_words=unique_words,
            avg_sentence_length=avg_sentence_length,
            detected_language=detected_language,
            language_confidence=language_confidence,
            quality_score=quality_score,
            has_timestamps=True,  # Assume true for YouTube transcripts
            is_complete=is_complete,
            reading_level=reading_level
        )
        
        logger.debug(f"Analysis complete: {metrics.dict()}")
        return metrics
    
    def calculate_quality_score(
        self,
        content: str,
        is_auto_generated: bool,
        word_count: int,
        sentence_count: int,
        unique_words: int
    ) -> float:
        """
        Calculate overall quality score for transcript.
        
        Args:
            content: Transcript text
            is_auto_generated: Whether transcript is auto-generated
            word_count: Total word count
            sentence_count: Total sentence count
            unique_words: Number of unique words
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if word_count == 0:
            return 0.0
        
        score = 1.0
        
        # Factor 1: Auto-generated penalty
        if is_auto_generated:
            score *= 0.9
        
        # Factor 2: Word count adequacy
        if word_count < settings.MIN_TRANSCRIPT_WORD_COUNT:
            score *= 0.7
        elif word_count < settings.MIN_TRANSCRIPT_WORD_COUNT * 2:
            score *= 0.85
        
        # Factor 3: Vocabulary diversity
        vocab_diversity = unique_words / word_count if word_count > 0 else 0
        if vocab_diversity < 0.2:  # Very low diversity
            score *= 0.8
        elif vocab_diversity < 0.3:  # Low diversity
            score *= 0.9
        elif vocab_diversity > 0.7:  # Very high diversity (might be noise)
            score *= 0.95
        
        # Factor 4: Sentence structure
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if avg_sentence_length < 5:  # Very short sentences
                score *= 0.85
            elif avg_sentence_length > 30:  # Very long sentences
                score *= 0.9
        else:
            score *= 0.7  # No proper sentences detected
        
        # Factor 5: Check for quality indicators
        quality_penalties = self._assess_quality_indicators(content)
        score *= quality_penalties
        
        # Factor 6: Coherence check
        coherence_score = self._assess_coherence(content)
        score *= coherence_score
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, score))
    
    def _assess_quality_indicators(self, content: str) -> float:
        """
        Assess quality based on presence of quality indicators.
        
        Args:
            content: Transcript text
            
        Returns:
            Penalty multiplier (1.0 = no penalty)
        """
        penalty = 1.0
        content_lower = content.lower()
        
        # Check for excessive filler words
        filler_count = len(self.quality_indicators['filler_words'].findall(content))
        word_count = len(content.split())
        if word_count > 0:
            filler_ratio = filler_count / word_count
            if filler_ratio > 0.05:  # More than 5% filler words
                penalty *= 0.95
        
        # Check for incomplete sentences
        incomplete_count = len(self.quality_indicators['incomplete_sentences'].findall(content))
        if word_count > 0:
            incomplete_ratio = incomplete_count / word_count
            if incomplete_ratio > 0.1:  # More than 10% incomplete markers
                penalty *= 0.9
        
        # Check for word repetitions
        repetition_count = len(self.quality_indicators['repetition_pattern'].findall(content))
        if repetition_count > 10:
            penalty *= 0.95
        
        # Check for all caps abuse
        all_caps_count = len(self.quality_indicators['all_caps'].findall(content))
        if all_caps_count > word_count * 0.1:  # More than 10% all caps
            penalty *= 0.9
        
        return penalty
    
    def _assess_coherence(self, content: str) -> float:
        """
        Assess text coherence and structure.
        
        Args:
            content: Transcript text
            
        Returns:
            Coherence score multiplier
        """
        sentences = self.extract_sentences(content)
        if len(sentences) < 2:
            return 1.0  # Not enough data to assess
        
        coherence_score = 1.0
        
        # Check sentence length variation
        sentence_lengths = [len(s.split()) for s in sentences]
        if sentence_lengths:
            std_dev = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
            mean_length = statistics.mean(sentence_lengths)
            
            # High variation might indicate issues
            if mean_length > 0:
                cv = std_dev / mean_length  # Coefficient of variation
                if cv > 1.5:  # Very high variation
                    coherence_score *= 0.95
        
        # Check for common word transitions between sentences
        transition_score = self._check_sentence_transitions(sentences)
        coherence_score *= transition_score
        
        return coherence_score
    
    def _check_sentence_transitions(self, sentences: List[str]) -> float:
        """
        Check for smooth transitions between sentences.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Transition quality score
        """
        if len(sentences) < 2:
            return 1.0
        
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'nevertheless', 'meanwhile', 'subsequently', 'additionally',
            'similarly', 'likewise', 'conversely', 'finally', 'thus'
        }
        
        transition_count = 0
        for sentence in sentences[1:]:  # Skip first sentence
            words = sentence.lower().split()
            if words and words[0] in transition_words:
                transition_count += 1
        
        # Some transitions are good, but not too many
        transition_ratio = transition_count / (len(sentences) - 1)
        if 0.1 <= transition_ratio <= 0.3:
            return 1.0
        elif transition_ratio < 0.05:
            return 0.95  # Too few transitions
        elif transition_ratio > 0.5:
            return 0.9  # Too many transitions (might be artificial)
        else:
            return 0.97
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the text with confidence score.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            # Use first 1000 characters for faster detection
            sample = text[:1000] if len(text) > 1000 else text
            
            # Get language probabilities
            langs = detect_langs(sample)
            
            if langs:
                # Return the most probable language
                top_lang = langs[0]
                return top_lang.lang, top_lang.prob
            else:
                return "en", 0.5  # Default fallback
                
        except LangDetectException as e:
            logger.warning(f"Language detection failed: {e}")
            return "en", 0.5  # Default fallback
    
    def count_words(self, text: str) -> int:
        """Count total words in text."""
        words = self.word_pattern.findall(text)
        return len(words)
    
    def count_sentences(self, text: str) -> int:
        """Count total sentences in text."""
        sentences = self.sentence_pattern.split(text)
        # Filter out empty strings and very short fragments
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return len(sentences)
    
    def count_unique_words(self, text: str) -> int:
        """Count unique words in text (case-insensitive)."""
        words = self.word_pattern.findall(text.lower())
        return len(set(words))
    
    def calculate_avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = self.extract_sentences(text)
        if not sentences:
            return 0.0
        
        lengths = [len(self.word_pattern.findall(s)) for s in sentences]
        return statistics.mean(lengths) if lengths else 0.0
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract individual sentences from text."""
        sentences = self.sentence_pattern.split(text)
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        return sentences
    
    def extract_metrics(self, content: str) -> Dict[str, Any]:
        """
        Extract detailed metrics from content.
        
        Args:
            content: Transcript text
            
        Returns:
            Dictionary of metrics
        """
        words = self.word_pattern.findall(content.lower())
        word_count = len(words)
        
        # Calculate various metrics
        metrics = {
            "word_count": word_count,
            "sentence_count": self.count_sentences(content),
            "unique_words": len(set(words)),
            "character_count": len(content),
            "avg_word_length": statistics.mean([len(w) for w in words]) if words else 0,
            "lexical_diversity": len(set(words)) / word_count if word_count > 0 else 0,
            "paragraph_count": len([p for p in content.split('\n\n') if p.strip()]),
        }
        
        # Top words (excluding stop words)
        content_words = [w for w in words if w not in self.stop_words]
        word_freq = Counter(content_words)
        metrics["top_words"] = word_freq.most_common(10)
        
        # Readability metrics
        if metrics["sentence_count"] > 0:
            metrics["avg_sentence_length"] = word_count / metrics["sentence_count"]
        else:
            metrics["avg_sentence_length"] = 0
        
        return metrics
    
    def check_completeness(self, content: str, word_count: int, sentence_count: int) -> bool:
        """
        Check if transcript appears complete.
        
        Args:
            content: Transcript text
            word_count: Word count
            sentence_count: Sentence count
            
        Returns:
            True if transcript appears complete
        """
        # Check minimum thresholds
        if word_count < settings.MIN_TRANSCRIPT_WORD_COUNT:
            return False
        
        if sentence_count < 3:  # At least 3 sentences
            return False
        
        # Check if text ends properly (with punctuation)
        content_trimmed = content.strip()
        if content_trimmed and content_trimmed[-1] not in '.!?"\'':
            return False
        
        # Check for truncation indicators
        truncation_indicators = ['...', '[truncated]', '[cut off]', '[incomplete]']
        for indicator in truncation_indicators:
            if indicator.lower() in content.lower()[-100:]:  # Check last 100 chars
                return False
        
        return True
    
    def estimate_reading_level(self, avg_sentence_length: float, unique_word_ratio: float) -> str:
        """
        Estimate reading level based on text complexity.
        
        Args:
            avg_sentence_length: Average sentence length in words
            unique_word_ratio: Ratio of unique words to total words
            
        Returns:
            Estimated reading level
        """
        # Simplified reading level estimation
        # Could be enhanced with Flesch-Kincaid or similar algorithms
        
        complexity_score = (avg_sentence_length * 0.5) + (unique_word_ratio * 100 * 0.5)
        
        if complexity_score < 30:
            return "Elementary"
        elif complexity_score < 50:
            return "Middle School"
        elif complexity_score < 70:
            return "High School"
        elif complexity_score < 90:
            return "College"
        else:
            return "Graduate"
    
    def _empty_metrics(self) -> TranscriptMetrics:
        """Return empty metrics for invalid content."""
        return TranscriptMetrics(
            word_count=0,
            sentence_count=0,
            unique_words=0,
            avg_sentence_length=0.0,
            detected_language="unknown",
            language_confidence=0.0,
            quality_score=0.0,
            has_timestamps=False,
            is_complete=False,
            reading_level=None
        )
    
    def get_summary_statistics(self, content: str) -> Dict[str, Any]:
        """
        Get comprehensive summary statistics for reporting.
        
        Args:
            content: Transcript text
            
        Returns:
            Dictionary of summary statistics
        """
        metrics = self.analyze_transcript(content)
        detailed_metrics = self.extract_metrics(content)
        
        return {
            "basic_metrics": metrics.dict(),
            "detailed_metrics": detailed_metrics,
            "quality_assessment": {
                "score": metrics.quality_score,
                "is_complete": metrics.is_complete,
                "language": metrics.detected_language,
                "language_confidence": metrics.language_confidence,
                "reading_level": metrics.reading_level
            }
        }