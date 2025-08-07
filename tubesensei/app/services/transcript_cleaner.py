import re
import logging
from typing import Optional, List, Dict, Tuple
import ftfy
from bs4 import BeautifulSoup
import html2text
import unicodedata

from ..config import settings

logger = logging.getLogger(__name__)


class TranscriptCleaner:
    """
    Service for cleaning and normalizing transcript text.
    Removes artifacts, fixes encoding, and improves readability.
    """
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # HTML to text converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # Don't wrap lines
        
        logger.info("Initialized TranscriptCleaner")
    
    def _compile_patterns(self):
        """Compile regex patterns for cleaning operations."""
        # YouTube artifacts patterns
        self.artifact_patterns = {
            'music': re.compile(r'\[(?:Music|♪|♫|🎵|🎶|🎼|🎤)\]', re.IGNORECASE),
            'applause': re.compile(r'\[(?:Applause|👏)\]', re.IGNORECASE),
            'laughter': re.compile(r'\[(?:Laughter|Laughs?|😂|🤣|Haha)\]', re.IGNORECASE),
            'inaudible': re.compile(r'\[(?:Inaudible|Unintelligible|Unclear|__)\]', re.IGNORECASE),
            'crosstalk': re.compile(r'\[(?:Crosstalk|Overlapping)\]', re.IGNORECASE),
            'silence': re.compile(r'\[(?:Silence|Pause|\.\.\.)\]', re.IGNORECASE),
            'sound_effects': re.compile(r'\[(?:Sound Effect|SFX|Noise|Bang|Crash|Boom)\]', re.IGNORECASE),
            'generic_brackets': re.compile(r'\[[^\]]{1,50}\]'),  # Any short bracketed content
            'parenthetical_sounds': re.compile(r'\((?:laughs?|sighs?|coughs?|clears throat)\)', re.IGNORECASE)
        }
        
        # Speaker identification patterns
        self.speaker_patterns = {
            'speaker_label': re.compile(r'^(?:Speaker \d+:|Host:|Guest:|Interviewer:|[A-Z][a-z]+:)\s*', re.MULTILINE),
            'timestamp': re.compile(r'(?:\d{1,2}:\d{2}(?::\d{2})?|\[\d{1,2}:\d{2}(?::\d{2})?\])\s*'),
            'chapter_marker': re.compile(r'^\s*(?:Chapter \d+|Part \d+|Section \d+)[:\s]*', re.MULTILINE | re.IGNORECASE)
        }
        
        # Normalization patterns
        self.normalization_patterns = {
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_newlines': re.compile(r'\n{3,}'),
            'leading_trailing_spaces': re.compile(r'^\s+|\s+$', re.MULTILINE),
            'space_before_punctuation': re.compile(r'\s+([.!?,;:])'),
            'missing_space_after_punctuation': re.compile(r'([.!?,;:])([A-Za-z])'),
            'multiple_punctuation': re.compile(r'([.!?]){2,}'),
            'quotes_normalization': re.compile(r'[""]'),
            'apostrophe_normalization': re.compile(r'['']'),
            'dash_normalization': re.compile(r'[–—]'),
            'ellipsis_normalization': re.compile(r'\.{3,}')
        }
        
        # URL and email patterns
        self.link_patterns = {
            'url': re.compile(r'https?://[^\s]+|www\.[^\s]+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        }
        
        # Common OCR/ASR errors
        self.error_corrections = {
            r'\bIm\b': "I'm",
            r'\bIve\b': "I've",
            r'\bId\b': "I'd",
            r'\bIll\b': "I'll",
            r'\byoure\b': "you're",
            r'\byouve\b': "you've",
            r'\byoud\b': "you'd",
            r'\byoull\b': "you'll",
            r'\btheyre\b': "they're",
            r'\btheyve\b': "they've",
            r'\btheyd\b': "they'd",
            r'\btheyll\b': "they'll",
            r'\bwere\b': "we're",
            r'\bweve\b': "we've",
            r'\bwed\b': "we'd",
            r'\bwell\b': "we'll",
            r'\bcant\b': "can't",
            r'\bwont\b': "won't",
            r'\bdont\b': "don't",
            r'\bdoesnt\b': "doesn't",
            r'\bdidnt\b': "didn't",
            r'\bhasnt\b': "hasn't",
            r'\bhavent\b': "haven't",
            r'\bhadnt\b': "hadn't",
            r'\bwouldnt\b': "wouldn't",
            r'\bcouldnt\b': "couldn't",
            r'\bshouldnt\b': "shouldn't",
            r'\bmightnt\b': "mightn't",
            r'\bmustnt\b': "mustn't",
            r'\bwhats\b': "what's",
            r'\bthats\b': "that's",
            r'\bwhos\b': "who's",
            r'\bwheres\b': "where's",
            r'\bwhens\b': "when's",
            r'\bwhys\b': "why's",
            r'\bhows\b': "how's",
            r'\btheres\b': "there's",
            r'\bheres\b': "here's",
            r'\blets\b': "let's"
        }
    
    def clean_transcript(
        self,
        content: str,
        remove_artifacts: bool = True,
        fix_encoding: bool = True,
        normalize_text: bool = True,
        remove_timestamps: bool = True,
        remove_speaker_labels: bool = True,
        fix_common_errors: bool = True,
        preserve_urls: bool = False
    ) -> str:
        """
        Main cleaning function that applies all cleaning operations.
        
        Args:
            content: Raw transcript text
            remove_artifacts: Remove YouTube artifacts like [Music]
            fix_encoding: Fix encoding issues
            normalize_text: Normalize whitespace and punctuation
            remove_timestamps: Remove timestamp markers
            remove_speaker_labels: Remove speaker identification
            fix_common_errors: Fix common OCR/ASR errors
            preserve_urls: Keep URLs in the text
            
        Returns:
            Cleaned transcript text
        """
        if not content:
            return ""
        
        original_length = len(content)
        logger.debug(f"Starting transcript cleaning (original length: {original_length})")
        
        # Step 1: Fix encoding issues first
        if fix_encoding:
            content = self.fix_encoding(content)
        
        # Step 2: Clean HTML if present
        if '<' in content and '>' in content:
            content = self.clean_html(content)
        
        # Step 3: Remove artifacts
        if remove_artifacts:
            content = self.remove_artifacts(content)
        
        # Step 4: Remove timestamps
        if remove_timestamps:
            content = self.remove_timestamps(content)
        
        # Step 5: Remove speaker labels
        if remove_speaker_labels:
            content = self.remove_speaker_labels(content)
        
        # Step 6: Fix common errors
        if fix_common_errors:
            content = self.fix_common_errors(content)
        
        # Step 7: Handle URLs
        if not preserve_urls:
            content = self.remove_urls(content)
        
        # Step 8: Normalize text (should be last)
        if normalize_text:
            content = self.normalize_text(content)
        
        # Final validation
        content = self.validate_and_trim(content)
        
        cleaned_length = len(content)
        reduction_percent = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
        logger.debug(f"Cleaning complete (new length: {cleaned_length}, reduction: {reduction_percent:.1f}%)")
        
        return content
    
    def remove_artifacts(self, text: str) -> str:
        """
        Remove YouTube-specific artifacts from transcript.
        
        Args:
            text: Transcript text
            
        Returns:
            Text without artifacts
        """
        for pattern_name, pattern in self.artifact_patterns.items():
            before_count = len(pattern.findall(text))
            if before_count > 0:
                text = pattern.sub('', text)
                logger.debug(f"Removed {before_count} {pattern_name} artifacts")
        
        return text
    
    def remove_timestamps(self, text: str) -> str:
        """
        Remove timestamp markers from transcript.
        
        Args:
            text: Transcript text
            
        Returns:
            Text without timestamps
        """
        return self.speaker_patterns['timestamp'].sub('', text)
    
    def remove_speaker_labels(self, text: str) -> str:
        """
        Remove speaker identification labels.
        
        Args:
            text: Transcript text
            
        Returns:
            Text without speaker labels
        """
        text = self.speaker_patterns['speaker_label'].sub('', text)
        text = self.speaker_patterns['chapter_marker'].sub('', text)
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize whitespace, punctuation, and formatting.
        
        Args:
            text: Transcript text
            
        Returns:
            Normalized text
        """
        # Normalize quotes and apostrophes
        text = self.normalization_patterns['quotes_normalization'].sub('"', text)
        text = self.normalization_patterns['apostrophe_normalization'].sub("'", text)
        
        # Normalize dashes
        text = self.normalization_patterns['dash_normalization'].sub('-', text)
        
        # Normalize ellipsis
        text = self.normalization_patterns['ellipsis_normalization'].sub('...', text)
        
        # Fix spacing around punctuation
        text = self.normalization_patterns['space_before_punctuation'].sub(r'\1', text)
        text = self.normalization_patterns['missing_space_after_punctuation'].sub(r'\1 \2', text)
        
        # Reduce multiple punctuation
        text = self.normalization_patterns['multiple_punctuation'].sub(r'\1', text)
        
        # Normalize whitespace
        text = self.normalization_patterns['multiple_spaces'].sub(' ', text)
        text = self.normalization_patterns['multiple_newlines'].sub('\n\n', text)
        text = self.normalization_patterns['leading_trailing_spaces'].sub('', text)
        
        # Final strip
        text = text.strip()
        
        return text
    
    def fix_encoding(self, text: str) -> str:
        """
        Fix encoding issues using ftfy library.
        
        Args:
            text: Transcript text
            
        Returns:
            Text with fixed encoding
        """
        try:
            # Use ftfy to fix mojibake and other encoding issues
            fixed_text = ftfy.fix_text(text)
            
            # Additional normalization for Unicode
            fixed_text = unicodedata.normalize('NFKC', fixed_text)
            
            return fixed_text
        except Exception as e:
            logger.warning(f"Error fixing encoding: {e}")
            return text
    
    def fix_common_errors(self, text: str) -> str:
        """
        Fix common OCR/ASR errors.
        
        Args:
            text: Transcript text
            
        Returns:
            Text with corrected errors
        """
        for pattern, replacement in self.error_corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def clean_html(self, text: str) -> str:
        """
        Clean HTML content if present.
        
        Args:
            text: Text possibly containing HTML
            
        Returns:
            Plain text without HTML
        """
        try:
            # First try BeautifulSoup for proper HTML parsing
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text(separator=' ')
            
            # Then use html2text for any remaining markup
            text = self.html_converter.handle(text)
            
        except Exception as e:
            logger.warning(f"Error cleaning HTML: {e}")
        
        return text
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Args:
            text: Transcript text
            
        Returns:
            Text without URLs
        """
        text = self.link_patterns['url'].sub('', text)
        return text
    
    def validate_and_trim(self, text: str) -> str:
        """
        Validate and trim text to maximum length.
        
        Args:
            text: Transcript text
            
        Returns:
            Validated and trimmed text
        """
        # Check maximum length
        if len(text) > settings.MAX_TRANSCRIPT_LENGTH:
            logger.warning(f"Transcript exceeds max length, trimming from {len(text)} to {settings.MAX_TRANSCRIPT_LENGTH}")
            text = text[:settings.MAX_TRANSCRIPT_LENGTH]
            
            # Try to cut at sentence boundary
            last_period = text.rfind('.')
            if last_period > settings.MAX_TRANSCRIPT_LENGTH * 0.9:
                text = text[:last_period + 1]
        
        # Check minimum word count
        word_count = len(text.split())
        if word_count < settings.MIN_TRANSCRIPT_WORD_COUNT:
            logger.warning(f"Transcript has only {word_count} words, below minimum of {settings.MIN_TRANSCRIPT_WORD_COUNT}")
        
        return text.strip()
    
    def get_cleaning_statistics(self, original: str, cleaned: str) -> Dict[str, any]:
        """
        Get statistics about the cleaning process.
        
        Args:
            original: Original text
            cleaned: Cleaned text
            
        Returns:
            Dictionary with cleaning statistics
        """
        original_words = len(original.split())
        cleaned_words = len(cleaned.split())
        
        return {
            "original_length": len(original),
            "cleaned_length": len(cleaned),
            "length_reduction_percent": ((len(original) - len(cleaned)) / len(original) * 100) if len(original) > 0 else 0,
            "original_word_count": original_words,
            "cleaned_word_count": cleaned_words,
            "word_reduction_percent": ((original_words - cleaned_words) / original_words * 100) if original_words > 0 else 0,
            "artifacts_removed": sum(len(p.findall(original)) for p in self.artifact_patterns.values()),
            "urls_removed": len(self.link_patterns['url'].findall(original))
        }
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract individual sentences from text.
        
        Args:
            text: Transcript text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK or spaCy)
        sentence_endings = re.compile(r'[.!?]+')
        sentences = sentence_endings.split(text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        sentences = [s for s in sentences if len(s.split()) > 2]  # Filter very short sentences
        
        return sentences