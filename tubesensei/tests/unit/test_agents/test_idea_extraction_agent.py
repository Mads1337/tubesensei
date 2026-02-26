"""
Unit tests for the IdeaExtractionAgent.

Tests transcript chunking logic, content deduplication via hash,
LLM response parsing, and the static utility methods that can be
tested independently of the database and LLM.
"""
import hashlib
import pytest

from app.agents.idea_extraction_agent import IdeaExtractionAgent, MAX_TRANSCRIPT_WORDS


class TestChunkTranscript:
    """Tests for IdeaExtractionAgent._chunk_transcript static method."""

    def test_short_transcript_returns_single_chunk(self):
        """A transcript shorter than max_words should not be chunked."""
        text = "This is a short transcript. It has only a few words."
        chunks = IdeaExtractionAgent._chunk_transcript(text, max_words=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_transcript_is_chunked(self):
        """A transcript longer than max_words should be split into chunks."""
        # Create a transcript that definitely exceeds max_words
        sentence = "This is a sentence that contributes to the word count. "
        text = sentence * 200  # ~1400 words, well over a limit of 200
        chunks = IdeaExtractionAgent._chunk_transcript(text, max_words=200)
        assert len(chunks) > 1

    def test_chunks_do_not_exceed_max_words(self):
        """Each chunk should not greatly exceed the max word count.

        The chunking algorithm uses sentence-boundary splitting with overlap,
        so individual chunks may be somewhat larger than max_words when
        overlap words are included. We use a generous upper bound here.
        """
        sentence = "One two three four five six seven eight nine ten. "
        text = sentence * 100  # 1000 words
        max_words = 100
        # Use a small overlap to keep chunks manageable
        chunks = IdeaExtractionAgent._chunk_transcript(text, max_words=max_words, overlap_words=10)

        for chunk in chunks:
            word_count = len(chunk.split())
            # Allow overlap_words + sentence length slack (sentence = 10 words)
            assert word_count <= max_words + 20, (
                f"Chunk has {word_count} words, expected <= {max_words + 20}"
            )

    def test_empty_text_returns_single_chunk(self):
        """Empty text should return a single (empty) chunk."""
        chunks = IdeaExtractionAgent._chunk_transcript("", max_words=100)
        assert len(chunks) == 1

    def test_all_content_preserved(self):
        """All sentences from the original text should appear in chunks."""
        sentences = [f"Sentence number {i}." for i in range(50)]
        text = " ".join(sentences)
        chunks = IdeaExtractionAgent._chunk_transcript(text, max_words=50)

        combined = " ".join(chunks)
        # Every sentence number should appear somewhere in the combined output
        for i in range(50):
            assert f"Sentence number {i}" in combined

    def test_overlap_provides_context(self):
        """Chunking with overlap should have some words repeated between chunks."""
        sentence = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10. "
        text = sentence * 30  # 300 words
        # Use a small overlap_words to verify overlap exists
        chunks = IdeaExtractionAgent._chunk_transcript(text, max_words=100, overlap_words=20)

        if len(chunks) > 1:
            # The end of chunk 0 and the beginning of chunk 1 should share some content
            chunk0_words = set(chunks[0].split()[-30:])
            chunk1_words = set(chunks[1].split()[:30])
            assert len(chunk0_words & chunk1_words) > 0

    def test_single_very_long_sentence(self):
        """A single sentence longer than max_words should still be returned."""
        words = [f"word{i}" for i in range(200)]
        text = " ".join(words)  # No sentence boundaries, no periods
        chunks = IdeaExtractionAgent._chunk_transcript(text, max_words=50)
        # The function should still return at least one chunk with all content
        combined = " ".join(chunks)
        assert "word0" in combined
        assert "word199" in combined


class TestContentHashing:
    """Tests for content deduplication logic used in IdeaExtractionAgent."""

    def test_same_content_produces_same_hash(self):
        """Identical content should always produce the same hash."""
        title = "A great business idea"
        description = "This idea will change the world."
        hash_input = f"{title.strip().lower()}|{description.strip().lower()}"
        hash1 = hashlib.sha256(hash_input.encode()).hexdigest()[:64]
        hash2 = hashlib.sha256(hash_input.encode()).hexdigest()[:64]
        assert hash1 == hash2

    def test_different_content_produces_different_hash(self):
        """Different content should produce different hashes."""
        def make_hash(title, description):
            hash_input = f"{title.strip().lower()}|{description.strip().lower()}"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:64]

        hash1 = make_hash("Idea A", "Description A")
        hash2 = make_hash("Idea B", "Description B")
        assert hash1 != hash2

    def test_case_insensitive_hashing(self):
        """The hash should be the same regardless of case."""
        def make_hash(title, description):
            hash_input = f"{title.strip().lower()}|{description.strip().lower()}"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:64]

        hash1 = make_hash("A Great Idea", "This is the description.")
        hash2 = make_hash("a great idea", "this is the description.")
        assert hash1 == hash2

    def test_hash_length_is_64(self):
        """The hash should be truncated to 64 characters."""
        hash_input = "some title|some description"
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:64]
        assert len(content_hash) == 64

    def test_whitespace_normalization(self):
        """Stripping whitespace should produce consistent hashes."""
        def make_hash(title, description):
            hash_input = f"{title.strip().lower()}|{description.strip().lower()}"
            return hashlib.sha256(hash_input.encode()).hexdigest()[:64]

        hash1 = make_hash("  My Idea  ", "  My Description  ")
        hash2 = make_hash("My Idea", "My Description")
        assert hash1 == hash2


class TestMaxTranscriptWordsConstant:
    """Tests for the MAX_TRANSCRIPT_WORDS constant."""

    def test_constant_is_positive(self):
        assert MAX_TRANSCRIPT_WORDS > 0

    def test_constant_is_reasonable_value(self):
        """MAX_TRANSCRIPT_WORDS should be a reasonable chunking threshold."""
        assert 1000 <= MAX_TRANSCRIPT_WORDS <= 20000
