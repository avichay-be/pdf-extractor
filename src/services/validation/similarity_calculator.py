"""
Similarity calculation for cross-validation.

Provides multiple similarity algorithms:
- Number frequency (cosine similarity on number distributions)
- Levenshtein distance (character-level edit distance)
- Jaccard similarity (quick pre-check)
"""
import math
import logging
from typing import Dict, List
from collections import Counter
import Levenshtein

from src.core.config import settings

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Calculate similarity between two text contents."""

    def __init__(self, normalizer=None):
        """
        Initialize similarity calculator.

        Args:
            normalizer: ContentNormalizer instance for text preprocessing
        """
        self.normalizer = normalizer

    def _calculate_number_frequency(self, numbers: List[str]) -> Dict[str, int]:
        """
        Build frequency map of numbers.

        Args:
            numbers: List of normalized number strings

        Returns:
            Dictionary mapping each unique number to its frequency
        """
        return dict(Counter(numbers))

    def _calculate_cosine_similarity(self, freq1: Dict[str, int], freq2: Dict[str, int]) -> float:
        """
        Calculate cosine similarity between two frequency distributions.

        Cosine similarity measures the angle between two vectors, ranging from 0 to 1.
        It's robust to different magnitudes and focuses on distribution shape.

        Args:
            freq1: Frequency distribution of numbers from first text
            freq2: Frequency distribution of numbers from second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not freq1 and not freq2:
            return 1.0  # Both empty = identical

        if not freq1 or not freq2:
            return 0.0  # One empty = completely different

        # Get all unique numbers from both distributions
        all_numbers = set(freq1.keys()) | set(freq2.keys())

        # Build vectors
        vec1 = [freq1.get(num, 0) for num in all_numbers]
        vec2 = [freq2.get(num, 0) for num in all_numbers]

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Cosine similarity
        similarity = dot_product / (magnitude1 * magnitude2)

        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

    def calculate_similarity_number_frequency(self, content1: str, content2: str) -> float:
        """
        Calculate similarity based on number frequency distributions.

        This method is ideal for financial documents where numerical accuracy
        is paramount. It compares the frequency of numbers appearing in both texts,
        ignoring all textual content and formatting.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity score between 0.0 and 1.0 (1.0 = identical number distributions)
        """
        if not self.normalizer:
            logger.warning("No normalizer provided - cannot extract numbers")
            return 0.0

        # Extract numbers from both texts
        numbers1 = self.normalizer.extract_numbers(content1)
        numbers2 = self.normalizer.extract_numbers(content2)

        # Build frequency distributions
        freq1 = self._calculate_number_frequency(numbers1)
        freq2 = self._calculate_number_frequency(numbers2)

        # Calculate cosine similarity between distributions
        similarity = self._calculate_cosine_similarity(freq1, freq2)

        # Log numeric comparison details
        logger.info(f"Numbers extracted from content1: {len(numbers1)} numbers - {freq1}")
        logger.info(f"Numbers extracted from content2: {len(numbers2)} numbers - {freq2}")
        logger.info(f"Number-based similarity: {similarity:.2%}")

        return similarity

    def calculate_similarity_levenshtein(self, content1: str, content2: str) -> float:
        """
        Calculate similarity using Levenshtein distance (character-level).
        Only alphanumeric characters are considered - formatting, punctuation,
        and whitespace differences are ignored.

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity score between 0.0 and 1.0 (1.0 = identical)
        """
        if not content1 and not content2:
            logger.info("Levenshtein: Both contents empty, returning 1.0")
            return 1.0  # Both empty = identical

        if not content1 or not content2:
            logger.info("Levenshtein: One content empty, returning 0.0")
            return 0.0  # One empty, one not = completely different

        if not self.normalizer:
            logger.warning("No normalizer provided - using raw content")
            normalized1 = content1.lower()
            normalized2 = content2.lower()
        else:
            # Normalize both strings to only alphanumeric characters
            normalized1 = self.normalizer.normalize_for_comparison(content1)
            normalized2 = self.normalizer.normalize_for_comparison(content2)

        # Handle edge case where both normalize to empty
        if not normalized1 and not normalized2:
            logger.info("Levenshtein: Both normalized to empty, returning 1.0")
            return 1.0  # Both have no alphanumeric content = identical

        if not normalized1 or not normalized2:
            logger.info("Levenshtein: One normalized to empty, returning 0.0")
            return 0.0  # One has content, the other doesn't

        # Calculate Levenshtein distance on normalized text
        distance = Levenshtein.distance(normalized1, normalized2)

        # Calculate similarity score based on normalized length
        max_length = max(len(normalized1), len(normalized2))
        similarity = 1.0 - (distance / max_length)

        # Log Levenshtein details
        logger.info(f"Levenshtein: normalized lengths: {len(normalized1)} vs {len(normalized2)}")
        logger.info(f"Levenshtein: edit distance: {distance}, max_length: {max_length}")
        logger.info(f"Levenshtein similarity: {similarity:.2%}")

        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

    def _quick_similarity(self, content1: str, content2: str) -> float:
        """
        Fast pre-check similarity using Jaccard similarity on word sets.

        This is much faster than full similarity calculation and good enough
        for early exit when content is obviously very similar (>95%).

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Jaccard similarity score between 0.0 and 1.0
        """
        # Quick length check - if lengths differ by >5%, likely different
        len1, len2 = len(content1), len(content2)
        if len1 == 0 or len2 == 0:
            return 0.0

        length_diff = abs(len1 - len2) / max(len1, len2)
        if length_diff > 0.05:
            return 0.0  # Not similar enough for early exit

        # Jaccard similarity on word sets
        words1 = set(content1.split())
        words2 = set(content2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def calculate_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate similarity between two strings.

        Performance optimization: Uses quick pre-check for early exit on obviously similar content.

        The method used depends on the VALIDATION_SIMILARITY_METHOD setting:
        - "number_frequency": Compare based on number frequency distributions (best for financial data)
        - "levenshtein": Compare based on character-level edit distance (alphanumeric only)

        Args:
            content1: First content string
            content2: Second content string

        Returns:
            Similarity score between 0.0 and 1.0 (1.0 = identical)
        """
        # Quick pre-check: if content is obviously very similar, skip expensive calculation
        quick_score = self._quick_similarity(content1, content2)
        if quick_score > 0.95:
            logger.info(f"Early exit: quick similarity {quick_score:.2%} > 95% (skipping full calculation)")
            return quick_score

        # Fall back to full calculation for more accurate scoring
        method = settings.VALIDATION_SIMILARITY_METHOD

        if method == "number_frequency":
            return self.calculate_similarity_number_frequency(content1, content2)
        elif method == "levenshtein":
            return self.calculate_similarity_levenshtein(content1, content2)
        else:
            logger.warning(f"Unknown similarity method '{method}', falling back to 'number_frequency'")
            return self.calculate_similarity_number_frequency(content1, content2)
