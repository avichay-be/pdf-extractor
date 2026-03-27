"""
Cross-validation for smart extraction results.

Validates page extraction results using existing ProblemDetector and
SimilarityCalculator components. Works with PageExtractionResult pairs
(not coupled to MistralOCRResponse).
"""
import asyncio
import logging
from dataclasses import replace
from typing import Optional

from src.services.gemini_client import GeminiDocumentClient
from src.services.smart_extraction.extraction_router import PageExtractionResult
from src.services.validation.problem_detector import ProblemDetector
from src.services.validation.similarity_calculator import SimilarityCalculator
from src.services.validation.content_normalizer import ContentNormalizer

logger = logging.getLogger(__name__)

# Similarity threshold for merging dual-source pages
MERGE_SIMILARITY_THRESHOLD = 0.8


class SmartValidationService:
    """Validates and improves smart extraction results."""

    def __init__(
        self,
        gemini_client: Optional[GeminiDocumentClient] = None,
        enable_single_source_fallback: bool = True,
    ):
        self._normalizer = ContentNormalizer()
        self._problem_detector = ProblemDetector(
            number_extractor=self._normalizer.extract_numbers
        )
        self._similarity_calculator = SimilarityCalculator(
            normalizer=self._normalizer
        )
        self._gemini_client = gemini_client
        self._enable_single_source_fallback = enable_single_source_fallback
        self._concurrency = 4

    async def validate_results(
        self,
        results: list[PageExtractionResult],
        pdf_bytes: Optional[bytes] = None,
    ) -> list[PageExtractionResult]:
        """
        Validate and improve extraction results.

        For dual-source pages (IMAGE_WITH_TEXT):
        - Check both sources for problems
        - Pick the clean one or merge if both clean but different
        - Fall back to Mistral if both problematic

        For single-source pages:
        - Check for problems
        - If problematic and Gemini available, re-extract

        Args:
            results: Page extraction results from ExtractionRouter
            pdf_bytes: PDF bytes for fallback re-extraction

        Returns:
            Validated (and possibly improved) list of PageExtractionResult
        """
        semaphore = asyncio.Semaphore(self._concurrency)

        async def validate_result(result: PageExtractionResult) -> PageExtractionResult:
            async with semaphore:
                if result.alternative_content is not None:
                    return await self._validate_dual_source(result)
                return await self._validate_single_source(result, pdf_bytes)

        validated = await asyncio.gather(*[
            validate_result(result) for result in results
        ])

        # Log summary
        improved = sum(
            1 for v, r in zip(validated, results)
            if v.content != r.content
        )
        if improved:
            logger.info(f"Validation improved {improved}/{len(results)} pages")

        return validated

    async def _validate_dual_source(
        self,
        result: PageExtractionResult
    ) -> PageExtractionResult:
        """
        Validate a page with both primary and alternative content.

        Decision logic:
        1. Check both for problems
        2. If one clean, one problematic -> use clean
        3. If both clean -> compare similarity, merge if different
        4. If both problematic -> keep primary (Mistral)
        """
        primary = result.content
        alternative = result.alternative_content or ""

        primary_has_problem = self._has_problems(primary)
        alt_has_problem = self._has_problems(alternative)

        if primary_has_problem and not alt_has_problem and alternative.strip():
            logger.info(
                f"Page {result.page_number}: primary has problems, "
                f"using alternative ({result.source} -> pdfplumber)"
            )
            return replace(
                result,
                content=alternative,
                source="pdfplumber",
                strategy="text_extraction",
                confidence=0.85,
            )

        if not primary_has_problem and alt_has_problem:
            # Primary is clean, keep it
            return replace(result, content=primary, confidence=0.9)

        if not primary_has_problem and not alt_has_problem:
            # Both clean - check similarity
            similarity = self._calculate_similarity(primary, alternative)
            if similarity < MERGE_SIMILARITY_THRESHOLD:
                # Different content - merge (take longer, supplement)
                merged = self._merge_contents(primary, alternative)
                logger.info(
                    f"Page {result.page_number}: merging dual sources "
                    f"(similarity={similarity:.2f})"
                )
                return replace(
                    result,
                    content=merged,
                    source="merged",
                    confidence=0.95,
                )
            # Similar enough - keep primary
            return replace(result, content=primary, confidence=0.9)

        # Both problematic - keep primary (Mistral typically better for images)
        return result

    async def _validate_single_source(
        self,
        result: PageExtractionResult,
        pdf_bytes: Optional[bytes] = None,
    ) -> PageExtractionResult:
        """
        Validate a single-source page.

        If problematic and Gemini is available, attempt re-extraction.
        """
        if not result.content.strip():
            return result

        has_problem = self._has_problems(result.content)
        if not has_problem or not self._enable_single_source_fallback:
            return result

        if result.page_range != (result.page_number, result.page_number):
            return result

        # Try Gemini fallback if available
        if pdf_bytes and self._gemini_client:
            try:
                content = await asyncio.to_thread(
                    self._gemini_client.extract_page_content,
                    pdf_bytes,
                    result.page_number,
                )
                if content.strip() and not self._has_problems(content):
                    logger.info(
                        f"Page {result.page_number}: Gemini fallback "
                        f"replaced problematic {result.source} content"
                    )
                    return replace(
                        result,
                        content=content,
                        source="gemini",
                        confidence=0.8,
                    )
            except Exception as e:
                logger.warning(
                    f"Gemini fallback failed for page {result.page_number}: {e}"
                )

        return result

    def _has_problems(self, content: str) -> bool:
        """Check content for quality problems using ProblemDetector."""
        if not content.strip():
            return True
        problems = self._problem_detector.detect_all_problems(content)
        return any(problems.values())

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return self._similarity_calculator.calculate_similarity(text1, text2)

    def _merge_contents(self, primary: str, alternative: str) -> str:
        """
        Merge two content sources, keeping the longer and supplementing.

        Takes the longer content as base, then appends any unique
        substantial lines from the shorter content.
        """
        if len(primary) >= len(alternative):
            base, supplement = primary, alternative
        else:
            base, supplement = alternative, primary

        # Find lines in supplement that are not in base
        base_lines = set(line.strip().lower() for line in base.split('\n') if line.strip())
        extra_lines = []
        for line in supplement.split('\n'):
            stripped = line.strip()
            if stripped and stripped.lower() not in base_lines and len(stripped) > 20:
                extra_lines.append(stripped)

        if extra_lines:
            return base + "\n\n" + "\n".join(extra_lines)
        return base
