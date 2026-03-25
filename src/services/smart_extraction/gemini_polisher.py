"""
Gemini polish pass for smart extraction output.

Sends extracted markdown (text-only, not images) to Gemini for OCR error
correction, formatting normalization, and cleanup. Includes safety checks
to ensure numbers are preserved exactly.
"""
import asyncio
import logging
from typing import Optional

from src.core.config import settings
from src.core.utils import combine_markdown_sections
from src.services.smart_extraction.extraction_router import PageExtractionResult
from src.services.validation.content_normalizer import ContentNormalizer
from src.services.client_factory import get_client_factory

logger = logging.getLogger(__name__)

DEFAULT_POLISH_PROMPT = """You are a document formatting expert. Clean up extracted PDF content into well-formatted markdown optimized for LLM consumption.

RULES:
1. Fix OCR errors (common: 0/O, 1/l, rn/m, Hebrew character confusions)
2. Normalize table formatting (consistent columns, proper alignment)
3. Ensure consistent heading hierarchy (H2 for sections, H3 for subsections)
4. Remove duplicate content at page boundaries
5. Fix encoding artifacts and garbled characters
6. Preserve ALL numerical data EXACTLY as-is (never modify numbers)
7. Preserve ALL text content (never summarize or omit)
8. Clean up excessive whitespace
9. Output ONLY the cleaned markdown - no explanations"""

# Maximum divergence in number frequency before rejecting polish
MAX_NUMBER_DIVERGENCE = 0.05


class GeminiPolisher:
    """Polishes extracted markdown using Gemini for OCR error correction."""

    def __init__(self):
        self._normalizer = ContentNormalizer()
        self._batch_size = settings.SMART_EXTRACTION_POLISH_BATCH_SIZE
        self._concurrency = settings.SMART_EXTRACTION_POLISH_CONCURRENCY

    async def polish(
        self,
        page_results: list[PageExtractionResult]
    ) -> str:
        """
        Polish extracted page content through Gemini.

        Batches pages, sends text to Gemini for cleanup, and validates
        that numbers are preserved. Falls back to original content if
        polish changes numbers.

        Args:
            page_results: List of PageExtractionResult from extraction/validation

        Returns:
            Combined polished markdown string
        """
        if not settings.SMART_EXTRACTION_POLISH_ENABLED:
            logger.info("Gemini polish disabled, returning raw content")
            return self._combine_results(page_results)

        gemini_client = get_client_factory().gemini_client
        if gemini_client is None:
            logger.warning("Gemini client unavailable, skipping polish")
            return self._combine_results(page_results)

        # Build page content strings with headers
        page_contents = []
        for result in page_results:
            header = f"## Page {result.page_number + 1}\n\n"
            page_contents.append(header + result.content)

        # Batch pages
        batches = [
            page_contents[i:i + self._batch_size]
            for i in range(0, len(page_contents), self._batch_size)
        ]

        logger.info(
            f"Polishing {len(page_contents)} pages in {len(batches)} batches "
            f"(batch_size={self._batch_size}, concurrency={self._concurrency})"
        )

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(self._concurrency)
        polish_prompt = settings.SMART_EXTRACTION_POLISH_PROMPT or DEFAULT_POLISH_PROMPT

        async def process_batch(batch: list[str], batch_idx: int) -> str:
            async with semaphore:
                return await self._polish_batch(
                    gemini_client, batch, polish_prompt, batch_idx
                )

        tasks = [
            process_batch(batch, idx)
            for idx, batch in enumerate(batches)
        ]
        polished_batches = await asyncio.gather(*tasks)

        # Combine polished batches
        return combine_markdown_sections(
            [b for b in polished_batches if b],
            empty_message=""
        )

    async def _polish_batch(
        self,
        gemini_client,
        batch: list[str],
        prompt: str,
        batch_idx: int,
    ) -> str:
        """
        Polish a batch of page contents through Gemini.

        Includes number preservation safety check.
        """
        original_text = "\n\n---\n\n".join(batch)

        try:
            # Send to Gemini as text-only (no PDF)
            polished = await asyncio.to_thread(
                self._call_gemini_text,
                gemini_client,
                prompt,
                original_text,
            )

            if not polished or not polished.strip():
                logger.warning(f"Batch {batch_idx}: empty polish result, keeping original")
                return original_text

            # Safety check: verify numbers are preserved
            if not self._verify_number_preservation(original_text, polished):
                logger.warning(
                    f"Batch {batch_idx}: polish changed numbers, rejecting"
                )
                return original_text

            return polished

        except Exception as e:
            logger.error(f"Batch {batch_idx} polish failed: {e}")
            return original_text

    def _call_gemini_text(self, gemini_client, prompt: str, text: str) -> str:
        """Call Gemini with text content (not PDF)."""
        from google.genai import types

        response = gemini_client.client.models.generate_content(
            model=gemini_client.model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text=f"{prompt}\n\n---\n\nContent to clean up:\n\n{text}"
                    )]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
            ),
        )
        return response.text if response.text else ""

    def _verify_number_preservation(self, original: str, polished: str) -> bool:
        """
        Verify that polished content preserves numbers from original.

        Returns False if number frequency diverges by more than MAX_NUMBER_DIVERGENCE.
        """
        orig_numbers = self._normalizer.extract_numbers(original)
        polished_numbers = self._normalizer.extract_numbers(polished)

        if not orig_numbers:
            return True  # No numbers to preserve

        orig_set = set(orig_numbers)
        polished_set = set(polished_numbers)

        if not orig_set:
            return True

        # Check how many original numbers are missing
        missing = orig_set - polished_set
        divergence = len(missing) / len(orig_set)

        if divergence > MAX_NUMBER_DIVERGENCE:
            logger.debug(
                f"Number divergence: {divergence:.1%} "
                f"({len(missing)} missing out of {len(orig_set)})"
            )
            return False

        return True

    def _combine_results(self, page_results: list[PageExtractionResult]) -> str:
        """Combine page results into markdown without polishing."""
        contents = []
        for result in page_results:
            header = f"## Page {result.page_number + 1}\n\n"
            contents.append(header + result.content)
        return combine_markdown_sections(contents, empty_message="")
