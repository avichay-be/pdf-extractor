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
from src.core.utils import combine_markdown_sections, repair_hebrew_ocr_text
from src.services.gemini_client import (
    GeminiDocumentClient,
    extract_gemini_text_response,
)
from src.services.smart_extraction.extraction_router import PageExtractionResult
from src.services.validation.content_normalizer import ContentNormalizer

logger = logging.getLogger(__name__)

# Maximum divergence in number frequency before rejecting polish
MAX_NUMBER_DIVERGENCE = 0.05
DEPRECATED_POLISH_MODELS = {
    "gemini-3-1-flash-lite-preview",
}


class GeminiPolisher:
    """Polishes extracted markdown using Gemini for OCR error correction."""

    def __init__(self, gemini_client: Optional[GeminiDocumentClient] = None):
        self._gemini_client = gemini_client
        self._normalizer = ContentNormalizer()
        self._batch_size = settings.SMART_EXTRACTION_POLISH_BATCH_SIZE
        self._concurrency = settings.SMART_EXTRACTION_POLISH_CONCURRENCY
        self._model_name = self._select_model_name()

    def _select_model_name(self) -> str:
        """Select a supported polish model, falling back to the extraction model."""
        configured_model = settings.SMART_EXTRACTION_POLISH_MODEL
        if configured_model in DEPRECATED_POLISH_MODELS:
            logger.warning(
                f"Configured polish model {configured_model} is deprecated; "
                f"using {settings.GEMINI_MODEL}"
            )
            return settings.GEMINI_MODEL
        return configured_model or settings.GEMINI_MODEL

    async def polish(
        self,
        page_results: list[PageExtractionResult],
        enabled: Optional[bool] = None,
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
        should_polish = (
            enabled
            if enabled is not None
            else settings.SMART_EXTRACTION_POLISH_ENABLED
        )

        if not should_polish:
            logger.info("Gemini polish disabled, returning raw content")
            return self._combine_results(page_results)

        if not self._needs_polish(page_results):
            logger.info("Gemini polish skipped, content is already digital/table-clean")
            return self._combine_results(page_results)

        if self._gemini_client is None:
            logger.warning("Gemini client unavailable, skipping polish")
            return self._combine_results(page_results)

        polishable_segments = self._collect_polishable_segments(page_results)
        if not polishable_segments:
            logger.info("Gemini polish skipped, no OCR-derived results were found")
            return self._combine_results(page_results)

        polishable_count = sum(len(segment) for segment in polishable_segments)
        batch_count = sum(
            (len(segment) + self._batch_size - 1) // self._batch_size
            for segment in polishable_segments
        )

        logger.info(
            f"Polishing {polishable_count} OCR-derived results across "
            f"{len(polishable_segments)} segments in {batch_count} batches "
            f"(batch_size={self._batch_size}, concurrency={self._concurrency}, "
            f"model={self._model_name})"
        )

        combined_parts: list[str] = []
        current_segment: list[PageExtractionResult] = []

        for result in page_results:
            if self._should_polish_result(result):
                current_segment.append(result)
                continue

            if current_segment:
                combined_parts.append(await self._polish_segment(current_segment))
                current_segment = []

            combined_parts.append(self._render_result(result))

        if current_segment:
            combined_parts.append(await self._polish_segment(current_segment))

        return combine_markdown_sections(
            [part for part in combined_parts if part],
            empty_message=""
        )

    def _needs_polish(self, page_results: list[PageExtractionResult]) -> bool:
        """Return True when the document likely benefits from Gemini cleanup."""
        return any(self._should_polish_result(result) for result in page_results)

    def _collect_polishable_segments(
        self,
        page_results: list[PageExtractionResult],
    ) -> list[list[PageExtractionResult]]:
        """Return contiguous OCR-derived segments that should be polished."""
        segments: list[list[PageExtractionResult]] = []
        current_segment: list[PageExtractionResult] = []

        for result in page_results:
            if self._should_polish_result(result):
                current_segment.append(result)
                continue

            if current_segment:
                segments.append(current_segment)
                current_segment = []

        if current_segment:
            segments.append(current_segment)

        return segments

    def _should_polish_result(self, result: PageExtractionResult) -> bool:
        """Return True when a result comes from OCR-derived extraction."""
        if result.source in {"mistral", "gemini", "merged"}:
            return True
        if result.strategy == "ocr" and result.source != "pdfplumber":
            return True
        return False

    async def _polish_segment(
        self,
        segment_results: list[PageExtractionResult],
    ) -> str:
        """Polish a contiguous OCR-only segment while keeping its internal order."""
        page_contents = [self._render_result(result) for result in segment_results]
        batches = [
            page_contents[i:i + self._batch_size]
            for i in range(0, len(page_contents), self._batch_size)
        ]

        semaphore = asyncio.Semaphore(self._concurrency)
        polish_prompt = settings.get_smart_extraction_polish_prompt()

        async def process_batch(batch: list[str], batch_idx: int) -> str:
            async with semaphore:
                return await self._polish_batch(
                    self._gemini_client, batch, polish_prompt, batch_idx
                )

        polished_batches = await asyncio.gather(
            *[process_batch(batch, idx) for idx, batch in enumerate(batches)]
        )

        return combine_markdown_sections(
            [batch for batch in polished_batches if batch],
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
                self._model_name,
                prompt,
                original_text,
            )

            if not polished or not polished.strip():
                logger.warning(f"Batch {batch_idx}: empty polish result, keeping original")
                return repair_hebrew_ocr_text(original_text)

            # Safety check: verify numbers are preserved
            if not self._verify_number_preservation(original_text, polished):
                logger.warning(
                    f"Batch {batch_idx}: polish changed numbers, rejecting"
                )
                return repair_hebrew_ocr_text(original_text)

            return repair_hebrew_ocr_text(polished)

        except Exception as e:
            logger.error(f"Batch {batch_idx} polish failed: {e}")
            return repair_hebrew_ocr_text(original_text)

    def _call_gemini_text(
        self,
        gemini_client,
        model_name: str,
        prompt: str,
        text: str,
    ) -> str:
        """Call Gemini with text content (not PDF)."""
        from google.genai import types

        response = gemini_client.client.models.generate_content(
            model=model_name,
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
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            ),
        )
        return extract_gemini_text_response(response)

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
        contents = [self._render_result(result) for result in page_results]
        return combine_markdown_sections(contents, empty_message="")

    def _render_result(self, result: PageExtractionResult) -> str:
        """Render one result with its page header."""
        return self._build_header(result) + repair_hebrew_ocr_text(result.content)

    def _build_header(self, result: PageExtractionResult) -> str:
        """Build a markdown header for a single page or multi-page range."""
        start_page, end_page = result.page_range
        if start_page == end_page:
            return f"## Page {start_page + 1}\n\n"
        return f"## Pages {start_page + 1}-{end_page + 1}\n\n"
