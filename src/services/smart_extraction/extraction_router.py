"""
Extraction router for smart page-level PDF processing.

Routes pages to the optimal extraction method based on PageAnalysis results
and collects extraction output for cross-validation and polishing.
"""
import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

from src.core.config import settings
from src.core.utils import format_page_header, normalize_hebrew_text
from src.services.page_analyzer import PageAnalysis, PageType
from src.services.client_factory import get_client_factory

logger = logging.getLogger(__name__)


@dataclass
class PageExtractionResult:
    """Result from extracting a single page."""
    page_number: int
    content: str                           # Extracted markdown
    source: str                            # "mistral", "pdfplumber", "gemini", "merged"
    alternative_content: Optional[str] = None  # Second source for cross-validation
    confidence: float = 1.0                # 0-1 score


class ExtractionRouter:
    """Routes pages to appropriate extraction methods based on content analysis."""

    def __init__(self):
        self._client_factory = get_client_factory()

    async def extract_pages(
        self,
        pdf_path: str,
        analyses: list[PageAnalysis]
    ) -> list[PageExtractionResult]:
        """
        Extract content from all pages using the best method for each.

        Routes pages by type:
        - TEXT_ONLY -> pdfplumber (text + tables)
        - IMAGE_WITH_TEXT -> Mistral OCR (primary) + pdfplumber (secondary)
        - IMAGE_ONLY -> Mistral OCR (batched)
        - EMPTY -> Gemini (multimodal fallback)

        Args:
            pdf_path: Path to the PDF file
            analyses: List of PageAnalysis from PageAnalyzer

        Returns:
            List of PageExtractionResult, one per page, sorted by page_number
        """
        # Group pages by type
        text_pages = [a for a in analyses if a.page_type == PageType.TEXT_ONLY]
        image_with_text_pages = [a for a in analyses if a.page_type == PageType.IMAGE_WITH_TEXT]
        image_only_pages = [a for a in analyses if a.page_type == PageType.IMAGE_ONLY]
        empty_pages = [a for a in analyses if a.page_type == PageType.EMPTY]

        logger.info(
            f"Routing {len(analyses)} pages: "
            f"text={len(text_pages)}, image+text={len(image_with_text_pages)}, "
            f"image_only={len(image_only_pages)}, empty={len(empty_pages)}"
        )

        # Read PDF bytes once for all extraction methods
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        # All image pages (IMAGE_ONLY + IMAGE_WITH_TEXT) go to Mistral
        all_image_pages = image_only_pages + image_with_text_pages

        # Execute all extraction types in parallel
        tasks = []
        tasks.append(self._extract_text_pages(pdf_path, text_pages))
        tasks.append(self._extract_image_pages(pdf_path, pdf_bytes, all_image_pages))
        tasks.append(self._extract_empty_pages(pdf_bytes, empty_pages))

        text_results, image_results, empty_results = await asyncio.gather(*tasks)

        # For IMAGE_WITH_TEXT pages, also extract pdfplumber text as alternative
        iwt_page_numbers = {a.page_number for a in image_with_text_pages}
        if iwt_page_numbers:
            pdfplumber_alt = await self._extract_text_pages(
                pdf_path, image_with_text_pages
            )
            # Merge alternative content into image results
            alt_by_page = {r.page_number: r.content for r in pdfplumber_alt}
            for result in image_results:
                if result.page_number in iwt_page_numbers:
                    result.alternative_content = alt_by_page.get(result.page_number)

        # Combine all results and sort by page number
        all_results = text_results + image_results + empty_results
        all_results.sort(key=lambda r: r.page_number)

        return all_results

    async def _extract_text_pages(
        self,
        pdf_path: str,
        pages: list[PageAnalysis]
    ) -> list[PageExtractionResult]:
        """Extract text pages using pdfplumber (text + tables)."""
        if not pages:
            return []

        results = await asyncio.to_thread(
            self._pdfplumber_extract, pdf_path, pages
        )
        return results

    def _pdfplumber_extract(
        self,
        pdf_path: str,
        pages: list[PageAnalysis]
    ) -> list[PageExtractionResult]:
        """
        Synchronous pdfplumber extraction for specified pages.

        Extracts both full text and tables, combining them into markdown.
        """
        import pdfplumber

        results = []
        page_indices = {a.page_number for a in pages}

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for analysis in pages:
                    idx = analysis.page_number
                    if idx >= len(pdf.pages):
                        continue

                    page = pdf.pages[idx]
                    parts = []

                    # Extract full text
                    text = normalize_hebrew_text(page.extract_text() or "")
                    if text.strip():
                        parts.append(text.strip())

                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table_data in tables:
                            md_table = self._table_to_markdown(table_data)
                            if md_table:
                                parts.append(md_table)

                    content = "\n\n".join(parts) if parts else ""

                    results.append(PageExtractionResult(
                        page_number=idx,
                        content=content,
                        source="pdfplumber",
                        confidence=0.9 if content else 0.1,
                    ))
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            # Return empty results for failed pages
            for analysis in pages:
                results.append(PageExtractionResult(
                    page_number=analysis.page_number,
                    content="",
                    source="pdfplumber",
                    confidence=0.0,
                ))

        return results

    def _table_to_markdown(self, table_data: list[list]) -> str:
        """Convert pdfplumber table data to markdown table format."""
        if not table_data or len(table_data) < 2:
            return ""

        rows = []
        for row in table_data:
            cells = [
                normalize_hebrew_text(str(cell).strip()) if cell else ""
                for cell in row
            ]
            rows.append("| " + " | ".join(cells) + " |")

        if len(rows) < 2:
            return ""

        # Insert separator after header
        col_count = len(table_data[0])
        separator = "| " + " | ".join(["---"] * col_count) + " |"

        return rows[0] + "\n" + separator + "\n" + "\n".join(rows[1:])

    async def _extract_image_pages(
        self,
        pdf_path: str,
        pdf_bytes: bytes,
        pages: list[PageAnalysis]
    ) -> list[PageExtractionResult]:
        """Extract image pages using Mistral OCR with batching."""
        if not pages:
            return []

        mistral_client = self._client_factory.mistral_client
        batch_size = settings.SMART_EXTRACTION_MISTRAL_BATCH_SIZE

        # Sort by page number for consecutive batching
        sorted_pages = sorted(pages, key=lambda a: a.page_number)

        # Group consecutive pages into batches
        batches = self._create_consecutive_batches(sorted_pages, batch_size)

        logger.info(
            f"Processing {len(sorted_pages)} image pages in "
            f"{len(batches)} Mistral batches"
        )

        # Process batches with concurrency limit
        semaphore = asyncio.Semaphore(3)
        results = []

        async def process_batch(batch: list[PageAnalysis]) -> list[PageExtractionResult]:
            async with semaphore:
                return await self._process_mistral_batch(
                    pdf_bytes, batch, mistral_client
                )

        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)

        for batch_result in batch_results:
            results.extend(batch_result)

        return results

    def _create_consecutive_batches(
        self,
        sorted_pages: list[PageAnalysis],
        max_batch_size: int
    ) -> list[list[PageAnalysis]]:
        """Group pages into batches of consecutive pages."""
        if not sorted_pages:
            return []

        batches = []
        current_batch = [sorted_pages[0]]

        for i in range(1, len(sorted_pages)):
            prev_num = sorted_pages[i - 1].page_number
            curr_num = sorted_pages[i].page_number

            # Start new batch if not consecutive or batch is full
            if curr_num != prev_num + 1 or len(current_batch) >= max_batch_size:
                batches.append(current_batch)
                current_batch = [sorted_pages[i]]
            else:
                current_batch.append(sorted_pages[i])

        batches.append(current_batch)
        return batches

    async def _process_mistral_batch(
        self,
        pdf_bytes: bytes,
        batch: list[PageAnalysis],
        mistral_client
    ) -> list[PageExtractionResult]:
        """
        Process a batch of consecutive pages through Mistral OCR.

        Creates a mini-PDF containing only the batch pages, sends to Mistral,
        then maps results back to original page numbers.
        """
        page_numbers = [a.page_number for a in batch]
        from_page = page_numbers[0]
        to_page = page_numbers[-1]

        # Create mini-PDF for batch
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mini_pdf = fitz.open()
        mini_pdf.insert_pdf(doc, from_page=from_page, to_page=to_page)
        mini_pdf_bytes = mini_pdf.tobytes()
        mini_pdf.close()
        doc.close()

        # Write mini-PDF to temp file for Mistral
        results = []
        tmp_path = None
        try:
            import base64
            mini_base64 = base64.b64encode(mini_pdf_bytes).decode('utf-8')

            content, _ = await mistral_client.process_document(
                pdf_base64=mini_base64,
                enable_validation=False,
            )

            # Split content by page headers (Mistral returns per-page markdown)
            page_contents = self._split_mistral_output(content, len(batch))

            for i, analysis in enumerate(batch):
                page_content = page_contents[i] if i < len(page_contents) else ""
                results.append(PageExtractionResult(
                    page_number=analysis.page_number,
                    content=page_content,
                    source="mistral",
                    confidence=0.85 if page_content.strip() else 0.1,
                ))

        except Exception as e:
            logger.error(
                f"Mistral batch failed for pages {from_page}-{to_page}: {e}"
            )
            for analysis in batch:
                results.append(PageExtractionResult(
                    page_number=analysis.page_number,
                    content="",
                    source="mistral",
                    confidence=0.0,
                ))

        return results

    def _split_mistral_output(self, content: str, expected_pages: int) -> list[str]:
        """
        Split Mistral's combined output into per-page content.

        Mistral typically separates pages with '# Page N' headers or '---'.
        """
        import re

        # Try splitting on page headers first
        parts = re.split(r'(?=^# Page \d+)', content, flags=re.MULTILINE)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == expected_pages:
            return parts

        # Fallback: split on horizontal rules
        parts = re.split(r'\n---\n', content)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) == expected_pages:
            return parts

        # Last resort: return whole content for first page, empty for rest
        if expected_pages == 1:
            return [content]

        # Distribute content evenly if we can't split properly
        return [content] + [""] * (expected_pages - 1)

    async def _extract_empty_pages(
        self,
        pdf_bytes: bytes,
        pages: list[PageAnalysis]
    ) -> list[PageExtractionResult]:
        """Extract empty pages using Gemini as multimodal fallback."""
        if not pages:
            return []

        gemini_client = self._client_factory.gemini_client
        if gemini_client is None:
            logger.warning("Gemini client unavailable, returning empty for empty pages")
            return [
                PageExtractionResult(
                    page_number=a.page_number,
                    content="",
                    source="gemini",
                    confidence=0.0,
                )
                for a in pages
            ]

        results = []
        for analysis in pages:
            try:
                content = await asyncio.to_thread(
                    gemini_client.extract_page_content,
                    pdf_bytes,
                    analysis.page_number,
                )
                results.append(PageExtractionResult(
                    page_number=analysis.page_number,
                    content=content,
                    source="gemini",
                    confidence=0.7 if content.strip() else 0.1,
                ))
            except Exception as e:
                logger.error(
                    f"Gemini fallback failed for page {analysis.page_number}: {e}"
                )
                results.append(PageExtractionResult(
                    page_number=analysis.page_number,
                    content="",
                    source="gemini",
                    confidence=0.0,
                ))

        return results
