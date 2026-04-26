"""
Extraction router for the retained page-aware pipeline.

Routes digital pages to pdfplumber and OCR pages to Mistral, while preserving
secondary digital text for mixed-page validation.
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF

from src.core.config import settings
from src.core.logging_utils import build_log_extra, log_event
from src.core.utils import normalize_hebrew_text, repair_hebrew_ocr_text
from src.services.mistral_client import MistralDocumentClient
from src.services.page_analyzer import PageAnalysis, PageType

logger = logging.getLogger(__name__)


@dataclass
class PageExtractionResult:
    """Result from extracting a single page."""

    page_number: int
    content: str
    source: str
    alternative_content: Optional[str] = None
    confidence: float = 1.0
    strategy: str = "text_extraction"
    ocr_mode: Optional[str] = None
    validation_source: Optional[str] = None
    page_range: Optional[tuple[int, int]] = None

    def __post_init__(self):
        self.content = repair_hebrew_ocr_text(self.content)
        if self.alternative_content is not None:
            self.alternative_content = repair_hebrew_ocr_text(self.alternative_content)
        if self.page_range is None:
            self.page_range = (self.page_number, self.page_number)

    def to_metadata(self) -> dict:
        metadata = {
            "page_number": self.page_number + 1,
            "page_range": [self.page_range[0] + 1, self.page_range[1] + 1],
            "strategy": self.strategy,
            "source": self.source,
            "confidence": round(self.confidence, 3),
        }
        if self.ocr_mode:
            metadata["ocr_mode"] = self.ocr_mode
        if self.validation_source:
            metadata["validation_source"] = self.validation_source
        return metadata


class ExtractionRouter:
    """Routes pages to retained extraction methods based on page analysis."""

    def __init__(self, mistral_client: MistralDocumentClient):
        self._mistral_client = mistral_client

    async def extract_pages(
        self,
        pdf_path: str,
        analyses: list[PageAnalysis],
        pdf_bytes: Optional[bytes] = None,
    ) -> list[PageExtractionResult]:
        text_pages = [a for a in analyses if a.page_type == PageType.TEXT_ONLY]
        mixed_pages = [a for a in analyses if a.page_type == PageType.IMAGE_WITH_TEXT]
        image_pages = [
            a for a in analyses if a.page_type in {PageType.IMAGE_ONLY, PageType.EMPTY}
        ]

        log_event(
            logger,
            logging.INFO,
            "page_routing_completed",
            total_pages=len(analyses),
            text_pages=len(text_pages),
            mixed_pages=len(mixed_pages),
            image_only_pages=len(image_pages),
        )

        if pdf_bytes is None:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

        text_results, mixed_results, ocr_results = await asyncio.gather(
            self._extract_text_pages(pdf_path, text_pages),
            self._extract_mixed_pages(pdf_path, pdf_bytes, mixed_pages),
            self._extract_ocr_pages(pdf_bytes, image_pages),
        )

        all_results = text_results + mixed_results + ocr_results
        all_results.sort(key=lambda result: result.page_number)
        return all_results

    async def _extract_text_pages(
        self,
        pdf_path: str,
        pages: list[PageAnalysis],
    ) -> list[PageExtractionResult]:
        if not pages:
            return []
        return await asyncio.to_thread(self._pdfplumber_extract, pdf_path, pages)

    def _pdfplumber_extract(
        self,
        pdf_path: str,
        pages: list[PageAnalysis],
    ) -> list[PageExtractionResult]:
        import pdfplumber

        results: list[PageExtractionResult] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for analysis in pages:
                    idx = analysis.page_number
                    if idx >= len(pdf.pages):
                        continue

                    page = pdf.pages[idx]
                    parts = []

                    text = normalize_hebrew_text(page.extract_text() or "")
                    if text.strip():
                        parts.append(text.strip())

                    tables = page.extract_tables()
                    if tables:
                        for table_data in tables:
                            markdown = self._table_to_markdown(table_data)
                            if markdown:
                                parts.append(markdown)

                    content = "\n\n".join(parts) if parts else ""
                    results.append(
                        PageExtractionResult(
                            page_number=idx,
                            content=content,
                            source="pdfplumber",
                            confidence=0.9 if content else 0.1,
                            strategy="text_extraction",
                        )
                    )
        except Exception as exc:
            logger.exception(
                "text_page_extraction_failed",
                extra=build_log_extra(
                    page_count=len(pages),
                    error_type=type(exc).__name__,
                    reason=str(exc),
                ),
            )
            for analysis in pages:
                results.append(
                    PageExtractionResult(
                        page_number=analysis.page_number,
                        content="",
                        source="pdfplumber",
                        confidence=0.0,
                        strategy="text_extraction",
                    )
                )

        return results

    def _table_to_markdown(self, table_data: list[list]) -> str:
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

        separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
        return rows[0] + "\n" + separator + "\n" + "\n".join(rows[1:])

    async def _extract_mixed_pages(
        self,
        pdf_path: str,
        pdf_bytes: bytes,
        pages: list[PageAnalysis],
    ) -> list[PageExtractionResult]:
        if not pages:
            return []

        primary_results = await self._extract_ocr_pages(pdf_bytes, pages)
        alt_results = await self._extract_text_pages(pdf_path, pages)
        alt_by_page = {result.page_number: result.content for result in alt_results}

        for result in primary_results:
            result.alternative_content = alt_by_page.get(result.page_number, "")
            result.validation_source = "text_extraction"

        return primary_results

    async def _extract_ocr_pages(
        self,
        pdf_bytes: bytes,
        pages: list[PageAnalysis],
    ) -> list[PageExtractionResult]:
        if not pages:
            return []

        batch_size = settings.SMART_EXTRACTION_MISTRAL_BATCH_SIZE
        sorted_pages = sorted(pages, key=lambda analysis: analysis.page_number)
        batches = self._create_consecutive_batches(sorted_pages, batch_size)

        log_event(
            logger,
            logging.INFO,
            "ocr_batches_created",
            engine="mistral",
            page_count=len(sorted_pages),
            batch_count=len(batches),
        )

        semaphore = asyncio.Semaphore(3)

        async def process_batch(batch: list[PageAnalysis]) -> list[PageExtractionResult]:
            async with semaphore:
                return await self._process_mistral_batch(pdf_bytes, batch)

        batch_results = await asyncio.gather(
            *[process_batch(batch) for batch in batches]
        )

        results: list[PageExtractionResult] = []
        for batch_result in batch_results:
            results.extend(batch_result)
        return results

    def _create_consecutive_batches(
        self,
        sorted_pages: list[PageAnalysis],
        max_batch_size: int,
    ) -> list[list[PageAnalysis]]:
        if not sorted_pages:
            return []

        batches = []
        current_batch = [sorted_pages[0]]

        for index in range(1, len(sorted_pages)):
            previous = sorted_pages[index - 1].page_number
            current = sorted_pages[index].page_number

            if current != previous + 1 or len(current_batch) >= max_batch_size:
                batches.append(current_batch)
                current_batch = [sorted_pages[index]]
            else:
                current_batch.append(sorted_pages[index])

        batches.append(current_batch)
        return batches

    async def _process_mistral_batch(
        self,
        pdf_bytes: bytes,
        batch: list[PageAnalysis],
    ) -> list[PageExtractionResult]:
        page_numbers = [analysis.page_number for analysis in batch]
        from_page = page_numbers[0]
        to_page = page_numbers[-1]

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mini_pdf = fitz.open()
        mini_pdf.insert_pdf(doc, from_page=from_page, to_page=to_page)
        mini_pdf_bytes = mini_pdf.tobytes()
        mini_pdf.close()
        doc.close()

        results: list[PageExtractionResult] = []
        try:
            import base64

            mini_base64 = base64.b64encode(mini_pdf_bytes).decode("utf-8")
            content, _ = await self._mistral_client.process_document(
                pdf_base64=mini_base64,
                pdf_bytes=mini_pdf_bytes,
                enable_validation=False,
            )
            page_contents = self._split_mistral_output(content, len(batch))

            for index, analysis in enumerate(batch):
                page_content = page_contents[index] if index < len(page_contents) else ""
                results.append(
                    PageExtractionResult(
                        page_number=analysis.page_number,
                        content=page_content,
                        source="mistral",
                        confidence=0.85 if page_content.strip() else 0.1,
                        strategy="ocr",
                        ocr_mode="all_around",
                    )
                )
        except Exception as exc:
            logger.exception(
                "ocr_batch_failed",
                extra=build_log_extra(
                    engine="mistral",
                    from_page=from_page + 1,
                    to_page=to_page + 1,
                    error_type=type(exc).__name__,
                    reason=str(exc),
                ),
            )
            for analysis in batch:
                results.append(
                    PageExtractionResult(
                        page_number=analysis.page_number,
                        content="",
                        source="mistral",
                        confidence=0.0,
                        strategy="ocr",
                        ocr_mode="all_around",
                    )
                )

        return results

    def _split_mistral_output(self, content: str, expected_pages: int) -> list[str]:
        import re

        parts = re.split(r"(?=^# Page \d+)", content, flags=re.MULTILINE)
        parts = [part.strip() for part in parts if part.strip()]
        if len(parts) == expected_pages:
            return parts

        parts = re.split(r"\n---\n", content)
        parts = [part.strip() for part in parts if part.strip()]
        if len(parts) == expected_pages:
            return parts

        if expected_pages == 1:
            return [content]
        return [content] + [""] * (expected_pages - 1)
