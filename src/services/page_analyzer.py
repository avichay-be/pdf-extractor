"""
Page-level content analysis for smart extraction routing.

Classifies each PDF page by content type (text, image, mixed, empty)
using PyMuPDF and pdfplumber to determine the optimal extraction method.
"""
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF

from src.core.config import settings

logger = logging.getLogger(__name__)

# Minimum image dimensions to be considered meaningful (filters tiny icons)
MIN_IMAGE_WIDTH = 50
MIN_IMAGE_HEIGHT = 50

# Image must cover this fraction of page area to be a "content image" (photo, scan, chart)
# Images below this ratio are treated as decorative (logos, headers, stamps)
MIN_IMAGE_AREA_RATIO = 0.30


class PageType(Enum):
    """Classification of a PDF page's content type."""
    TEXT_ONLY = "text_only"
    IMAGE_WITH_TEXT = "image_with_text"
    IMAGE_ONLY = "image_only"
    EMPTY = "empty"


@dataclass
class PageAnalysis:
    """Analysis result for a single PDF page."""
    page_number: int       # 0-based
    page_type: PageType
    text_length: int       # length of embedded text
    image_count: int       # number of content images (area ratio >= 30%)
    has_tables: bool       # pdfplumber detected table structures
    text_preview: str      # first 200 chars (for debug logging)
    max_image_area_ratio: float = 0.0  # largest image area ratio on this page


class PageAnalyzer:
    """Analyzes PDF pages to classify content type for optimal extraction routing."""

    def __init__(self, text_threshold: Optional[int] = None):
        """
        Initialize page analyzer.

        Args:
            text_threshold: Minimum character count to consider a page as having text.
                           Defaults to settings.SMART_EXTRACTION_TEXT_THRESHOLD.
        """
        self.text_threshold = text_threshold or settings.SMART_EXTRACTION_TEXT_THRESHOLD

    def analyze_pdf(self, pdf_path: str) -> list[PageAnalysis]:
        """
        Analyze all pages in a PDF and classify each by content type.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of PageAnalysis objects, one per page
        """
        results = []
        doc = fitz.open(pdf_path)

        # Optional pdfplumber table detection
        table_pages = self._detect_table_pages(pdf_path)

        try:
            for page_idx in range(len(doc)):
                page = doc.load_page(page_idx)
                analysis = self._analyze_page(page, page_idx, table_pages)
                results.append(analysis)
        finally:
            doc.close()

        # Log summary
        type_counts = {}
        for r in results:
            type_counts[r.page_type.value] = type_counts.get(r.page_type.value, 0) + 1
        logger.info(f"Page analysis complete: {len(results)} pages - {type_counts}")

        return results

    def quick_is_mixed(self, pdf_path: str) -> bool:
        """
        Fast check to determine if a PDF has mixed content types.

        Scans first 20 pages (or all if fewer). Returns True if at least
        one page is TEXT_ONLY and at least one is IMAGE_ONLY or IMAGE_WITH_TEXT.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            True if PDF has mixed content requiring smart extraction
        """
        doc = fitz.open(pdf_path)
        max_pages = min(len(doc), 20)

        has_text_dominant = False
        has_image_dominant = False

        try:
            for page_idx in range(max_pages):
                page = doc.load_page(page_idx)
                text = page.get_text("text").strip()
                text_length = len(text)
                content_image_count, _ = self._count_meaningful_images(page)

                if text_length >= self.text_threshold and content_image_count == 0:
                    has_text_dominant = True
                elif content_image_count > 0:
                    has_image_dominant = True

                # Early exit if we already found both types
                if has_text_dominant and has_image_dominant:
                    logger.info(
                        f"Mixed content detected at page {page_idx + 1}: "
                        f"has text-dominant and image-dominant pages"
                    )
                    return True
        finally:
            doc.close()

        logger.info(
            f"Auto-detect: PDF is not mixed content "
            f"(text_dominant={has_text_dominant}, image_dominant={has_image_dominant}, "
            f"pages_checked={max_pages})"
        )
        return False

    def _analyze_page(
        self,
        page: fitz.Page,
        page_idx: int,
        table_pages: set[int]
    ) -> PageAnalysis:
        """
        Classify a single page by content type.

        Args:
            page: PyMuPDF page object
            page_idx: 0-based page index
            table_pages: Set of page indices that contain tables (from pdfplumber)

        Returns:
            PageAnalysis for this page
        """
        text = page.get_text("text").strip()
        text_length = len(text)
        image_count, max_image_area_ratio = self._count_meaningful_images(page)
        has_tables = page_idx in table_pages
        text_preview = text[:200] if text else ""

        # Classification logic
        has_text = text_length >= self.text_threshold

        if not has_text and image_count == 0:
            page_type = PageType.EMPTY
        elif has_text and image_count == 0:
            page_type = PageType.TEXT_ONLY
        elif has_text and image_count > 0:
            page_type = PageType.IMAGE_WITH_TEXT
        else:  # not has_text and image_count > 0
            page_type = PageType.IMAGE_ONLY

        return PageAnalysis(
            page_number=page_idx,
            page_type=page_type,
            text_length=text_length,
            image_count=image_count,
            has_tables=has_tables,
            text_preview=text_preview,
            max_image_area_ratio=max_image_area_ratio,
        )

    def _count_meaningful_images(self, page: fitz.Page) -> tuple[int, float]:
        """
        Count content images on a page using area-ratio filtering.

        An image is considered "content" (photo, scan, chart) only if it
        covers >= MIN_IMAGE_AREA_RATIO (30%) of the page area. Smaller
        images (logos, headers, stamps) are treated as decorative.

        Args:
            page: PyMuPDF page object

        Returns:
            Tuple of (content_image_count, max_image_area_ratio)
        """
        images = page.get_images(full=True)
        page_area = page.rect.width * page.rect.height
        if page_area == 0:
            return 0, 0.0

        count = 0
        max_ratio = 0.0
        seen_xrefs = set()

        for img in images:
            # img tuple: (xref, smask, width, height, bpc, colorspace, ...)
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            width = img[2]
            height = img[3]
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            # Use rendered size on page for accurate area ratio
            try:
                rects = page.get_image_rects(xref)
            except Exception:
                continue

            for rect in rects:
                if rect.is_empty or rect.is_infinite:
                    continue
                ratio = (rect.width * rect.height) / page_area
                max_ratio = max(max_ratio, ratio)
                if ratio >= MIN_IMAGE_AREA_RATIO:
                    count += 1
                    break  # Count each image once

        return count, max_ratio

    def _detect_table_pages(self, pdf_path: str) -> set[int]:
        """
        Use pdfplumber to detect which pages contain tables.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Set of 0-based page indices that contain tables
        """
        table_pages = set()
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    if tables:
                        table_pages.add(i)
        except Exception as e:
            logger.warning(f"pdfplumber table detection failed, continuing without: {e}")
        return table_pages
