"""
Tests for PageAnalyzer - PDF page content classification.
"""
import os
import tempfile
import pytest
import fitz  # PyMuPDF

from src.services.page_analyzer import PageAnalyzer, PageType, PageAnalysis, MIN_IMAGE_AREA_RATIO


# --- Fixtures ---

@pytest.fixture
def analyzer():
    """Create a PageAnalyzer with default settings."""
    return PageAnalyzer(text_threshold=50)


@pytest.fixture
def text_only_pdf(tmp_path):
    """Create a PDF with only embedded text (no images)."""
    pdf_path = str(tmp_path / "text_only.pdf")
    doc = fitz.open()
    page = doc.new_page()
    # Insert enough text to exceed threshold
    text = "This is a financial report with substantial text content. " * 10
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def image_only_pdf(tmp_path):
    """Create a PDF with only images (no text).

    Image covers >30% of page area to be classified as content image.
    """
    pdf_path = str(tmp_path / "image_only.pdf")
    doc = fitz.open()
    page = doc.new_page()  # Default 595x842
    # Insert image covering ~70% of page area (content image, not decorative)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), 1)
    pix.set_rect(pix.irect, (255, 0, 0, 255))
    page.insert_image(fitz.Rect(50, 50, 550, 750), pixmap=pix)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def mixed_content_pdf(tmp_path):
    """Create a PDF with text pages and image pages.

    Images cover >30% of page area to be classified as content images.
    """
    pdf_path = str(tmp_path / "mixed.pdf")
    doc = fitz.open()

    # Page 0: text only
    page0 = doc.new_page()
    text = "Financial statement data with numbers 12345.67 and dates 01/01/2025. " * 5
    page0.insert_text((72, 72), text, fontsize=12)

    # Page 1: image only (no text) — image covers ~70% of page
    page1 = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200), 1)
    pix.set_rect(pix.irect, (0, 0, 255, 255))
    page1.insert_image(fitz.Rect(50, 50, 550, 750), pixmap=pix)

    # Page 2: text + large image (~45% of page)
    page2 = doc.new_page()
    page2.insert_text((72, 72), "Revenue breakdown for Q4 2025 with detailed analysis. " * 5, fontsize=12)
    pix2 = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 150, 150), 1)
    pix2.set_rect(pix2.irect, (0, 255, 0, 255))
    page2.insert_image(fitz.Rect(50, 300, 550, 750), pixmap=pix2)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def empty_pdf(tmp_path):
    """Create a PDF with an empty page."""
    pdf_path = str(tmp_path / "empty.pdf")
    doc = fitz.open()
    doc.new_page()  # Empty page
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def multi_text_pdf(tmp_path):
    """Create a PDF with multiple text-only pages (no images)."""
    pdf_path = str(tmp_path / "multi_text.pdf")
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        text = f"Page {i+1}: Balance sheet item {i+1} with value {1000 * (i+1):,}. " * 5
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


# --- PageAnalyzer.analyze_pdf tests ---

class TestAnalyzePdf:
    """Tests for full page analysis."""

    def test_text_only_page(self, analyzer, text_only_pdf):
        """Text-only pages should be classified as TEXT_ONLY."""
        results = analyzer.analyze_pdf(text_only_pdf)
        assert len(results) == 1
        assert results[0].page_type == PageType.TEXT_ONLY
        assert results[0].page_number == 0
        assert results[0].text_length >= 50
        assert results[0].image_count == 0

    def test_image_only_page(self, analyzer, image_only_pdf):
        """Image-only pages (no text) should be classified as IMAGE_ONLY."""
        results = analyzer.analyze_pdf(image_only_pdf)
        assert len(results) == 1
        assert results[0].page_type == PageType.IMAGE_ONLY
        assert results[0].text_length < 50
        assert results[0].image_count >= 1

    def test_empty_page(self, analyzer, empty_pdf):
        """Empty pages should be classified as EMPTY."""
        results = analyzer.analyze_pdf(empty_pdf)
        assert len(results) == 1
        assert results[0].page_type == PageType.EMPTY
        assert results[0].text_length < 50
        assert results[0].image_count == 0

    def test_mixed_content_pdf(self, analyzer, mixed_content_pdf):
        """Mixed PDFs should have diverse page classifications."""
        results = analyzer.analyze_pdf(mixed_content_pdf)
        assert len(results) == 3

        # Page 0: text only
        assert results[0].page_type == PageType.TEXT_ONLY

        # Page 1: image only
        assert results[1].page_type == PageType.IMAGE_ONLY

        # Page 2: image with text
        assert results[2].page_type == PageType.IMAGE_WITH_TEXT

    def test_analysis_returns_correct_page_numbers(self, analyzer, mixed_content_pdf):
        """Page numbers should be 0-based and sequential."""
        results = analyzer.analyze_pdf(mixed_content_pdf)
        for i, result in enumerate(results):
            assert result.page_number == i

    def test_text_preview_truncated(self, analyzer, text_only_pdf):
        """Text preview should be at most 200 chars."""
        results = analyzer.analyze_pdf(text_only_pdf)
        assert len(results[0].text_preview) <= 200

    def test_multi_page_text_pdf(self, analyzer, multi_text_pdf):
        """All pages in a text-only PDF should be TEXT_ONLY."""
        results = analyzer.analyze_pdf(multi_text_pdf)
        assert len(results) == 5
        for result in results:
            assert result.page_type == PageType.TEXT_ONLY


# --- PageAnalyzer.quick_is_mixed tests ---

class TestQuickIsMixed:
    """Tests for the fast mixed-content detection."""

    def test_mixed_pdf_detected(self, analyzer, mixed_content_pdf):
        """Mixed PDFs should be detected."""
        assert analyzer.quick_is_mixed(mixed_content_pdf) is True

    def test_text_only_not_mixed(self, analyzer, text_only_pdf):
        """Text-only PDFs should not be detected as mixed."""
        assert analyzer.quick_is_mixed(text_only_pdf) is False

    def test_multi_text_not_mixed(self, analyzer, multi_text_pdf):
        """PDFs with only text pages should not be detected as mixed."""
        assert analyzer.quick_is_mixed(multi_text_pdf) is False

    def test_image_only_not_mixed(self, analyzer, image_only_pdf):
        """PDFs with only image pages should not be detected as mixed."""
        assert analyzer.quick_is_mixed(image_only_pdf) is False

    def test_empty_pdf_not_mixed(self, analyzer, empty_pdf):
        """Empty PDFs should not be detected as mixed."""
        assert analyzer.quick_is_mixed(empty_pdf) is False


# --- PageType classification logic ---

class TestPageTypeClassification:
    """Tests for individual page classification logic."""

    def test_text_threshold_configurable(self, text_only_pdf):
        """Text threshold should be configurable."""
        strict_analyzer = PageAnalyzer(text_threshold=100000)
        results = strict_analyzer.analyze_pdf(text_only_pdf)
        # With very high threshold, text page becomes IMAGE_ONLY or EMPTY
        assert results[0].page_type in (PageType.EMPTY, PageType.IMAGE_ONLY)

    def test_low_text_threshold(self, text_only_pdf):
        """Low threshold should detect more text pages."""
        lenient_analyzer = PageAnalyzer(text_threshold=1)
        results = lenient_analyzer.analyze_pdf(text_only_pdf)
        assert results[0].page_type == PageType.TEXT_ONLY

    def test_small_images_filtered(self, tmp_path):
        """Images smaller than 50x50 should be filtered out."""
        pdf_path = str(tmp_path / "small_img.pdf")
        doc = fitz.open()
        page = doc.new_page()
        # Insert a tiny image (10x10)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), 1)
        pix.set_rect(pix.irect, (255, 0, 0, 255))
        page.insert_image(fitz.Rect(72, 72, 82, 82), pixmap=pix)
        doc.save(pdf_path)
        doc.close()

        analyzer = PageAnalyzer(text_threshold=50)
        results = analyzer.analyze_pdf(pdf_path)
        # Small image should be filtered, page should be EMPTY (no text, no meaningful images)
        assert results[0].page_type == PageType.EMPTY
        assert results[0].image_count == 0


# --- Edge cases ---

class TestEdgeCases:
    """Edge case tests."""

    def test_single_char_text_below_threshold(self, tmp_path):
        """Page with text below threshold and no images -> EMPTY."""
        pdf_path = str(tmp_path / "short.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hi", fontsize=12)
        doc.save(pdf_path)
        doc.close()

        analyzer = PageAnalyzer(text_threshold=50)
        results = analyzer.analyze_pdf(pdf_path)
        assert results[0].page_type == PageType.EMPTY

    def test_has_tables_flag(self, tmp_path):
        """has_tables should be set when pdfplumber detects tables."""
        # This is hard to test without a real table PDF, but we can verify
        # the flag is at least set to False for a simple text page
        pdf_path = str(tmp_path / "no_tables.pdf")
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Just text, no tables here. " * 10, fontsize=12)
        doc.save(pdf_path)
        doc.close()

        analyzer = PageAnalyzer(text_threshold=50)
        results = analyzer.analyze_pdf(pdf_path)
        assert results[0].has_tables is False


# --- Area ratio filtering tests ---

class TestAreaRatioFiltering:
    """Tests for area-ratio-based image classification."""

    @pytest.fixture
    def text_with_header_logo_pdf(self, tmp_path):
        """Create a PDF where every page has text and a small header logo.

        The logo covers ~2% of page area — should be filtered as decorative.
        """
        pdf_path = str(tmp_path / "text_with_logo.pdf")
        doc = fitz.open()
        logo_pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 50), 1)
        logo_pix.set_rect(logo_pix.irect, (100, 100, 100, 255))

        for i in range(5):
            page = doc.new_page()
            # Small header logo: 200x50 points on a 595x842 page = ~2% area
            page.insert_image(fitz.Rect(10, 10, 210, 60), pixmap=logo_pix)
            text = f"Financial data page {i+1}. Revenue: ${1000*(i+1):,}. " * 10
            page.insert_text((72, 100), text, fontsize=12)

        doc.save(pdf_path)
        doc.close()
        return pdf_path

    def test_header_logo_filtered_as_decorative(self, analyzer, text_with_header_logo_pdf):
        """Pages with only a small header logo should be classified as TEXT_ONLY."""
        results = analyzer.analyze_pdf(text_with_header_logo_pdf)
        assert len(results) == 5
        for result in results:
            assert result.page_type == PageType.TEXT_ONLY
            assert result.image_count == 0
            assert result.max_image_area_ratio < MIN_IMAGE_AREA_RATIO

    def test_header_logo_pdf_not_mixed(self, analyzer, text_with_header_logo_pdf):
        """PDF with text + decorative logos on every page should NOT be mixed."""
        assert analyzer.quick_is_mixed(text_with_header_logo_pdf) is False

    def test_large_image_counted_as_content(self, analyzer, tmp_path):
        """An image covering >30% of the page should be counted as content."""
        pdf_path = str(tmp_path / "large_img.pdf")
        doc = fitz.open()
        page = doc.new_page()  # 595x842
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 100, 100), 1)
        pix.set_rect(pix.irect, (255, 0, 0, 255))
        # Image covers ~70% of page
        page.insert_image(fitz.Rect(50, 50, 550, 750), pixmap=pix)
        doc.save(pdf_path)
        doc.close()

        results = analyzer.analyze_pdf(pdf_path)
        assert results[0].image_count == 1
        assert results[0].max_image_area_ratio >= MIN_IMAGE_AREA_RATIO

    def test_medium_decorative_image_filtered(self, analyzer, tmp_path):
        """A medium image (e.g. 200x100 rendered) covering <30% should be filtered."""
        pdf_path = str(tmp_path / "medium_img.pdf")
        doc = fitz.open()
        page = doc.new_page()  # 595x842
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 100), 1)
        pix.set_rect(pix.irect, (0, 128, 0, 255))
        # Rendered at 200x100 points = 20,000 / 500,990 = ~4% of page
        page.insert_image(fitz.Rect(50, 50, 250, 150), pixmap=pix)
        text = "This page has text and a small decorative banner image. " * 10
        page.insert_text((72, 200), text, fontsize=12)
        doc.save(pdf_path)
        doc.close()

        results = analyzer.analyze_pdf(pdf_path)
        assert results[0].page_type == PageType.TEXT_ONLY
        assert results[0].image_count == 0
        assert results[0].max_image_area_ratio < MIN_IMAGE_AREA_RATIO

    def test_max_image_area_ratio_populated(self, analyzer, image_only_pdf):
        """max_image_area_ratio should reflect the largest image on the page."""
        results = analyzer.analyze_pdf(image_only_pdf)
        assert results[0].max_image_area_ratio > 0.0
        assert results[0].max_image_area_ratio >= MIN_IMAGE_AREA_RATIO
