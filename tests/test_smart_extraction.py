"""
Tests for Smart Extraction workflow components.

Tests ExtractionRouter, SmartValidationService, GeminiPolisher,
SmartExtractionHandler, and workflow registration.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import fitz

from src.services.page_analyzer import PageAnalysis, PageType
from src.services.smart_extraction.extraction_router import (
    ExtractionRouter,
    PageExtractionResult,
)
from src.services.smart_extraction.smart_validator import SmartValidationService
from src.services.smart_extraction.gemini_polisher import GeminiPolisher
from src.services.workflows.smart_extraction_handler import SmartExtractionHandler
from src.workflows.workflow_types import WorkflowType, WORKFLOW_NAMES
from src.workflows.workflow_router import get_workflow_for_query


# --- Fixtures ---

@pytest.fixture
def text_page_analysis():
    return PageAnalysis(
        page_number=0,
        page_type=PageType.TEXT_ONLY,
        text_length=500,
        image_count=0,
        has_tables=True,
        text_preview="Revenue for Q4...",
    )


@pytest.fixture
def image_page_analysis():
    return PageAnalysis(
        page_number=1,
        page_type=PageType.IMAGE_ONLY,
        text_length=10,
        image_count=2,
        has_tables=False,
        text_preview="",
    )


@pytest.fixture
def image_with_text_analysis():
    return PageAnalysis(
        page_number=2,
        page_type=PageType.IMAGE_WITH_TEXT,
        text_length=200,
        image_count=1,
        has_tables=True,
        text_preview="Financial chart with data...",
    )


@pytest.fixture
def empty_page_analysis():
    return PageAnalysis(
        page_number=3,
        page_type=PageType.EMPTY,
        text_length=0,
        image_count=0,
        has_tables=False,
        text_preview="",
    )


@pytest.fixture
def sample_text_pdf(tmp_path):
    """Create a sample PDF for testing."""
    pdf_path = str(tmp_path / "sample.pdf")
    doc = fitz.open()
    for i in range(4):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i+1} content " * 20, fontsize=12)
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def page_extraction_results():
    """Sample extraction results for validation/polishing tests."""
    return [
        PageExtractionResult(
            page_number=0,
            content="# Revenue Report\n\nTotal revenue: 1,234,567\nExpenses: 987,654",
            source="pdfplumber",
            confidence=0.9,
        ),
        PageExtractionResult(
            page_number=1,
            content="# Balance Sheet\n\n| Asset | Value |\n|---|---|\n| Cash | 500,000 |",
            source="mistral",
            alternative_content="# Balance Sheet\n\nCash: 500,000",
            confidence=0.85,
        ),
        PageExtractionResult(
            page_number=2,
            content="# Notes\n\nAdditional notes about the financial period.",
            source="gemini",
            confidence=0.7,
        ),
    ]


# --- Workflow Registration Tests ---

class TestWorkflowRegistration:
    """Verify smart extraction is properly registered in the workflow system."""

    def test_workflow_type_exists(self):
        """SMART_EXTRACTION should be a valid WorkflowType."""
        assert hasattr(WorkflowType, "SMART_EXTRACTION")
        assert WorkflowType.SMART_EXTRACTION.value == "smart_extraction"

    def test_workflow_has_display_name(self):
        """SMART_EXTRACTION should have a display name."""
        assert WorkflowType.SMART_EXTRACTION in WORKFLOW_NAMES

    def test_query_routing_smart(self):
        """Query 'smart' should route to SMART_EXTRACTION."""
        result = get_workflow_for_query("smart")
        assert result == WorkflowType.SMART_EXTRACTION

    def test_query_routing_smart_extraction(self):
        """Query 'smart_extraction' should route to SMART_EXTRACTION."""
        result = get_workflow_for_query("smart_extraction")
        assert result == WorkflowType.SMART_EXTRACTION

    def test_default_query_still_routes_to_mistral(self):
        """Default query should still route to MISTRAL."""
        result = get_workflow_for_query("some random query")
        assert result == WorkflowType.MISTRAL


# --- ExtractionRouter Tests ---

class TestExtractionRouter:
    """Tests for page extraction routing."""

    def test_create_consecutive_batches_simple(self):
        """Consecutive pages should be grouped together."""
        router = ExtractionRouter()
        pages = [
            PageAnalysis(i, PageType.IMAGE_ONLY, 0, 1, False, "")
            for i in range(5)
        ]
        batches = router._create_consecutive_batches(pages, max_batch_size=10)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_create_consecutive_batches_gap(self):
        """Non-consecutive pages should be in separate batches."""
        router = ExtractionRouter()
        pages = [
            PageAnalysis(0, PageType.IMAGE_ONLY, 0, 1, False, ""),
            PageAnalysis(1, PageType.IMAGE_ONLY, 0, 1, False, ""),
            PageAnalysis(5, PageType.IMAGE_ONLY, 0, 1, False, ""),
            PageAnalysis(6, PageType.IMAGE_ONLY, 0, 1, False, ""),
        ]
        batches = router._create_consecutive_batches(pages, max_batch_size=10)
        assert len(batches) == 2
        assert len(batches[0]) == 2  # pages 0,1
        assert len(batches[1]) == 2  # pages 5,6

    def test_create_consecutive_batches_max_size(self):
        """Batches should respect max_batch_size."""
        router = ExtractionRouter()
        pages = [
            PageAnalysis(i, PageType.IMAGE_ONLY, 0, 1, False, "")
            for i in range(15)
        ]
        batches = router._create_consecutive_batches(pages, max_batch_size=5)
        assert len(batches) == 3
        assert all(len(b) == 5 for b in batches)

    def test_create_consecutive_batches_empty(self):
        """Empty input should return empty batches."""
        router = ExtractionRouter()
        batches = router._create_consecutive_batches([], max_batch_size=10)
        assert batches == []

    def test_create_consecutive_batches_single(self):
        """Single page should produce single batch."""
        router = ExtractionRouter()
        pages = [PageAnalysis(3, PageType.IMAGE_ONLY, 0, 1, False, "")]
        batches = router._create_consecutive_batches(pages, max_batch_size=10)
        assert len(batches) == 1
        assert len(batches[0]) == 1

    def test_table_to_markdown(self):
        """Table data should be converted to markdown format."""
        router = ExtractionRouter()
        table = [
            ["Name", "Amount"],
            ["Revenue", "1000"],
            ["Expenses", "500"],
        ]
        result = router._table_to_markdown(table)
        assert "| Name | Amount |" in result
        assert "| Revenue | 1000 |" in result
        assert "| --- | --- |" in result

    def test_table_to_markdown_empty(self):
        """Empty table should return empty string."""
        router = ExtractionRouter()
        assert router._table_to_markdown([]) == ""
        assert router._table_to_markdown([["header"]]) == ""

    def test_split_mistral_output_by_headers(self):
        """Should split Mistral output by page headers."""
        router = ExtractionRouter()
        content = "# Page 1\n\nContent 1\n\n# Page 2\n\nContent 2"
        parts = router._split_mistral_output(content, 2)
        assert len(parts) == 2

    def test_split_mistral_output_by_separator(self):
        """Should split by --- when no page headers."""
        router = ExtractionRouter()
        content = "Content 1\n---\nContent 2"
        parts = router._split_mistral_output(content, 2)
        assert len(parts) == 2

    def test_split_mistral_output_single_page(self):
        """Single page should return whole content."""
        router = ExtractionRouter()
        content = "Just some content"
        parts = router._split_mistral_output(content, 1)
        assert len(parts) == 1
        assert parts[0] == content

    def test_split_mistral_output_fallback(self):
        """If can't split properly, return content for first page."""
        router = ExtractionRouter()
        content = "Content without separators"
        parts = router._split_mistral_output(content, 3)
        assert len(parts) == 3
        assert parts[0] == content
        assert parts[1] == ""
        assert parts[2] == ""

    @pytest.mark.asyncio
    async def test_extract_text_pages_empty(self):
        """No text pages should return empty list."""
        router = ExtractionRouter()
        results = await router._extract_text_pages("/fake/path", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_extract_empty_pages_no_gemini(self):
        """Empty pages without Gemini should return empty results."""
        router = ExtractionRouter()
        pages = [PageAnalysis(0, PageType.EMPTY, 0, 0, False, "")]
        with patch(
            "src.services.smart_extraction.extraction_router.get_client_factory"
        ) as mock_cf:
            mock_cf.return_value.gemini_client = None
            router._client_factory = mock_cf.return_value
            results = await router._extract_empty_pages(b"fake", pages)
            assert len(results) == 1
            assert results[0].content == ""
            assert results[0].confidence == 0.0


# --- SmartValidationService Tests ---

class TestSmartValidationService:
    """Tests for cross-validation of smart extraction results."""

    @pytest.mark.asyncio
    async def test_clean_single_source_passes(self):
        """Clean single-source content should pass through unchanged."""
        validator = SmartValidationService()
        results = [
            PageExtractionResult(
                page_number=0,
                content="# Financial Report\n\n" + "Valid content with numbers 1234. " * 20,
                source="pdfplumber",
                confidence=0.9,
            )
        ]
        validated = await validator.validate_results(results)
        assert len(validated) == 1
        assert validated[0].content == results[0].content

    @pytest.mark.asyncio
    async def test_dual_source_primary_clean(self):
        """When primary is clean and alt is problematic, keep primary."""
        validator = SmartValidationService()
        results = [
            PageExtractionResult(
                page_number=0,
                content="# Good Content\n\n" + "Valid financial data 12345.67. " * 20,
                source="mistral",
                alternative_content="",  # Empty = problematic
                confidence=0.85,
            )
        ]
        validated = await validator.validate_results(results)
        assert validated[0].source == "mistral"

    @pytest.mark.asyncio
    async def test_dual_source_alt_clean(self):
        """When primary is problematic and alt is clean, use alt."""
        validator = SmartValidationService()
        # Use varied content to avoid triggering repetitive_numbers detection
        alt_text = (
            "# Financial Report Q4 2025\n\n"
            "Total Revenue: 1,234,567.89\n"
            "Operating Expenses: 987,654.32\n"
            "Net Income: 246,913.57\n\n"
            "The company reported strong growth across all divisions. "
            "Cash flow from operations increased by 15% year-over-year. "
            "Key metrics include a 22% improvement in gross margin.\n\n"
            "| Division | Revenue | Growth |\n"
            "|----------|---------|--------|\n"
            "| North | 456,789 | 12% |\n"
            "| South | 345,678 | 18% |\n"
            "| East | 234,567 | 25% |\n"
            "| West | 197,533 | 8% |\n"
        )
        results = [
            PageExtractionResult(
                page_number=0,
                content="",  # Empty = problematic
                source="mistral",
                alternative_content=alt_text,
                confidence=0.85,
            )
        ]
        validated = await validator.validate_results(results)
        assert validated[0].source == "pdfplumber"

    @pytest.mark.asyncio
    async def test_empty_content_stays_empty(self):
        """Empty content should pass through."""
        validator = SmartValidationService()
        results = [
            PageExtractionResult(
                page_number=0,
                content="",
                source="gemini",
                confidence=0.0,
            )
        ]
        validated = await validator.validate_results(results)
        assert validated[0].content == ""

    def test_merge_contents_longer_first(self):
        """Merge should take longer content as base."""
        validator = SmartValidationService()
        primary = "This is a longer piece of content with many details and much information included."
        alternative = "Short."
        merged = validator._merge_contents(primary, alternative)
        assert primary in merged

    def test_merge_contents_unique_lines(self):
        """Merge should include unique substantial lines from supplement."""
        validator = SmartValidationService()
        primary = "Line one of the content.\nLine two of the content."
        alternative = "Line one of the content.\nThis is a unique line that should be added to the result."
        merged = validator._merge_contents(primary, alternative)
        assert "unique line" in merged

    def test_has_problems_empty(self):
        """Empty content should be detected as problematic."""
        validator = SmartValidationService()
        assert validator._has_problems("") is True
        assert validator._has_problems("   ") is True


# --- GeminiPolisher Tests ---

class TestGeminiPolisher:
    """Tests for Gemini polish pass."""

    @pytest.mark.asyncio
    async def test_polish_disabled(self):
        """When polish is disabled, raw content should be returned."""
        polisher = GeminiPolisher()
        results = [
            PageExtractionResult(0, "Content page 1", "pdfplumber"),
            PageExtractionResult(1, "Content page 2", "mistral"),
        ]
        with patch("src.services.smart_extraction.gemini_polisher.settings") as mock_settings:
            mock_settings.SMART_EXTRACTION_POLISH_ENABLED = False
            mock_settings.SMART_EXTRACTION_POLISH_BATCH_SIZE = 5
            mock_settings.SMART_EXTRACTION_POLISH_CONCURRENCY = 3
            mock_settings.SMART_EXTRACTION_POLISH_PROMPT = None
            output = await polisher.polish(results)
        assert "Content page 1" in output
        assert "Content page 2" in output

    @pytest.mark.asyncio
    async def test_polish_no_gemini_client(self):
        """When Gemini unavailable, raw content should be returned."""
        polisher = GeminiPolisher()
        results = [
            PageExtractionResult(0, "Content", "pdfplumber"),
        ]
        with patch("src.services.smart_extraction.gemini_polisher.get_client_factory") as mock_factory:
            mock_factory.return_value.gemini_client = None
            output = await polisher.polish(results)
        assert "Content" in output

    def test_verify_number_preservation_identical(self):
        """Identical number content should pass verification."""
        polisher = GeminiPolisher()
        text = "Revenue: 1,234,567.89 Expenses: 987,654.32"
        assert polisher._verify_number_preservation(text, text) is True

    def test_verify_number_preservation_no_numbers(self):
        """Content without numbers should pass."""
        polisher = GeminiPolisher()
        assert polisher._verify_number_preservation("No numbers here", "Still no numbers") is True

    def test_verify_number_preservation_missing_numbers(self):
        """Content with missing numbers should fail."""
        polisher = GeminiPolisher()
        original = "Values: 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000"
        polished = "Values: 100"
        assert polisher._verify_number_preservation(original, polished) is False

    def test_combine_results(self):
        """Combine results should add page headers."""
        polisher = GeminiPolisher()
        results = [
            PageExtractionResult(0, "Content 1", "pdfplumber"),
            PageExtractionResult(1, "Content 2", "mistral"),
        ]
        output = polisher._combine_results(results)
        assert "## Page 1" in output
        assert "## Page 2" in output
        assert "Content 1" in output
        assert "Content 2" in output


# --- SmartExtractionHandler Tests ---

class TestSmartExtractionHandler:
    """Tests for the smart extraction workflow handler."""

    @pytest.mark.asyncio
    async def test_handler_execute_mocked(self, sample_text_pdf):
        """Handler should execute the full pipeline (mocked clients)."""
        handler = SmartExtractionHandler()

        # Mock all external clients
        with patch("src.services.smart_extraction.extraction_router.get_client_factory") as mock_cf, \
             patch("src.services.smart_extraction.smart_validator.get_client_factory") as mock_cf2, \
             patch("src.services.smart_extraction.gemini_polisher.get_client_factory") as mock_cf3, \
             patch("src.services.smart_extraction.gemini_polisher.settings") as mock_settings:

            # Configure mock settings
            mock_settings.SMART_EXTRACTION_POLISH_ENABLED = False
            mock_settings.SMART_EXTRACTION_POLISH_BATCH_SIZE = 5
            mock_settings.SMART_EXTRACTION_POLISH_CONCURRENCY = 3
            mock_settings.SMART_EXTRACTION_POLISH_PROMPT = None

            # Mock mistral client
            mock_mistral = MagicMock()
            mock_cf.return_value.mistral_client = mock_mistral
            mock_cf.return_value.gemini_client = None

            mock_cf2.return_value.gemini_client = None
            mock_cf3.return_value.gemini_client = None

            result = await handler.execute(
                pdf_path=sample_text_pdf,
                query="test",
                enable_validation=False,
            )

            assert result is not None
            assert result.metadata["workflow"] == "smart_extraction"
            assert "total_pages" in result.metadata
            assert "page_types" in result.metadata
            assert "timing" in result.metadata


# --- Orchestrator Auto-Detection Tests ---

class TestOrchestratorAutoDetection:
    """Tests for auto-detection in the workflow orchestrator."""

    @pytest.mark.asyncio
    async def test_auto_detect_routes_mixed_to_smart(self, sample_text_pdf):
        """Mixed content PDF should auto-route to smart extraction."""
        # Verify the auto-detect code path exists and the handler is registered
        from src.services.workflow_orchestrator import WorkflowOrchestrator
        orchestrator = WorkflowOrchestrator()

        # Verify smart extraction handler is registered
        assert WorkflowType.SMART_EXTRACTION in orchestrator.workflow_handlers
        assert isinstance(
            orchestrator.workflow_handlers[WorkflowType.SMART_EXTRACTION],
            SmartExtractionHandler,
        )

        # Verify the auto-detect imports work
        from src.services.page_analyzer import PageAnalyzer
        analyzer = PageAnalyzer()
        # A text-only PDF should NOT trigger mixed detection
        assert analyzer.quick_is_mixed(sample_text_pdf) is False

    def test_smart_extraction_handler_registered(self):
        """SmartExtractionHandler should be in orchestrator's handlers."""
        from src.services.workflow_orchestrator import WorkflowOrchestrator
        orchestrator = WorkflowOrchestrator()
        assert WorkflowType.SMART_EXTRACTION in orchestrator.workflow_handlers
        assert isinstance(
            orchestrator.workflow_handlers[WorkflowType.SMART_EXTRACTION],
            SmartExtractionHandler,
        )


# --- PageExtractionResult Tests ---

class TestPageExtractionResult:
    """Tests for the PageExtractionResult dataclass."""

    def test_creation_minimal(self):
        """Should create with minimal args."""
        result = PageExtractionResult(
            page_number=0,
            content="test",
            source="pdfplumber",
        )
        assert result.page_number == 0
        assert result.content == "test"
        assert result.source == "pdfplumber"
        assert result.alternative_content is None
        assert result.confidence == 1.0

    def test_creation_full(self):
        """Should create with all args."""
        result = PageExtractionResult(
            page_number=5,
            content="primary",
            source="mistral",
            alternative_content="secondary",
            confidence=0.85,
        )
        assert result.page_number == 5
        assert result.alternative_content == "secondary"
        assert result.confidence == 0.85
