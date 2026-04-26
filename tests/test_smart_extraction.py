"""
Tests for the retained page-aware extraction pipeline.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

from src.core.config import settings
from src.services.page_aware_extraction_service import PageAwareExtractionService
from src.services.page_analyzer import PageAnalysis, PageType
from src.services.smart_extraction.extraction_router import (
    ExtractionRouter,
    PageExtractionResult,
)
from src.services.smart_extraction.gemini_polisher import GeminiPolisher
from src.services.smart_extraction.smart_validator import SmartValidationService


@pytest.fixture
def text_page():
    return PageAnalysis(0, PageType.TEXT_ONLY, 500, 0, True, "Revenue")


@pytest.fixture
def mixed_page():
    return PageAnalysis(1, PageType.IMAGE_WITH_TEXT, 200, 1, True, "Chart")


@pytest.fixture
def image_page():
    return PageAnalysis(2, PageType.IMAGE_ONLY, 0, 2, True, "")


class TestExtractionRouter:
    def test_create_consecutive_batches_with_gap(self):
        router = ExtractionRouter(mistral_client=MagicMock())
        pages = [
            PageAnalysis(0, PageType.IMAGE_ONLY, 0, 1, False, ""),
            PageAnalysis(1, PageType.IMAGE_ONLY, 0, 1, False, ""),
            PageAnalysis(4, PageType.IMAGE_ONLY, 0, 1, False, ""),
        ]
        batches = router._create_consecutive_batches(pages, max_batch_size=10)
        assert len(batches) == 2
        assert [page.page_number for page in batches[0]] == [0, 1]
        assert [page.page_number for page in batches[1]] == [4]

    @pytest.mark.asyncio
    async def test_text_only_pages_use_digital_extraction(self, text_page):
        router = ExtractionRouter(mistral_client=MagicMock())
        with patch.object(router, "_pdfplumber_extract") as mock_extract:
            mock_extract.return_value = [
                PageExtractionResult(
                    page_number=0,
                    content="Digital content",
                    source="pdfplumber",
                    strategy="text_extraction",
                )
            ]
            results = await router._extract_text_pages("/fake.pdf", [text_page])

        assert len(results) == 1
        assert results[0].strategy == "text_extraction"
        assert results[0].source == "pdfplumber"

    @pytest.mark.asyncio
    async def test_mixed_pages_use_ocr_primary_and_text_validation(self, mixed_page):
        router = ExtractionRouter(mistral_client=MagicMock())
        with patch.object(router, "_extract_ocr_pages", new_callable=AsyncMock) as mock_ocr, patch.object(
            router, "_extract_text_pages", new_callable=AsyncMock
        ) as mock_text:
            mock_ocr.return_value = [
                PageExtractionResult(
                    page_number=1,
                    content="OCR content",
                    source="mistral",
                    strategy="ocr",
                    ocr_mode="all_around",
                )
            ]
            mock_text.return_value = [
                PageExtractionResult(
                    page_number=1,
                    content="Digital text",
                    source="pdfplumber",
                    strategy="text_extraction",
                )
            ]

            results = await router._extract_mixed_pages("/fake.pdf", b"fake", [mixed_page])

        assert len(results) == 1
        assert results[0].content == "OCR content"
        assert results[0].alternative_content == "Digital text"
        assert results[0].validation_source == "text_extraction"

    @pytest.mark.asyncio
    async def test_ocr_pages_use_mistral_batches(self, image_page):
        mistral_client = MagicMock()
        router = ExtractionRouter(mistral_client=mistral_client)
        pages = [
            PageAnalysis(2, PageType.IMAGE_ONLY, 0, 2, True, ""),
            PageAnalysis(3, PageType.IMAGE_ONLY, 0, 2, True, ""),
        ]
        with patch.object(router, "_process_mistral_batch", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = [
                PageExtractionResult(
                    page_number=2,
                    page_range=(2, 3),
                    content="OCR batch",
                    source="mistral",
                    strategy="ocr",
                    ocr_mode="all_around",
                )
            ]
            results = await router._extract_ocr_pages(b"fake-pdf", pages)

        assert len(results) == 1
        assert results[0].page_range == (2, 3)
        assert results[0].ocr_mode == "all_around"


class TestSmartValidationService:
    @pytest.mark.asyncio
    async def test_dual_source_prefers_clean_alternative(self):
        validator = SmartValidationService(enable_single_source_fallback=False)
        result = PageExtractionResult(
            page_number=0,
            content="",
            source="mistral",
            strategy="ocr",
            ocr_mode="all_around",
            alternative_content="Valid financial text 12345 " * 10,
            validation_source="text_extraction",
        )

        validated = await validator.validate_results([result])
        assert validated[0].source == "pdfplumber"
        assert validated[0].strategy == "text_extraction"

    @pytest.mark.asyncio
    async def test_single_source_fallback_can_be_disabled(self):
        validator = SmartValidationService(enable_single_source_fallback=False)
        result = PageExtractionResult(
            page_number=0,
            content="| | |\n| | |\n| | |\n| | |\n| | |",
            source="mistral",
            strategy="ocr",
            ocr_mode="all_around",
        )

        validated = await validator.validate_results([result], pdf_bytes=b"fake")
        assert validated[0].content == result.content
        assert validated[0].source == "mistral"

    @pytest.mark.asyncio
    async def test_pdfplumber_single_source_does_not_call_gemini_by_default(self, monkeypatch):
        monkeypatch.setattr(
            settings,
            "SMART_EXTRACTION_GEMINI_FALLBACK_FOR_PDFPLUMBER",
            False,
        )
        gemini_client = MagicMock()
        validator = SmartValidationService(gemini_client=gemini_client)
        result = PageExtractionResult(
            page_number=0,
            content="| | |\n| | |\n| | |\n| | |\n| | |",
            source="pdfplumber",
            strategy="text_extraction",
        )

        validated = await validator.validate_results([result], pdf_bytes=b"fake")

        assert validated[0].source == "pdfplumber"
        gemini_client.extract_page_content.assert_not_called()


class TestGeminiPolisher:
    def test_combine_results_uses_page_ranges(self):
        polisher = GeminiPolisher()
        output = polisher._combine_results(
            [
                PageExtractionResult(
                    page_number=0,
                    page_range=(0, 1),
                    content="Merged content",
                    source="mistral",
                    strategy="ocr",
                    ocr_mode="all_around",
                )
            ]
        )
        assert "## Pages 1-2" in output
        assert "Merged content" in output

    @pytest.mark.asyncio
    async def test_polish_skips_digital_only_content(self):
        polisher = GeminiPolisher()
        output = await polisher.polish(
            [
                PageExtractionResult(
                    page_number=0,
                    content="Digital text only",
                    source="pdfplumber",
                    strategy="text_extraction",
                )
            ]
        )
        assert "Digital text only" in output

    @pytest.mark.asyncio
    async def test_polish_can_be_disabled_per_request(self):
        polisher = GeminiPolisher()
        output = await polisher.polish(
            [
                PageExtractionResult(
                    page_number=0,
                    content="OCR content",
                    source="mistral",
                    strategy="ocr",
                    ocr_mode="all_around",
                )
            ],
            enabled=False,
        )
        assert "OCR content" in output

    @pytest.mark.asyncio
    async def test_polish_only_sends_ocr_results(self):
        mock_gemini_client = MagicMock()
        polisher = GeminiPolisher(gemini_client=mock_gemini_client)

        with patch.object(polisher, "_polish_batch", new_callable=AsyncMock) as mock_polish_batch:
            mock_polish_batch.return_value = "## Page 2\n\nPolished OCR"

            output = await polisher.polish(
                [
                    PageExtractionResult(
                        page_number=0,
                        content="Digital text",
                        source="pdfplumber",
                        strategy="text_extraction",
                    ),
                    PageExtractionResult(
                        page_number=1,
                        content="OCR text",
                        source="mistral",
                        strategy="ocr",
                        ocr_mode="all_around",
                    ),
                    PageExtractionResult(
                        page_number=2,
                        content="More digital text",
                        source="pdfplumber",
                        strategy="text_extraction",
                    ),
                ]
            )

        assert "## Page 1\n\nDigital text" in output
        assert "## Page 2\n\nPolished OCR" in output
        assert "## Page 3\n\nMore digital text" in output
        assert mock_polish_batch.await_count == 1
        sent_batch = mock_polish_batch.await_args.args[1]
        assert sent_batch == ["## Page 2\n\nOCR text"]

    def test_polish_uses_configured_model(self):
        original_model = settings.SMART_EXTRACTION_POLISH_MODEL
        try:
            settings.SMART_EXTRACTION_POLISH_MODEL = "gemini-custom-polish"
            mock_models = MagicMock()
            mock_models.generate_content.return_value = MagicMock(text="Polished")
            mock_gemini_client = MagicMock()
            mock_gemini_client.client.models = mock_models
            mock_gemini_client.model_name = "gemini-2.5-flash"

            polisher = GeminiPolisher(gemini_client=mock_gemini_client)
            result = polisher._call_gemini_text(
                mock_gemini_client,
                polisher._model_name,
                "Prompt",
                "Body",
            )

            assert result == "Polished"
            mock_models.generate_content.assert_called_once()
            call_kwargs = mock_models.generate_content.call_args.kwargs
            assert call_kwargs["model"] == "gemini-custom-polish"
            assert call_kwargs["config"].automatic_function_calling.disable is True
        finally:
            settings.SMART_EXTRACTION_POLISH_MODEL = original_model

    def test_polish_deprecated_model_falls_back_to_gemini_model(self):
        original_polish_model = settings.SMART_EXTRACTION_POLISH_MODEL
        original_gemini_model = settings.GEMINI_MODEL
        try:
            settings.SMART_EXTRACTION_POLISH_MODEL = "gemini-3-1-flash-lite-preview"
            settings.GEMINI_MODEL = "gemini-2.5-flash"

            polisher = GeminiPolisher(gemini_client=MagicMock())

            assert polisher._model_name == "gemini-2.5-flash"
        finally:
            settings.SMART_EXTRACTION_POLISH_MODEL = original_polish_model
            settings.GEMINI_MODEL = original_gemini_model


class TestPageAwareExtractionService:
    @pytest.mark.asyncio
    async def test_service_returns_retained_metadata(self):
        analyses = [
            PageAnalysis(0, PageType.TEXT_ONLY, 100, 0, True, "Text"),
            PageAnalysis(1, PageType.IMAGE_ONLY, 0, 1, True, ""),
        ]
        routed_results = [
            PageExtractionResult(
                page_number=0,
                content="Digital text",
                source="pdfplumber",
                strategy="text_extraction",
            ),
            PageExtractionResult(
                page_number=1,
                content="OCR page",
                source="mistral",
                strategy="ocr",
                ocr_mode="all_around",
            ),
        ]

        mistral_client = MagicMock()
        gemini_client = MagicMock()
        analyzer = MagicMock()
        analyzer.analyze_pdf.return_value = analyses
        service = PageAwareExtractionService(
            mistral_client=mistral_client,
            gemini_client=gemini_client,
            page_analyzer=analyzer,
        )

        with patch("builtins.open", mock_open(read_data=b"fake-pdf")), patch(
            "src.services.page_aware_extraction_service.ExtractionRouter"
        ) as mock_router_class, patch(
            "src.services.page_aware_extraction_service.GeminiPolisher"
        ) as mock_polisher_class:
            mock_router = MagicMock()
            mock_router.extract_pages = AsyncMock(return_value=routed_results)
            mock_router_class.return_value = mock_router

            mock_polisher = MagicMock()
            mock_polisher.polish = AsyncMock(return_value="Combined content")
            mock_polisher_class.return_value = mock_polisher

            result = await service.extract_document(
                "/fake.pdf",
                enable_validation=False,
                enable_polishing=False,
            )

        assert result.metadata["workflow"] == "ocr"
        assert result.metadata["ocr_mode"] == "all_around"
        assert result.metadata["routing_strategy"] == "page_aware"
        assert len(result.metadata["page_routes"]) == 2
        assert result.metadata["processing_options"]["cross_validation"] is False
        assert result.metadata["processing_options"]["polishing"] is False
        mock_polisher.polish.assert_awaited_once_with(routed_results, enabled=False)
