"""
Single page-aware extraction service for the retained PDF processing flow.
"""
import asyncio
import logging
import time
from typing import Optional

from src.core.config import settings
from src.core.error_handling import WorkflowExecutionError
from src.core.logging_utils import build_log_extra, log_event
from src.models.workflow_models import WorkflowResult
from src.services.gemini_client import GeminiDocumentClient
from src.services.mistral_client import MistralDocumentClient
from src.services.page_analyzer import PageAnalyzer
from src.services.smart_extraction.extraction_router import ExtractionRouter
from src.services.smart_extraction.gemini_polisher import GeminiPolisher
from src.services.smart_extraction.smart_validator import SmartValidationService

logger = logging.getLogger(__name__)


class PageAwareExtractionService:
    """Runs the retained page-aware extraction pipeline end to end."""

    def __init__(
        self,
        mistral_client: MistralDocumentClient,
        gemini_client: Optional[GeminiDocumentClient] = None,
        page_analyzer: Optional[PageAnalyzer] = None,
    ):
        self._mistral_client = mistral_client
        self._gemini_client = gemini_client
        self._page_analyzer = page_analyzer or PageAnalyzer()

    async def extract_document(
        self,
        pdf_path: str,
        enable_validation: Optional[bool] = None,
        enable_polishing: Optional[bool] = None,
    ) -> WorkflowResult:
        """Extract a PDF using the single retained runtime path."""
        start_time = time.time()

        try:
            analyses = await asyncio.to_thread(self._page_analyzer.analyze_pdf, pdf_path)
            analysis_time = time.time() - start_time
            log_event(
                logger,
                logging.INFO,
                "page_analysis_completed",
                total_pages=len(analyses),
                analysis_time_s=round(analysis_time, 2),
            )

            with open(pdf_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()

            router = ExtractionRouter(mistral_client=self._mistral_client)
            page_results = await router.extract_pages(pdf_path, analyses, pdf_bytes=pdf_bytes)
            extraction_time = time.time() - start_time - analysis_time

            should_validate_single_source = (
                enable_validation
                if enable_validation is not None
                else settings.ENABLE_CROSS_VALIDATION
            )

            validation_report = None
            dual_source_pages = sum(
                1 for result in page_results if result.alternative_content is not None
            )
            should_run_validation = should_validate_single_source or dual_source_pages > 0

            if should_run_validation:
                validator = SmartValidationService(
                    gemini_client=self._gemini_client,
                    enable_single_source_fallback=should_validate_single_source,
                )
                page_results = await validator.validate_results(page_results, pdf_bytes)

                source_distribution: dict[str, int] = {}
                for result in page_results:
                    source_distribution[result.source] = (
                        source_distribution.get(result.source, 0) + 1
                    )

                validation_report = {
                    "enabled": True,
                    "mode": (
                        "full"
                        if should_validate_single_source
                        else "mixed_page_cross_validation"
                    ),
                    "dual_source_pages": dual_source_pages,
                    "source_distribution": source_distribution,
                    "total_results": len(page_results),
                }

            validation_time = (
                time.time() - start_time - analysis_time - extraction_time
                if should_run_validation
                else 0.0
            )

            polisher = GeminiPolisher(gemini_client=self._gemini_client)
            combined_markdown = await polisher.polish(
                page_results,
                enabled=enable_polishing,
            )
            total_time = time.time() - start_time
            polish_time = total_time - analysis_time - extraction_time - validation_time

            page_types: dict[str, int] = {}
            for analysis in analyses:
                page_types[analysis.page_type.value] = (
                    page_types.get(analysis.page_type.value, 0) + 1
                )

            return WorkflowResult(
                content=combined_markdown,
                metadata={
                    "workflow": "ocr",
                    "ocr_mode": "all_around",
                    "routing_strategy": "page_aware",
                    "total_pages": len(analyses),
                    "page_types": page_types,
                    "page_routes": [result.to_metadata() for result in page_results],
                    "processing_options": {
                        "cross_validation": (
                            enable_validation
                            if enable_validation is not None
                            else settings.ENABLE_CROSS_VALIDATION
                        ),
                        "polishing": (
                            enable_polishing
                            if enable_polishing is not None
                            else settings.SMART_EXTRACTION_POLISH_ENABLED
                        ),
                    },
                    "timing": {
                        "analysis_s": round(analysis_time, 2),
                        "extraction_s": round(extraction_time, 2),
                        "validation_s": round(validation_time, 2),
                        "polish_s": round(polish_time, 2),
                        "total_s": round(total_time, 2),
                    },
                },
                sections=None,
                validation_report=validation_report,
            )

        except Exception as exc:
            logger.exception(
                "page_aware_extraction_failed",
                extra=build_log_extra(
                    error_type=type(exc).__name__,
                    reason=str(exc),
                ),
            )
            raise WorkflowExecutionError(
                f"Page-aware extraction failed: {str(exc)}"
            ) from exc


def build_page_aware_extraction_service() -> PageAwareExtractionService:
    """Construct the retained runtime service with explicit dependencies."""
    mistral_client = MistralDocumentClient(api_key=settings.AZURE_API_KEY)

    gemini_client = None
    if settings.GEMINI_API_KEY:
        try:
            gemini_client = GeminiDocumentClient()
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "gemini_client_unavailable",
                error_type=type(exc).__name__,
                reason=str(exc),
            )

    return PageAwareExtractionService(
        mistral_client=mistral_client,
        gemini_client=gemini_client,
    )
