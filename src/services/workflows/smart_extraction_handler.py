"""
Smart extraction workflow handler.

Orchestrates the full smart extraction pipeline:
1. Analyze pages (PageAnalyzer)
2. Route and extract (ExtractionRouter)
3. Cross-validate (SmartValidationService)
4. Gemini polish (GeminiPolisher)
"""
import asyncio
import logging
import time
from typing import Optional

from .base_handler import BaseWorkflowHandler
from src.models.workflow_models import WorkflowResult
from src.services.page_analyzer import PageAnalyzer, PageType
from src.services.smart_extraction.extraction_router import ExtractionRouter
from src.services.smart_extraction.smart_validator import SmartValidationService
from src.services.smart_extraction.gemini_polisher import GeminiPolisher
from src.core.error_handling import WorkflowExecutionError

logger = logging.getLogger(__name__)


class SmartExtractionHandler(BaseWorkflowHandler):
    """Handler for intelligent per-page extraction with auto-routing."""

    async def execute(
        self,
        pdf_path: str,
        query: str,
        enable_validation: Optional[bool] = None
    ) -> WorkflowResult:
        """Execute smart extraction workflow.

        Pipeline:
        1. Analyze all pages for content type
        2. Route each page to optimal extraction method
        3. Cross-validate dual-source pages
        4. Polish output through Gemini
        5. Return structured WorkflowResult

        Args:
            pdf_path: Path to the PDF file
            query: Query string (for metadata)
            enable_validation: Whether to enable cross-validation

        Returns:
            WorkflowResult with extracted content and rich metadata

        Raises:
            WorkflowExecutionError: If extraction fails
        """
        start_time = time.time()
        self._log_execution_start("Smart Extraction", pdf_path, query)

        try:
            # 1. Analyze pages
            analyzer = PageAnalyzer()
            analyses = await asyncio.to_thread(analyzer.analyze_pdf, pdf_path)

            analysis_time = time.time() - start_time
            logger.info(f"Page analysis completed in {analysis_time:.2f}s")

            # Build page type summary for metadata
            type_counts = {}
            for a in analyses:
                type_counts[a.page_type.value] = type_counts.get(a.page_type.value, 0) + 1

            # 2. Route and extract
            router = ExtractionRouter()
            page_results = await router.extract_pages(pdf_path, analyses)

            extraction_time = time.time() - start_time - analysis_time
            logger.info(f"Extraction completed in {extraction_time:.2f}s")

            # 3. Cross-validate
            should_validate = enable_validation if enable_validation is not None else True
            validation_report = None

            if should_validate:
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()

                validator = SmartValidationService()
                page_results = await validator.validate_results(page_results, pdf_bytes)

                # Build validation report
                sources = {}
                for r in page_results:
                    sources[r.source] = sources.get(r.source, 0) + 1

                validation_report = {
                    "enabled": True,
                    "status": "completed",
                    "source_distribution": sources,
                    "total_pages": len(page_results),
                }

            validation_time = time.time() - start_time - analysis_time - extraction_time

            # 4. Gemini polish
            polisher = GeminiPolisher()
            combined_markdown = await polisher.polish(page_results)

            total_time = time.time() - start_time
            polish_time = total_time - analysis_time - extraction_time - validation_time

            # 5. Build result
            result = WorkflowResult(
                content=combined_markdown,
                metadata={
                    "workflow": "smart_extraction",
                    "extraction_method": "smart_per_page",
                    "total_pages": len(analyses),
                    "page_types": type_counts,
                    "timing": {
                        "analysis_s": round(analysis_time, 2),
                        "extraction_s": round(extraction_time, 2),
                        "validation_s": round(validation_time, 2),
                        "polish_s": round(polish_time, 2),
                        "total_s": round(total_time, 2),
                    },
                    "query": query,
                },
                sections=None,
                validation_report=validation_report,
            )

            self._log_execution_complete("Smart Extraction", result, total_time)
            return result

        except Exception as e:
            logger.error(f"Smart extraction workflow failed: {e}")
            raise WorkflowExecutionError(
                f"Smart extraction failed: {str(e)}"
            )
