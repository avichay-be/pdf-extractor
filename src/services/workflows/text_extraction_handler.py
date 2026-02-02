"""
Text extraction workflow handler.

Uses pdfplumber to extract tables from digitally-generated PDFs without OCR/AI.
Ideal for documents where tables are already well-structured.
"""
import logging
import time
from typing import Optional

from .base_handler import BaseWorkflowHandler
from src.models.workflow_models import WorkflowResult
from src.services.extraction_service import process_text_extraction
from src.core.error_handling import WorkflowExecutionError

logger = logging.getLogger(__name__)


class TextExtractionHandler(BaseWorkflowHandler):
    """Handler for text extraction workflow using pdfplumber."""

    async def execute(
        self,
        pdf_path: str,
        query: str,
        enable_validation: Optional[bool] = None
    ) -> WorkflowResult:
        """Execute text extraction workflow.

        Args:
            pdf_path: Path to the PDF file
            query: Query string (for logging context)
            enable_validation: Ignored for this workflow (no validation needed)

        Returns:
            WorkflowResult with extracted table content

        Raises:
            WorkflowExecutionError: If extraction fails
        """
        start_time = time.time()
        self._log_execution_start("TextExtraction", pdf_path, query)

        try:
            # Extract tables using pdfplumber
            combined_markdown, metadata = await process_text_extraction(
                pdf_path=pdf_path,
                query=query
            )

            execution_time = time.time() - start_time

            # Build result
            result = WorkflowResult(
                content=combined_markdown,
                metadata={
                    **metadata,
                    "execution_time": execution_time,
                    "workflow": "text_extraction"
                },
                sections=None,  # Text extraction doesn't split by outlines
                validation_report=None  # No validation for this workflow
            )

            self._log_execution_complete("TextExtraction", result, execution_time)
            return result

        except Exception as e:
            logger.error(f"Text extraction workflow failed: {e}")
            raise WorkflowExecutionError(f"Text extraction failed: {str(e)}")
