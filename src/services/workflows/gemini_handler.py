"""
Gemini page-by-page workflow handler.

Uses Google Gemini to process each PDF page individually with async processing.
Ideal for documents requiring Gemini's multimodal capabilities.
"""
import logging
import time
from typing import Optional

from .base_handler import BaseWorkflowHandler
from src.models.workflow_models import WorkflowResult
from src.services.extraction_service import process_gemini_wf
from src.core.error_handling import WorkflowExecutionError

logger = logging.getLogger(__name__)


class GeminiHandler(BaseWorkflowHandler):
    """Handler for Gemini page-by-page workflow."""

    async def execute(
        self,
        pdf_path: str,
        query: str,
        enable_validation: Optional[bool] = None
    ) -> WorkflowResult:
        """Execute Gemini page-by-page workflow.

        Args:
            pdf_path: Path to the PDF file
            query: Query string (for logging context)
            enable_validation: Ignored for this workflow (Gemini doesn't use validation)

        Returns:
            WorkflowResult with extracted content

        Raises:
            WorkflowExecutionError: If extraction fails
        """
        start_time = time.time()
        self._log_execution_start("Gemini", pdf_path, query)

        try:
            # Process with Gemini page-by-page
            combined_markdown, metadata = await process_gemini_wf(
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
                    "workflow": "gemini-wf"
                },
                sections=None,  # Gemini doesn't split by outlines
                validation_report=None  # No validation for this workflow
            )

            self._log_execution_complete("Gemini", result, execution_time)
            return result

        except Exception as e:
            logger.error(f"Gemini workflow failed: {e}")
            raise WorkflowExecutionError(f"Gemini extraction failed: {str(e)}")
