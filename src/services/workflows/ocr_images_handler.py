"""
OCR with Images workflow handler.

Uses Mistral for OCR and OpenAI for extracting data from detected images.
Ideal for documents with embedded charts, diagrams, or complex visual data.
"""
import logging
import time
from typing import Optional

from .base_handler import BaseWorkflowHandler
from src.models.workflow_models import WorkflowResult
from src.services.extraction_service import process_ocr_with_images
from src.core.error_handling import WorkflowExecutionError

logger = logging.getLogger(__name__)


class OcrImagesHandler(BaseWorkflowHandler):
    """Handler for OCR with Images workflow (Mistral + OpenAI)."""

    async def execute(
        self,
        pdf_path: str,
        query: str,
        enable_validation: Optional[bool] = None
    ) -> WorkflowResult:
        """Execute OCR with Images workflow.

        Args:
            pdf_path: Path to the PDF file
            query: Query string (used as prompt for image extraction)
            enable_validation: Ignored for this workflow (has its own image processing)

        Returns:
            WorkflowResult with OCR content and extracted image data

        Raises:
            WorkflowExecutionError: If extraction fails
        """
        start_time = time.time()
        self._log_execution_start("OcrWithImages", pdf_path, query)

        try:
            # Process with Mistral OCR + OpenAI image extraction
            combined_markdown, metadata = await process_ocr_with_images(
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
                    "workflow": "ocr_with_images"
                },
                sections=None,  # OCR with images doesn't split by outlines
                validation_report=None  # No validation for this workflow
            )

            self._log_execution_complete("OcrWithImages", result, execution_time)
            return result

        except Exception as e:
            logger.error(f"OCR with Images workflow failed: {e}")
            raise WorkflowExecutionError(f"OCR with images extraction failed: {str(e)}")
