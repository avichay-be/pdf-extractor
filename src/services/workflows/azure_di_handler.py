"""
Azure Document Intelligence workflow handler.

Uses Azure's Document Intelligence API for intelligent table extraction and merging.
Ideal for complex financial documents with tables spanning multiple pages.
"""
import logging
import time
from typing import Optional

from .base_handler import BaseWorkflowHandler
from src.models.workflow_models import WorkflowResult
from src.services.extraction_service import process_azure_document_intelligence
from src.core.error_handling import WorkflowExecutionError

logger = logging.getLogger(__name__)


class AzureDIHandler(BaseWorkflowHandler):
    """Handler for Azure Document Intelligence workflow."""

    async def execute(
        self,
        pdf_path: str,
        query: str,
        enable_validation: Optional[bool] = None
    ) -> WorkflowResult:
        """Execute Azure Document Intelligence workflow.

        Args:
            pdf_path: Path to the PDF file
            query: Query string (for logging context)
            enable_validation: Ignored for this workflow (Azure DI doesn't use validation)

        Returns:
            WorkflowResult with extracted table content

        Raises:
            WorkflowExecutionError: If extraction fails
        """
        start_time = time.time()
        self._log_execution_start("AzureDocumentIntelligence", pdf_path, query)

        try:
            # Extract tables using Azure DI with intelligent merging
            combined_markdown, metadata = await process_azure_document_intelligence(
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
                    "workflow": "azure_document_intelligence"
                },
                sections=None,  # Azure DI doesn't split by outlines
                validation_report=None  # No validation for this workflow
            )

            self._log_execution_complete("AzureDocumentIntelligence", result, execution_time)
            return result

        except Exception as e:
            logger.error(f"Azure Document Intelligence workflow failed: {e}")
            raise WorkflowExecutionError(f"Azure DI extraction failed: {str(e)}")
