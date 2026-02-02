"""
Base workflow handler.

This module provides the abstract base class for all workflow handlers,
establishing a consistent interface for workflow execution.
"""
from abc import ABC, abstractmethod
from typing import Optional
import logging

from src.models.workflow_models import WorkflowResult

logger = logging.getLogger(__name__)


class BaseWorkflowHandler(ABC):
    """Abstract base class for all workflow handlers.

    Each workflow handler implements a specific PDF extraction strategy:
    - Text extraction (pdfplumber)
    - Azure Document Intelligence (table extraction)
    - OCR with images (Mistral + OpenAI)
    - Gemini page-by-page
    - Default Mistral with validation

    All handlers follow the same interface for consistent orchestration.
    """

    @abstractmethod
    async def execute(
        self,
        pdf_path: str,
        query: str,
        enable_validation: Optional[bool] = None
    ) -> WorkflowResult:
        """Execute the workflow.

        Args:
            pdf_path: Path to the PDF file to process
            query: Query string for filtering sections (e.g., "Financial Reports")
            enable_validation: Whether to enable cross-validation (None = use default)

        Returns:
            WorkflowResult containing extracted content, metadata, and optional validation

        Raises:
            WorkflowExecutionError: If workflow execution fails
            PDFValidationError: If PDF validation fails
            FileNotFoundError: If PDF file doesn't exist
        """
        pass

    def _log_execution_start(self, workflow_name: str, pdf_path: str, query: str):
        """Log workflow execution start."""
        logger.info(
            f"Starting {workflow_name} workflow: "
            f"pdf={pdf_path}, query={query}"
        )

    def _log_execution_complete(
        self,
        workflow_name: str,
        result: WorkflowResult,
        execution_time: float
    ):
        """Log workflow execution completion."""
        logger.info(
            f"Completed {workflow_name} workflow: "
            f"sections={result.section_count}, "
            f"validated={result.was_validated}, "
            f"time={execution_time:.2f}s"
        )
