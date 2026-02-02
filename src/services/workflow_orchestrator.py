"""
Workflow orchestrator for PDF extraction.

Routes extraction requests to appropriate workflow handlers based on query patterns.
Provides a single entry point for all extraction workflows.
"""
import logging
from typing import Optional

from src.workflows import get_workflow_for_query
from src.workflows.workflow_types import WorkflowType
from src.services.workflows.text_extraction_handler import TextExtractionHandler
from src.services.workflows.azure_di_handler import AzureDIHandler
from src.services.workflows.ocr_images_handler import OcrImagesHandler
from src.services.workflows.gemini_handler import GeminiHandler
from src.services.workflows.default_handler import DefaultHandler
from src.models.workflow_models import WorkflowResult
from src.core.error_handling import WorkflowExecutionError

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Orchestrates workflow execution based on query patterns.

    Uses the Strategy pattern to delegate extraction to specialized handlers:
    - TextExtractionHandler: pdfplumber table extraction
    - AzureDIHandler: Azure Document Intelligence
    - OcrImagesHandler: Mistral + OpenAI for images
    - GeminiHandler: Gemini page-by-page
    - DefaultHandler: Mistral with validation
    """

    def __init__(self):
        """Initialize orchestrator with all workflow handlers."""
        self.workflow_handlers = {
            WorkflowType.TEXT_EXTRACTION: TextExtractionHandler(),
            WorkflowType.AZURE_DOCUMENT_INTELLIGENCE: AzureDIHandler(),
            WorkflowType.OCR_WITH_IMAGES: OcrImagesHandler(),
            WorkflowType.GEMINI_WF: GeminiHandler(),
            WorkflowType.MISTRAL: DefaultHandler(),
            WorkflowType.OPENAI: DefaultHandler(),  # Uses same handler as Mistral
            WorkflowType.GEMINI: GeminiHandler(),  # Maps to same as GEMINI_WF
        }

        logger.info("Workflow orchestrator initialized with all handlers")

    async def execute_workflow(
        self,
        pdf_path: str,
        query: str,
        enable_validation: Optional[bool] = None
    ) -> WorkflowResult:
        """Execute appropriate workflow based on query.

        Determines the workflow type from the query pattern and delegates
        to the corresponding handler.

        Args:
            pdf_path: Path to the PDF file
            query: Query string for filtering and workflow selection
            enable_validation: Whether to enable cross-validation

        Returns:
            WorkflowResult from the executed workflow

        Raises:
            WorkflowExecutionError: If workflow execution fails
            ValueError: If workflow type is not supported
        """
        # Determine workflow type from query
        workflow_type = get_workflow_for_query(query)

        logger.info(
            f"Orchestrating workflow: type={workflow_type}, "
            f"query='{query}', validation={enable_validation}"
        )

        # Get handler for workflow type
        handler = self.workflow_handlers.get(workflow_type)

        if handler is None:
            raise ValueError(
                f"Unsupported workflow type: {workflow_type}. "
                f"Available workflows: {list(self.workflow_handlers.keys())}"
            )

        # Execute workflow
        try:
            result = await handler.execute(
                pdf_path=pdf_path,
                query=query,
                enable_validation=enable_validation
            )

            logger.info(
                f"Workflow {workflow_type} completed successfully: "
                f"sections={result.section_count}, "
                f"validated={result.was_validated}"
            )

            return result

        except Exception as e:
            logger.error(f"Workflow {workflow_type} execution failed: {e}")
            raise WorkflowExecutionError(
                f"Failed to execute {workflow_type} workflow: {str(e)}"
            )


# Singleton orchestrator instance
_orchestrator: Optional[WorkflowOrchestrator] = None


def get_workflow_orchestrator() -> WorkflowOrchestrator:
    """Get singleton workflow orchestrator instance.

    Returns:
        WorkflowOrchestrator singleton instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = WorkflowOrchestrator()
    return _orchestrator
