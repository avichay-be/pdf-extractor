"""
Workflow routing logic for PDF extraction.

Determines which workflow to use based on query patterns.
"""
import logging
from typing import Optional

from src.core.config import settings
from .workflow_types import WorkflowType

logger = logging.getLogger(__name__)


def get_workflow_for_query(query: Optional[str]) -> WorkflowType:
    """
    Determine which workflow to use based on query pattern.

    Uses the QUERY_WORKFLOW_MAPPING from config to map query patterns to workflows.

    Args:
        query: Search query string (can be None or empty)

    Returns:
        WorkflowType enum value for the appropriate workflow
    """
    if not query:
        default_workflow_str = settings.QUERY_WORKFLOW_MAPPING.get("default", "mistral")
        return _string_to_workflow_type(default_workflow_str)

    query_lower = query.lower().strip()

    # Check for exact or partial matches in the mapping
    for pattern, workflow_str in settings.QUERY_WORKFLOW_MAPPING.items():
        if pattern == "default":
            continue
        if pattern.lower() in query_lower:
            workflow = _string_to_workflow_type(workflow_str)
            logger.info(f"Query '{query}' matched pattern '{pattern}' -> workflow: {workflow}")
            return workflow

    # Default fallback
    default_workflow_str = settings.QUERY_WORKFLOW_MAPPING.get("default", "mistral")
    workflow = _string_to_workflow_type(default_workflow_str)
    logger.info(f"Query '{query}' using default workflow: {workflow}")
    return workflow


def is_text_extraction_query(query: str) -> bool:
    """
    Check if query should use text extraction workflow (pdfplumber).

    Args:
        query: Search query string

    Returns:
        True if query maps to text_extraction workflow
    """
    return get_workflow_for_query(query) == WorkflowType.TEXT_EXTRACTION


def is_azure_document_intelligence_query(query: str) -> bool:
    """
    Check if query should use Azure Document Intelligence workflow.

    Args:
        query: Search query string

    Returns:
        True if query maps to azure_document_intelligence workflow
    """
    return get_workflow_for_query(query) == WorkflowType.AZURE_DOCUMENT_INTELLIGENCE


def is_ocr_with_images_query(query: str) -> bool:
    """
    Check if query should use OCR with Images workflow.

    Args:
        query: Search query string

    Returns:
        True if query maps to ocr_with_images workflow
    """
    return get_workflow_for_query(query) == WorkflowType.OCR_WITH_IMAGES


def is_gemini_wf_query(query: str) -> bool:
    """
    Check if query should use Gemini page-by-page workflow.

    Args:
        query: Search query string

    Returns:
        True if query maps to gemini-wf workflow
    """
    return get_workflow_for_query(query) == WorkflowType.GEMINI_WF


def _string_to_workflow_type(workflow_str: str) -> WorkflowType:
    """
    Convert string workflow identifier to WorkflowType enum.

    Args:
        workflow_str: String workflow identifier (e.g., "mistral", "openai")

    Returns:
        Corresponding WorkflowType enum value

    Raises:
        ValueError: If workflow string is not recognized
    """
    workflow_map = {
        "mistral": WorkflowType.MISTRAL,
        "text_extraction": WorkflowType.TEXT_EXTRACTION,
        "azure_document_intelligence": WorkflowType.AZURE_DOCUMENT_INTELLIGENCE,
        "openai": WorkflowType.OPENAI,
        "gemini": WorkflowType.GEMINI,
        "gemini-wf": WorkflowType.GEMINI_WF,
        "ocr_with_images": WorkflowType.OCR_WITH_IMAGES,
    }

    workflow_type = workflow_map.get(workflow_str.lower())
    if workflow_type is None:
        logger.warning(f"Unknown workflow string '{workflow_str}', defaulting to MISTRAL")
        return WorkflowType.MISTRAL

    return workflow_type
