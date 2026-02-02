"""
Workflow management module for PDF extraction.

This module provides type-safe workflow definitions and routing logic.
"""
from .workflow_types import WorkflowType, WORKFLOW_NAMES
from .workflow_router import (
    get_workflow_for_query,
    is_text_extraction_query,
    is_azure_document_intelligence_query,
    is_ocr_with_images_query,
    is_gemini_wf_query
)

__all__ = [
    "WorkflowType",
    "WORKFLOW_NAMES",
    "get_workflow_for_query",
    "is_text_extraction_query",
    "is_azure_document_intelligence_query",
    "is_ocr_with_images_query",
    "is_gemini_wf_query",
]
