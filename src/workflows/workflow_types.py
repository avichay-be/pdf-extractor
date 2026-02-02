"""
Workflow type definitions for PDF extraction.

Replaces string-based workflow identification with type-safe enum.
"""
from enum import Enum


class WorkflowType(Enum):
    """Supported PDF extraction workflows."""
    MISTRAL = "mistral"
    TEXT_EXTRACTION = "text_extraction"
    AZURE_DOCUMENT_INTELLIGENCE = "azure_document_intelligence"
    OPENAI = "openai"
    GEMINI = "gemini"
    GEMINI_WF = "gemini-wf"
    OCR_WITH_IMAGES = "ocr_with_images"

    def __str__(self) -> str:
        """Return string value of the workflow type."""
        return self.value


# Workflow display names for logging
WORKFLOW_NAMES = {
    WorkflowType.MISTRAL: "Mistral Document AI",
    WorkflowType.TEXT_EXTRACTION: "Text Extraction",
    WorkflowType.AZURE_DOCUMENT_INTELLIGENCE: "Azure Document Intelligence",
    WorkflowType.OPENAI: "OpenAI Vision",
    WorkflowType.GEMINI: "Google Gemini",
    WorkflowType.GEMINI_WF: "Google Gemini Page-by-Page",
    WorkflowType.OCR_WITH_IMAGES: "OCR with Images (Mistral + OpenAI)"
}
