"""
Workflow handlers for PDF extraction.

This package provides specialized handlers for different PDF extraction workflows,
following the Strategy pattern for clean separation of concerns.
"""
from .base_handler import BaseWorkflowHandler
from .text_extraction_handler import TextExtractionHandler
from .azure_di_handler import AzureDIHandler
from .ocr_images_handler import OcrImagesHandler
from .gemini_handler import GeminiHandler
from .default_handler import DefaultHandler

__all__ = [
    "BaseWorkflowHandler",
    "TextExtractionHandler",
    "AzureDIHandler",
    "OcrImagesHandler",
    "GeminiHandler",
    "DefaultHandler",
]
