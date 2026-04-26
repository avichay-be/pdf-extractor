"""
Smart extraction pipeline for mixed-content PDFs.

Routes each page to the optimal extraction method based on content analysis.
"""
from src.services.smart_extraction.extraction_router import ExtractionRouter, PageExtractionResult
from src.services.smart_extraction.smart_validator import SmartValidationService
from src.services.smart_extraction.gemini_polisher import GeminiPolisher

__all__ = [
    "ExtractionRouter",
    "PageExtractionResult",
    "SmartValidationService",
    "GeminiPolisher",
]
