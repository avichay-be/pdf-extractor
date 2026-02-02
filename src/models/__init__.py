"""Pydantic models for API validation."""

from .api_models import (
    Base64FileRequest,
    ExtractionResponse,
    ExtractedContent,
    OutlineExtractionResponse
)
from .mistral_models import (
    DocumentInput,
    MistralOCRRequest,
    MistralOCRResponse,
    MistralErrorResponse,
    Page,
    Dimensions,
    UsageInfo
)
from .azure_document_intelligence_models import (
    BoundingRegion,
    Span,
    TableCell,
    Table,
    DocumentPage,
    AnalyzeResult,
    DocumentIntelligenceResponse
)

__all__ = [
    "Base64FileRequest",
    "ExtractionResponse",
    "ExtractedContent",
    "OutlineExtractionResponse",
    "DocumentInput",
    "MistralOCRRequest",
    "MistralOCRResponse",
    "MistralErrorResponse",
    "Page",
    "Dimensions",
    "UsageInfo",
    "BoundingRegion",
    "Span",
    "TableCell",
    "Table",
    "DocumentPage",
    "AnalyzeResult",
    "DocumentIntelligenceResponse"
]
