"""
Pydantic models for API request and response structures.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import base64


class Base64FileRequest(BaseModel):
    """Request model for base64-encoded PDF file."""

    filename: str = Field(..., description="Name of the PDF file (e.g., 'document.pdf')")
    file_content: str = Field(..., description="Base64-encoded PDF file content")
    request_id: Optional[str] = Field(
        default=None,
        description="Optional request ID for /extract-json. If omitted or null, the server generates one."
    )
    enable_cross_validation: Optional[bool] = Field(
        default=None,
        description="Enable or disable cross-validation explicitly. If None, the global ENABLE_CROSS_VALIDATION setting is used."
    )
    enable_polishing: Optional[bool] = Field(
        default=None,
        description="Enable or disable the Gemini polish pass. If None, uses the global SMART_EXTRACTION_POLISH_ENABLED setting."
    )

    @field_validator('filename')
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """Validate filename ends with .pdf."""
        if not v.lower().endswith('.pdf'):
            raise ValueError("Filename must end with .pdf")
        return v

    @field_validator('file_content')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """
        Validate that file_content is valid base64 format.

        Performance optimized: Only validates format without decoding entire payload.
        This saves 100-300ms for large PDFs during request validation.
        """
        if not v:
            raise ValueError("file_content cannot be empty")

        # Check if string contains only valid base64 characters
        # Valid base64: A-Z, a-z, 0-9, +, /, = (padding)
        import re
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', v):
            raise ValueError("file_content must be valid base64-encoded string")

        # Basic length check: base64 length must be multiple of 4
        if len(v) % 4 != 0:
            raise ValueError("file_content must be valid base64-encoded string (invalid length)")

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "document.pdf",
                "file_content": "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlL0NhdGFsb2...",
                "request_id": "req-123",
                "enable_cross_validation": True,
                "enable_polishing": True
            }
        }
    }


class ExtractionResponse(BaseModel):
    """Response model for PDF extraction (legacy - single content)."""

    filename: str = Field(..., description="Original filename of the processed PDF")
    content: str = Field(..., description="Extracted markdown content from the PDF")

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "document.pdf",
                "content": "# Document Title\n\nContent extracted from the PDF..."
            }
        }
    }


class ExtractedContent(BaseModel):
    """Single extracted content section."""

    filename: str = Field(..., description="Filename for this section")
    content: str = Field(..., description="Extracted markdown content for this section")

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "outline1_document.pdf",
                "content": "# Section 1\n\nContent..."
            }
        }
    }


class PageRouteMetadata(BaseModel):
    """Compact per-page routing metadata for API responses."""

    page_number: int = Field(..., description="1-based page number")
    page_range: List[int] = Field(..., description="Inclusive 1-based page range")
    strategy: str = Field(..., description="Business strategy used for the page")
    source: str = Field(..., description="Primary extraction source")
    confidence: float = Field(..., description="Confidence score for the selected result")
    ocr_mode: Optional[str] = Field(None, description="OCR mode used for the page when OCR was applied")
    validation_source: Optional[str] = Field(None, description="Secondary source used for validation, when available")


class ValidationSummary(BaseModel):
    """Normalized validation summary exposed in response metadata."""

    enabled: Optional[bool] = Field(None, description="Whether validation ran")
    mode: Optional[str] = Field(None, description="Validation mode used")
    status: Optional[str] = Field(None, description="Validation status")
    dual_source_pages: Optional[int] = Field(None, description="Number of pages with dual-source validation")
    total_results: Optional[int] = Field(None, description="Number of page results considered")
    source_distribution: Optional[Dict[str, int]] = Field(
        None,
        description="Count of final page results by selected source"
    )


class ExtractionResponseMetadata(BaseModel):
    """Rich workflow metadata returned with extraction responses."""

    workflow: Optional[str] = Field(None, description="Business workflow used for the document")
    ocr_mode: Optional[str] = Field(None, description="OCR subtype used for the document when applicable")
    routing_strategy: Optional[str] = Field(None, description="Routing strategy used for page selection")
    total_pages: Optional[int] = Field(None, description="Total number of pages analyzed")
    page_types: Optional[Dict[str, int]] = Field(None, description="Count of pages by detected page type")
    page_routes: Optional[List[PageRouteMetadata]] = Field(
        None,
        description="Per-page routing metadata"
    )
    timing: Optional[Dict[str, float]] = Field(None, description="Phase timing information in seconds")
    validation_summary: Optional[ValidationSummary] = Field(
        None,
        description="Summary of validation behavior for the extraction"
    )
    source_file_name: Optional[str] = Field(None, description="Original source PDF filename")


class OutlineExtractionResponse(BaseModel):
    """Response model for PDF extraction with outline-based splitting."""

    request_id: str = Field(..., description="Request ID used for tracing this extraction")
    file_name: str = Field(..., description="Original filename")
    request_time: datetime = Field(..., description="When the request was received")
    timestamp: datetime = Field(..., description="When processing completed")
    model: str = Field(default="pdf-extractor-v2", description="Model version")
    extracted_content: List[ExtractedContent] = Field(
        ...,
        description="Array of extracted content sections (one per outline, max 4)"
    )
    metadata: ExtractionResponseMetadata = Field(
        ...,
        description="Rich workflow metadata for the extraction"
    )
    validation: Optional[Dict[str, str]] = Field(
        None,
        description="Simple validation status (if enabled). Contains 'enabled' and 'status' fields. Status values: 'passed', 'problems_fixed', or 'warnings'."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req-123",
                "file_name": "document.pdf",
                "request_time": "2025-11-06T14:30:30Z",
                "timestamp": "2025-11-06T14:30:45Z",
                "model": "pdf-extractor-v2",
                "extracted_content": [
                    {
                        "filename": "outline1_document.pdf",
                        "content": "# Section 1\n\nContent..."
                    },
                    {
                        "filename": "outline2_document.pdf",
                        "content": "# Section 2\n\nContent..."
                    }
                ],
                "metadata": {
                    "workflow": "ocr",
                    "ocr_mode": "tables",
                    "routing_strategy": "page_aware",
                    "total_pages": 12,
                    "page_types": {
                        "text_only": 4,
                        "image_only": 8
                    },
                    "page_routes": [
                        {
                            "page_number": 1,
                            "page_range": [1, 1],
                            "strategy": "text_extraction",
                            "source": "pdfplumber",
                            "confidence": 1.0
                        }
                    ],
                    "timing": {
                        "total_s": 3.42
                    },
                    "validation_summary": {
                        "enabled": True,
                        "mode": "mixed_page_cross_validation",
                        "status": "passed",
                        "dual_source_pages": 2,
                        "total_results": 12,
                        "source_distribution": {
                            "pdfplumber": 4,
                            "mistral": 8
                        }
                    },
                    "source_file_name": "document.pdf"
                },
                "validation": {
                    "enabled": "true",
                    "status": "passed"
                }
            }
        }
    }
