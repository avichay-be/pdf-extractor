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
    query: str = Field(
        default="01_Fin_Reports",
        description="Filter outline sections by name and select workflow. Default: 01_Fin_Reports (Financial Reports with Mistral)"
    )
    model: Optional[str] = Field(
        default="mistral",
        description="Model to use for extraction. Options: 'mistral', 'openai', 'gemini'. Validation will automatically use a different model."
    )
    enable_validation: Optional[bool] = Field(
        default=None,
        description="Enable cross-validation. If None, uses global ENABLE_CROSS_VALIDATION setting. Set to true/false to override."
    )

    @field_validator('model')
    @classmethod
    def validate_model(cls, v: Optional[str]) -> str:
        """Validate model is one of the supported options."""
        if v is None:
            return "mistral"  # Default

        valid_models = ["mistral", "openai", "gemini"]
        if v.lower() not in valid_models:
            raise ValueError(f"Model must be one of: {', '.join(valid_models)}")
        return v.lower()

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
                "query": "דוחות כספיים",
                "model": "mistral",
                "enable_validation": True
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


class OutlineExtractionResponse(BaseModel):
    """Response model for PDF extraction with outline-based splitting."""

    file_name: str = Field(..., description="Original filename")
    request_time: datetime = Field(..., description="When the request was received")
    timestamp: datetime = Field(..., description="When processing completed")
    model: str = Field(default="pdf-extractor-v2", description="Model version")
    extracted_content: List[ExtractedContent] = Field(
        ...,
        description="Array of extracted content sections (one per outline, max 4)"
    )
    validation: Optional[Dict[str, str]] = Field(
        None,
        description="Simple validation status (if enabled). Contains 'enabled' and 'status' fields. Status values: 'passed', 'problems_fixed', or 'warnings'."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "validation": {
                    "enabled": "true",
                    "status": "passed"
                }
            }
        }
    }
