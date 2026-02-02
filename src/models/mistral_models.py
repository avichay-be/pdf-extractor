"""
Pydantic models for Mistral Document AI OCR API requests and responses.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class DocumentInput(BaseModel):
    """Document input for Mistral OCR API."""

    type: str = Field(default="document_url", description="Type of document input")
    document_url: str = Field(..., description="Base64 encoded document URL")

    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate type is document_url."""
        if v != "document_url":
            raise ValueError("Type must be 'document_url'")
        return v

    @field_validator('document_url')
    @classmethod
    def validate_document_url(cls, v: str) -> str:
        """Validate document URL starts with correct prefix."""
        if not v.startswith("data:application/pdf;base64,"):
            raise ValueError("Document URL must start with 'data:application/pdf;base64,'")
        return v


class MistralOCRRequest(BaseModel):
    """Request model for Mistral Document AI OCR API."""

    model: str = Field(default="mistral-document-ai-2505", description="Model identifier")
    document: DocumentInput = Field(..., description="Document to process")
    include_image_base64: bool = Field(
        default=False,
        description="Whether to include image base64 in response"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model": "mistral-document-ai-2505",
                "document": {
                    "type": "document_url",
                    "document_url": "data:application/pdf;base64,JVBERi0xLjQK..."
                },
                "include_image_base64": True
            }
        }
    }


class Dimensions(BaseModel):
    """Page dimensions."""

    dpi: int = Field(..., description="Dots per inch")
    height: int = Field(..., description="Height in pixels")
    width: int = Field(..., description="Width in pixels")


class Page(BaseModel):
    """Single page content from OCR."""

    index: int = Field(..., description="Page index (0-based)")
    markdown: str = Field(..., description="Markdown content of the page")
    dimensions: Dimensions = Field(..., description="Page dimensions")
    images: Optional[List[Dict[str, Any]]] = Field(None, description="Images detected on the page")


class UsageInfo(BaseModel):
    """Usage information for the API call."""

    pages_processed: int = Field(..., description="Number of pages processed")
    doc_size_bytes: int = Field(..., description="Document size in bytes")
    pages_processed_annotation: int = Field(..., description="Pages processed for annotation")


class MistralOCRResponse(BaseModel):
    """
    Response model from Mistral Document AI OCR API.

    The actual API returns pages with index, markdown, and dimensions.
    """

    pages: List[Page] = Field(..., description="List of processed pages")
    model: str = Field(..., description="Model used for processing")
    document_annotation: Optional[str] = Field(None, description="Document-level annotation")
    usage_info: UsageInfo = Field(..., description="Usage information")
    content_filter_results: Optional[Dict[str, Any]] = Field(None, description="Content filter results")

    @property
    def content(self) -> str:
        """Combine all page markdown into a single document with page numbers."""
        if not self.pages:
            return ""

        # Sort pages by index and combine markdown with page number headers
        sorted_pages = sorted(self.pages, key=lambda p: p.index)
        markdown_parts = []
        for page in sorted_pages:
            # Add page number header (1-based indexing for user display)
            page_header = f"# Page {page.index + 1}\n\n"
            markdown_parts.append(page_header + page.markdown)

        return "\n\n".join(markdown_parts)

    model_config = {
        "json_schema_extra": {
            "example": {
                "pages": [
                    {
                        "index": 0,
                        "markdown": "# Document Title\n\nContent...",
                        "dimensions": {
                            "dpi": 72,
                            "height": 1654,
                            "width": 2339
                        }
                    }
                ],
                "model": "mistral-document-ai-2505",
                "usage_info": {
                    "pages_processed": 1,
                    "doc_size_bytes": 12345,
                    "pages_processed_annotation": 0
                }
            }
        }
    }


class MistralErrorResponse(BaseModel):
    """Error response from Mistral API."""

    error: Dict[str, Any] = Field(..., description="Error details")

    @property
    def message(self) -> str:
        """Extract error message."""
        if isinstance(self.error, dict):
            return self.error.get('message', str(self.error))
        return str(self.error)

    @property
    def type(self) -> str:
        """Extract error type."""
        if isinstance(self.error, dict):
            return self.error.get('type', 'unknown_error')
        return 'unknown_error'

    @property
    def code(self) -> Optional[str]:
        """Extract error code."""
        if isinstance(self.error, dict):
            return self.error.get('code')
        return None
