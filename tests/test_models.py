"""
Unit tests for Pydantic models.
"""
import unittest
from pydantic import ValidationError

from src.models.mistral_models import (
    DocumentInput,
    MistralOCRRequest,
    MistralOCRResponse,
    MistralErrorResponse,
    Page,
    Dimensions,
    UsageInfo
)


class TestDocumentInput(unittest.TestCase):
    """Test cases for DocumentInput model."""

    def test_valid_document_input(self):
        """Test creating valid document input."""
        doc = DocumentInput(
            type="document_url",
            document_url="data:application/pdf;base64,JVBERi0xLjQK"
        )

        self.assertEqual(doc.type, "document_url")
        self.assertTrue(doc.document_url.startswith("data:application/pdf;base64,"))

    def test_invalid_type(self):
        """Test validation fails for invalid type."""
        with self.assertRaises(ValidationError):
            DocumentInput(
                type="invalid_type",
                document_url="data:application/pdf;base64,JVBERi0xLjQK"
            )

    def test_invalid_document_url_prefix(self):
        """Test validation fails for invalid document URL prefix."""
        with self.assertRaises(ValidationError):
            DocumentInput(
                type="document_url",
                document_url="https://example.com/doc.pdf"
            )

    def test_default_type(self):
        """Test default type is set correctly."""
        doc = DocumentInput(document_url="data:application/pdf;base64,test")
        self.assertEqual(doc.type, "document_url")


class TestMistralOCRRequest(unittest.TestCase):
    """Test cases for MistralOCRRequest model."""

    def test_valid_request(self):
        """Test creating valid OCR request."""
        request = MistralOCRRequest(
            model="mistral-document-ai-2505",
            document=DocumentInput(
                document_url="data:application/pdf;base64,JVBERi0xLjQK"
            ),
            include_image_base64=True
        )

        self.assertEqual(request.model, "mistral-document-ai-2505")
        self.assertTrue(request.include_image_base64)
        self.assertEqual(request.document.type, "document_url")

    def test_default_values(self):
        """Test default values are set correctly."""
        request = MistralOCRRequest(
            document=DocumentInput(
                document_url="data:application/pdf;base64,test"
            )
        )

        self.assertEqual(request.model, "mistral-document-ai-2505")
        self.assertFalse(request.include_image_base64)  # Default is False

    def test_model_dump(self):
        """Test model serialization."""
        request = MistralOCRRequest(
            document=DocumentInput(
                document_url="data:application/pdf;base64,test"
            )
        )

        data = request.model_dump()

        self.assertIn("model", data)
        self.assertIn("document", data)
        self.assertIn("include_image_base64", data)
        self.assertEqual(data["model"], "mistral-document-ai-2505")


class TestPage(unittest.TestCase):
    """Test cases for Page model."""

    def test_valid_page(self):
        """Test creating valid page."""
        page = Page(
            index=0,
            markdown="# Test Page\n\nContent here.",
            dimensions=Dimensions(dpi=72, height=1000, width=800)
        )

        self.assertEqual(page.index, 0)
        self.assertIn("# Test Page", page.markdown)
        self.assertEqual(page.dimensions.height, 1000)


class TestMistralOCRResponse(unittest.TestCase):
    """Test cases for MistralOCRResponse model."""

    def setUp(self):
        self.usage_info = UsageInfo(
            pages_processed=1,
            doc_size_bytes=1000,
            pages_processed_annotation=0
        )
        self.dimensions = Dimensions(dpi=72, height=1000, width=800)

    def test_valid_response(self):
        """Test creating valid OCR response."""
        response = MistralOCRResponse(
            model="mistral-document-ai-2505",
            pages=[
                Page(
                    index=0,
                    markdown="# Page 1",
                    dimensions=self.dimensions
                )
            ],
            usage_info=self.usage_info
        )

        self.assertEqual(response.model, "mistral-document-ai-2505")
        self.assertEqual(len(response.pages), 1)
        self.assertEqual(response.pages[0].index, 0)

    def test_content_property_single_page(self):
        """Test content property with single page."""
        response = MistralOCRResponse(
            model="mistral-document-ai-2505",
            pages=[
                Page(
                    index=0,
                    markdown="# Test Document\n\nContent.",
                    dimensions=self.dimensions
                )
            ],
            usage_info=self.usage_info
        )

        content = response.content
        self.assertIn("# Test Document", content)
        self.assertIn("Content.", content)

    def test_content_property_multiple_pages(self):
        """Test content property with multiple pages."""
        response = MistralOCRResponse(
            model="mistral-document-ai-2505",
            pages=[
                Page(index=0, markdown="# Page 1", dimensions=self.dimensions),
                Page(index=1, markdown="# Page 2", dimensions=self.dimensions),
                Page(index=2, markdown="# Page 3", dimensions=self.dimensions)
            ],
            usage_info=self.usage_info
        )

        content = response.content

        self.assertIn("# Page 1", content)
        self.assertIn("# Page 2", content)
        self.assertIn("# Page 3", content)

        # Check pages are joined with double newlines
        self.assertIn("\n\n", content)

    def test_content_property_empty_pages(self):
        """Test content with no pages."""
        response = MistralOCRResponse(
            model="mistral-document-ai-2505",
            pages=[],
            usage_info=self.usage_info
        )

        self.assertEqual(response.content, "")

    def test_content_property_unordered_pages(self):
        """Test content sorts pages correctly."""
        response = MistralOCRResponse(
            model="mistral-document-ai-2505",
            pages=[
                Page(index=2, markdown="# Page 3", dimensions=self.dimensions),
                Page(index=0, markdown="# Page 1", dimensions=self.dimensions),
                Page(index=1, markdown="# Page 2", dimensions=self.dimensions)
            ],
            usage_info=self.usage_info
        )

        content = response.content

        # Verify order is correct
        page1_pos = content.find("# Page 1")
        page2_pos = content.find("# Page 2")
        page3_pos = content.find("# Page 3")

        self.assertLess(page1_pos, page2_pos)
        self.assertLess(page2_pos, page3_pos)


class TestMistralErrorResponse(unittest.TestCase):
    """Test cases for MistralErrorResponse model."""

    def test_valid_error_response(self):
        """Test creating valid error response."""
        error = MistralErrorResponse(
            error={
                "message": "Invalid request",
                "type": "invalid_request_error",
                "code": "invalid_input"
            }
        )

        self.assertEqual(error.message, "Invalid request")
        self.assertEqual(error.type, "invalid_request_error")
        self.assertEqual(error.code, "invalid_input")

    def test_error_message_property(self):
        """Test error message extraction."""
        error = MistralErrorResponse(
            error={"message": "Test error message"}
        )

        self.assertEqual(error.message, "Test error message")

    def test_error_type_property(self):
        """Test error type extraction."""
        error = MistralErrorResponse(
            error={"type": "authentication_error"}
        )

        self.assertEqual(error.type, "authentication_error")

    def test_error_code_property(self):
        """Test error code extraction."""
        error = MistralErrorResponse(
            error={"code": "rate_limit_exceeded"}
        )

        self.assertEqual(error.code, "rate_limit_exceeded")

    def test_missing_fields_defaults(self):
        """Test default values when fields are missing."""
        error = MistralErrorResponse(error={})

        self.assertEqual(error.type, "unknown_error")
        self.assertIsNone(error.code)


if __name__ == '__main__':
    unittest.main()
