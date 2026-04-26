"""
Unit tests for API models (Base64FileRequest and ExtractionResponse).
"""
import unittest
import base64
from pydantic import ValidationError

from datetime import datetime, timezone

from src.models.api_models import (
    Base64FileRequest,
    ExtractionResponse,
    ExtractedContent,
    ExtractionResponseMetadata,
    OutlineExtractionResponse,
)


class TestBase64FileRequest(unittest.TestCase):
    """Test cases for Base64FileRequest model."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple base64 string (encoding "test pdf content")
        self.valid_base64 = base64.b64encode(b"test pdf content").decode('utf-8')

    def test_valid_request(self):
        """Test creating request with valid data."""
        request = Base64FileRequest(
            filename="test.pdf",
            file_content=self.valid_base64,
            request_id="req-123",
            enable_cross_validation=False,
            enable_polishing=True,
        )
        self.assertEqual(request.filename, "test.pdf")
        self.assertEqual(request.file_content, self.valid_base64)
        self.assertEqual(request.request_id, "req-123")
        self.assertFalse(request.enable_cross_validation)
        self.assertTrue(request.enable_polishing)

    def test_filename_without_pdf_extension(self):
        """Test that filename must end with .pdf."""
        with self.assertRaises(ValidationError) as context:
            Base64FileRequest(
                filename="test.txt",
                file_content=self.valid_base64
            )
        self.assertIn("must end with .pdf", str(context.exception))

    def test_filename_case_insensitive(self):
        """Test that .PDF extension works (case insensitive)."""
        request = Base64FileRequest(
            filename="test.PDF",
            file_content=self.valid_base64
        )
        self.assertEqual(request.filename, "test.PDF")

    def test_invalid_base64(self):
        """Test that invalid base64 content raises error."""
        with self.assertRaises(ValidationError) as context:
            Base64FileRequest(
                filename="test.pdf",
                file_content="not valid base64!!!"
            )
        self.assertIn("must be valid base64", str(context.exception))

    def test_empty_filename(self):
        """Test that empty filename raises error."""
        with self.assertRaises(ValidationError):
            Base64FileRequest(
                filename="",
                file_content=self.valid_base64
            )

    def test_missing_fields(self):
        """Test that missing required fields raise errors."""
        with self.assertRaises(ValidationError):
            Base64FileRequest(filename="test.pdf")

        with self.assertRaises(ValidationError):
            Base64FileRequest(file_content=self.valid_base64)

    def test_hebrew_filename(self):
        """Test that non-ASCII filenames work."""
        request = Base64FileRequest(
            filename="מסמך.pdf",
            file_content=self.valid_base64
        )
        self.assertEqual(request.filename, "מסמך.pdf")

    def test_model_dump(self):
        """Test that model can be serialized to dict."""
        request = Base64FileRequest(
            filename="test.pdf",
            file_content=self.valid_base64,
            request_id="req-123",
            enable_cross_validation=False,
            enable_polishing=True,
        )
        dumped = request.model_dump()
        self.assertEqual(dumped['filename'], "test.pdf")
        self.assertEqual(dumped['file_content'], self.valid_base64)
        self.assertEqual(dumped['request_id'], "req-123")
        self.assertFalse(dumped['enable_cross_validation'])
        self.assertTrue(dumped['enable_polishing'])

    def test_legacy_model_field_is_ignored(self):
        """Legacy provider model field should no longer be part of the schema."""
        request = Base64FileRequest(
            filename="test.pdf",
            file_content=self.valid_base64,
            model="mistral",
        )
        dumped = request.model_dump()
        self.assertNotIn("model", dumped)

    def test_legacy_query_field_is_ignored(self):
        """Legacy query field should no longer be part of the schema."""
        request = Base64FileRequest(
            filename="test.pdf",
            file_content=self.valid_base64,
            query="01_Fin_Reports",
        )
        dumped = request.model_dump()
        self.assertNotIn("query", dumped)

    def test_legacy_enable_validation_field_is_ignored(self):
        """Legacy validation flag should no longer be part of the public schema."""
        request = Base64FileRequest(
            filename="test.pdf",
            file_content=self.valid_base64,
            enable_validation=True,
        )
        dumped = request.model_dump()
        self.assertNotIn("enable_validation", dumped)


class TestExtractionResponse(unittest.TestCase):
    """Test cases for ExtractionResponse model."""

    def test_valid_response(self):
        """Test creating response with valid data."""
        response = ExtractionResponse(
            filename="test.pdf",
            content="# Test Document\n\nThis is a test."
        )
        self.assertEqual(response.filename, "test.pdf")
        self.assertEqual(response.content, "# Test Document\n\nThis is a test.")

    def test_empty_content(self):
        """Test that empty content is allowed."""
        response = ExtractionResponse(
            filename="test.pdf",
            content=""
        )
        self.assertEqual(response.content, "")

    def test_unicode_content(self):
        """Test that Unicode content works."""
        hebrew_content = "# כותרת\n\nתוכן בעברית"
        response = ExtractionResponse(
            filename="test.pdf",
            content=hebrew_content
        )
        self.assertEqual(response.content, hebrew_content)

    def test_missing_fields(self):
        """Test that missing required fields raise errors."""
        with self.assertRaises(ValidationError):
            ExtractionResponse(filename="test.pdf")

        with self.assertRaises(ValidationError):
            ExtractionResponse(content="content")

    def test_model_dump(self):
        """Test that model can be serialized to dict."""
        response = ExtractionResponse(
            filename="test.pdf",
            content="Test content"
        )
        dumped = response.model_dump()
        self.assertEqual(dumped['filename'], "test.pdf")
        self.assertEqual(dumped['content'], "Test content")

    def test_json_serialization(self):
        """Test that model can be serialized to JSON."""
        response = ExtractionResponse(
            filename="test.pdf",
            content="Test content"
        )
        json_str = response.model_dump_json()
        self.assertIn("test.pdf", json_str)
        self.assertIn("Test content", json_str)


class TestOutlineExtractionResponse(unittest.TestCase):
    """Test the rich JSON extraction response model."""

    def test_response_includes_request_id_and_metadata(self):
        response = OutlineExtractionResponse(
            request_id="req-123",
            file_name="test.pdf",
            request_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc),
            extracted_content=[
                ExtractedContent(filename="test.md", content="# Title")
            ],
            metadata=ExtractionResponseMetadata(
                workflow="ocr",
                ocr_mode="tables",
                source_file_name="test.pdf",
            ),
            validation={"enabled": "true", "status": "passed"},
        )

        dumped = response.model_dump()
        self.assertEqual(dumped["request_id"], "req-123")
        self.assertEqual(dumped["metadata"]["workflow"], "ocr")
        self.assertEqual(dumped["metadata"]["ocr_mode"], "tables")


if __name__ == '__main__':
    unittest.main()
