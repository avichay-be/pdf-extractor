"""
Unit tests for API models (Base64FileRequest and ExtractionResponse).
"""
import unittest
import base64
from pydantic import ValidationError

from src.models.api_models import Base64FileRequest, ExtractionResponse


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
            file_content=self.valid_base64
        )
        self.assertEqual(request.filename, "test.pdf")
        self.assertEqual(request.file_content, self.valid_base64)

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
            file_content=self.valid_base64
        )
        dumped = request.model_dump()
        self.assertEqual(dumped['filename'], "test.pdf")
        self.assertEqual(dumped['file_content'], self.valid_base64)


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


if __name__ == '__main__':
    unittest.main()
