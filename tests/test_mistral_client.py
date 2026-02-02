"""
Unit tests for Mistral API client.
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock
import base64
import tempfile
from pathlib import Path

import httpx
from pydantic import ValidationError

from src.services.mistral_client import MistralDocumentClient
from src.models.mistral_models import (
    MistralOCRRequest,
    MistralOCRResponse,
    MistralErrorResponse,
    Page,
    Dimensions,
    UsageInfo
)


class TestMistralDocumentClient(unittest.IsolatedAsyncioTestCase):
    """Test cases for MistralDocumentClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test_api_key_12345"
        self.client = MistralDocumentClient(api_key=self.api_key)

    def test_init_default_values(self):
        """Test client initialization with default values."""
        self.assertEqual(self.client.api_key, self.api_key)
        self.assertIn("Bearer", self.client.headers["Authorization"])
        self.assertEqual(self.client.model, "mistral-document-ai-2505")
        self.assertEqual(self.client.timeout, 120.0)

    def test_init_custom_values(self):
        """Test client initialization with custom values."""
        custom_url = "https://custom.api.com/ocr"
        custom_model = "custom-model"
        custom_timeout = 60.0

        client = MistralDocumentClient(
            api_key=self.api_key,
            api_url=custom_url,
            model=custom_model,
            timeout=custom_timeout
        )

        self.assertEqual(client.api_url, custom_url)
        self.assertEqual(client.model, custom_model)
        self.assertEqual(client.timeout, custom_timeout)

    def test_encode_pdf_to_base64(self):
        """Test PDF to base64 encoding."""
        # Create a temporary PDF-like file
        test_content = b"%PDF-1.4\ntest content"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name

        try:
            encoded = self.client._encode_pdf_to_base64(tmp_path)

            # Verify it's base64 encoded
            decoded = base64.b64decode(encoded)
            self.assertEqual(decoded, test_content)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def test_process_document_file_not_found(self):
        """Test processing raises error for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            await self.client.process_document('/fake/path/document.pdf')

    @patch('httpx.AsyncClient')
    async def test_process_document_success(self, mock_client_class):
        """Test successful document processing."""
        # Create a temporary PDF file
        test_pdf_content = b"%PDF-1.4\ntest"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(test_pdf_content)
            tmp_path = tmp.name

        try:
            # Mock successful API response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model": "mistral-document-ai-2505",
                "pages": [
                    {
                        "index": 0,
                        "markdown": "# Test Document\n\nContent here.",
                        "dimensions": {"dpi": 72, "height": 1000, "width": 800}
                    }
                ],
                "usage_info": {
                    "pages_processed": 1,
                    "doc_size_bytes": 100,
                    "pages_processed_annotation": 0
                }
            }

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result, validation = await self.client.process_document(tmp_path)

            # Verify result
            self.assertIn("# Test Document", result)
            self.assertIn("Content here.", result)

            # Verify API was called
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            self.assertEqual(call_args[0][0], self.client.api_url)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch('httpx.AsyncClient')
    async def test_process_document_api_error(self, mock_client_class):
        """Test handling of API error response."""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(b"%PDF-1.4\ntest")
            tmp_path = tmp.name

        try:
            # Mock error API response
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {
                "error": {
                    "message": "Invalid document format",
                    "type": "invalid_request_error",
                    "code": "invalid_document"
                }
            }
            mock_response.text = '{"error": {"message": "Invalid document format"}}'

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            with self.assertRaises(ValueError) as context:
                await self.client.process_document(tmp_path)

            msg = str(context.exception)
            if "Invalid document format" not in msg:
                self.fail(f"Expected 'Invalid document format' in '{msg}'")

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch('httpx.AsyncClient')
    async def test_process_document_multiple_pages(self, mock_client_class):
        """Test processing document with multiple pages."""
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(b"%PDF-1.4\ntest")
            tmp_path = tmp.name

        try:
            # Mock API response with multiple pages
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "model": "mistral-document-ai-2505",
                "pages": [
                    {
                        "index": 0,
                        "markdown": "# Page 1\n\nFirst page content.",
                        "dimensions": {"dpi": 72, "height": 1000, "width": 800}
                    },
                    {
                        "index": 1,
                        "markdown": "# Page 2\n\nSecond page content.",
                        "dimensions": {"dpi": 72, "height": 1000, "width": 800}
                    }
                ],
                "usage_info": {
                    "pages_processed": 2,
                    "doc_size_bytes": 200,
                    "pages_processed_annotation": 0
                }
            }

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result, validation = await self.client.process_document(tmp_path)

            # Verify both pages are in result
            self.assertIn("# Page 1", result)
            self.assertIn("First page content", result)
            self.assertIn("# Page 2", result)
            self.assertIn("Second page content", result)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch('httpx.AsyncClient')
    async def test_health_check_success(self, mock_client_class):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.status_code = 200

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.client.health_check()

        self.assertTrue(result)

    @patch('httpx.AsyncClient')
    async def test_health_check_failure(self, mock_client_class):
        """Test failed health check."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection failed")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        result = await self.client.health_check()

        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
