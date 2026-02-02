"""
Unit tests for Google Gemini Document Client.

This file tests the GeminiDocumentClient class which integrates with Google Gemini Flash
for PDF content extraction. Tests cover:
- Client initialization and API key handling
- PDF page extraction
- Content extraction with Gemini API
- Error handling
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import os

from src.services.gemini_client import GeminiDocumentClient


class TestGeminiDocumentClient(unittest.TestCase):
    """Test cases for Gemini Document Client."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_api_key = "test_gemini_api_key_12345"
        self.test_model = "gemini-2.5-flash"

    # ========== Initialization Tests ==========

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    def test_init_with_default_settings(self, mock_genai_client, mock_settings):
        """Test client initializes with settings from config."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key
        mock_settings.GEMINI_MODEL = self.test_model

        # Execute
        client = GeminiDocumentClient()

        # Assert
        self.assertEqual(client.api_key, self.test_api_key)
        self.assertEqual(client.model_name, self.test_model)

        # Verify genai.Client was created with API key
        mock_genai_client.assert_called_once_with(api_key=self.test_api_key)

    @patch('src.services.gemini_client.genai.Client')
    def test_init_with_custom_parameters(self, mock_genai_client):
        """Test client initializes with custom parameters overriding settings."""
        custom_key = "custom_api_key"
        custom_model = "gemini-1.5-pro"

        # Execute
        client = GeminiDocumentClient(
            api_key=custom_key,
            model_name=custom_model
        )

        # Assert
        self.assertEqual(client.api_key, custom_key)
        self.assertEqual(client.model_name, custom_model)
        mock_genai_client.assert_called_once_with(api_key=custom_key)

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'env_test_key'})
    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    def test_init_with_env_variable(self, mock_genai_client, mock_settings):
        """Test client uses environment variable when settings return None."""
        # Setup
        mock_settings.GEMINI_API_KEY = None  # Force use of env var
        mock_settings.GEMINI_MODEL = self.test_model

        # Execute
        client = GeminiDocumentClient()

        # Assert
        self.assertEqual(client.api_key, 'env_test_key')
        mock_genai_client.assert_called_once_with(api_key='env_test_key')

    @patch('src.services.gemini_client.settings')
    @patch.dict(os.environ, {}, clear=True)
    def test_init_missing_api_key_raises_error(self, mock_settings):
        """Test initialization fails when API key is not provided."""
        # Setup
        mock_settings.GEMINI_API_KEY = None

        # Execute & Assert
        with self.assertRaises(ValueError) as context:
            GeminiDocumentClient()

        self.assertIn("Gemini API key must be provided", str(context.exception))
        self.assertIn("GEMINI_API_KEY", str(context.exception))

    # ========== PDF Page Extraction Tests ==========

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch('src.services.gemini_client.fitz')
    def test_extract_single_page_pdf_success(self, mock_fitz, mock_genai_client, mock_settings):
        """Test successful extraction of single page from PDF."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key

        # Mock PDF document with 3 pages
        mock_source_doc = MagicMock()
        mock_source_doc.__len__.return_value = 3

        mock_target_doc = MagicMock()
        mock_target_doc.tobytes.return_value = b'single_page_pdf_bytes'

        # Setup fitz.open to return different docs for different calls
        mock_fitz.open.side_effect = [mock_source_doc, mock_target_doc]

        # Execute
        client = GeminiDocumentClient()
        pdf_bytes = b'%PDF-1.4 test multi-page PDF'
        result = client._extract_single_page_pdf(pdf_bytes, page_number=1)

        # Assert
        self.assertEqual(result, b'single_page_pdf_bytes')

        # Verify PDF operations
        self.assertEqual(mock_fitz.open.call_count, 2)  # Once for source, once for target

        # Verify insert_pdf was called with correct page range
        mock_target_doc.insert_pdf.assert_called_once_with(
            mock_source_doc,
            from_page=1,
            to_page=1
        )

        # Verify both docs were closed
        mock_source_doc.close.assert_called_once()
        mock_target_doc.close.assert_called_once()

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch('src.services.gemini_client.fitz')
    def test_extract_single_page_pdf_invalid_page(self, mock_fitz, mock_genai_client, mock_settings):
        """Test error handling for invalid page number."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key

        # Mock PDF with 3 pages
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 3
        mock_fitz.open.return_value = mock_doc

        # Execute & Assert
        client = GeminiDocumentClient()
        with self.assertRaises(ValueError) as context:
            client._extract_single_page_pdf(b'test_pdf', page_number=5)

        self.assertIn("Page 5 does not exist", str(context.exception))
        self.assertIn("PDF has 3 pages", str(context.exception))

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch('src.services.gemini_client.fitz')
    def test_extract_single_page_pdf_first_page(self, mock_fitz, mock_genai_client, mock_settings):
        """Test extracting first page (page 0)."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key

        mock_source_doc = MagicMock()
        mock_source_doc.__len__.return_value = 5
        mock_target_doc = MagicMock()
        mock_target_doc.tobytes.return_value = b'first_page'
        mock_fitz.open.side_effect = [mock_source_doc, mock_target_doc]

        # Execute
        client = GeminiDocumentClient()
        result = client._extract_single_page_pdf(b'test_pdf', page_number=0)

        # Assert
        self.assertEqual(result, b'first_page')
        mock_target_doc.insert_pdf.assert_called_once_with(
            mock_source_doc,
            from_page=0,
            to_page=0
        )

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch('src.services.gemini_client.fitz')
    def test_extract_single_page_pdf_last_page(self, mock_fitz, mock_genai_client, mock_settings):
        """Test extracting last page."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key

        mock_source_doc = MagicMock()
        mock_source_doc.__len__.return_value = 5  # Pages 0-4
        mock_target_doc = MagicMock()
        mock_target_doc.tobytes.return_value = b'last_page'
        mock_fitz.open.side_effect = [mock_source_doc, mock_target_doc]

        # Execute
        client = GeminiDocumentClient()
        result = client._extract_single_page_pdf(b'test_pdf', page_number=4)

        # Assert
        self.assertEqual(result, b'last_page')
        mock_target_doc.insert_pdf.assert_called_once_with(
            mock_source_doc,
            from_page=4,
            to_page=4
        )

    # ========== Content Extraction Tests ==========

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch.object(GeminiDocumentClient, '_extract_single_page_pdf')
    def test_extract_page_content_success(self, mock_extract_page, mock_genai_client_class, mock_settings):
        """Test successful content extraction from PDF page."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key
        mock_settings.GEMINI_MODEL = self.test_model
        mock_settings.get_system_prompt.return_value = "You are a helpful assistant."
        mock_settings.get_user_prompt_template.return_value = "Extract page {page_number}"

        # Mock Gemini client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Test Markdown\n\nExtracted content from Gemini"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_client_class.return_value = mock_client

        # Mock page extraction
        mock_extract_page.return_value = b'single_page_pdf_bytes'

        # Execute
        client = GeminiDocumentClient()
        result = client.extract_page_content(b'test_pdf_bytes', page_number=2)

        # Assert
        self.assertEqual(result, "# Test Markdown\n\nExtracted content from Gemini")

        # Verify page was extracted
        mock_extract_page.assert_called_once_with(b'test_pdf_bytes', 2)

        # Verify API call
        mock_client.models.generate_content.assert_called_once()
        call_args = mock_client.models.generate_content.call_args

        # Check model name
        self.assertEqual(call_args.kwargs['model'], self.test_model)

        # Check contents structure
        self.assertIn('contents', call_args.kwargs)
        contents = call_args.kwargs['contents']
        self.assertEqual(len(contents), 2)  # PDF part + prompt

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch('src.services.gemini_client.types.Part')
    @patch.object(GeminiDocumentClient, '_extract_single_page_pdf')
    def test_extract_page_content_uses_pdf_mime_type(
        self, mock_extract_page, mock_part, mock_genai_client_class, mock_settings
    ):
        """Test content extraction uses correct PDF MIME type."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key
        mock_settings.get_system_prompt.return_value = "Prompt"
        mock_settings.get_user_prompt_template.return_value = "Extract {page_number}"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_client_class.return_value = mock_client

        mock_extract_page.return_value = b'pdf_bytes'
        mock_part_instance = MagicMock()
        mock_part.from_bytes.return_value = mock_part_instance

        # Execute
        client = GeminiDocumentClient()
        client.extract_page_content(b'test', page_number=0)

        # Assert
        mock_part.from_bytes.assert_called_once_with(
            data=b'pdf_bytes',
            mime_type='application/pdf'
        )

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch.object(GeminiDocumentClient, '_extract_single_page_pdf')
    def test_extract_page_content_prompt_formatting(
        self, mock_extract_page, mock_genai_client_class, mock_settings
    ):
        """Test page number is correctly formatted in prompt (1-based)."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key
        mock_settings.get_system_prompt.return_value = "System"
        mock_settings.get_user_prompt_template.return_value = "Page {page_number} content"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_client_class.return_value = mock_client

        mock_extract_page.return_value = b'pdf'

        # Execute - request page 5 (0-based)
        client = GeminiDocumentClient()
        client.extract_page_content(b'test', page_number=5)

        # Assert - should be formatted as page 6 (1-based)
        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs['contents']

        # The prompt should be the last item (after PDF part)
        prompt = contents[-1]
        self.assertIn("Page 6 content", prompt)

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch.object(GeminiDocumentClient, '_extract_single_page_pdf')
    def test_extract_page_content_combines_prompts(
        self, mock_extract_page, mock_genai_client_class, mock_settings
    ):
        """Test system and user prompts are combined with newlines."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key
        custom_system = "Custom system instructions"
        custom_user = "Custom user prompt for page {page_number}"
        mock_settings.get_system_prompt.return_value = custom_system
        mock_settings.get_user_prompt_template.return_value = custom_user

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Test"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_client_class.return_value = mock_client

        mock_extract_page.return_value = b'pdf'

        # Execute
        client = GeminiDocumentClient()
        client.extract_page_content(b'test', page_number=0)

        # Assert
        mock_settings.get_system_prompt.assert_called_with("gemini")
        mock_settings.get_user_prompt_template.assert_called_with("gemini")

        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs['contents']
        full_prompt = contents[-1]

        # Should contain both prompts separated by newlines
        self.assertIn(custom_system, full_prompt)
        self.assertIn("Custom user prompt for page 1", full_prompt)
        self.assertIn("\n\n", full_prompt)  # Prompts separated by newlines

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch('src.services.gemini_client.logger')
    @patch.object(GeminiDocumentClient, '_extract_single_page_pdf')
    def test_extract_page_content_logs_progress(
        self, mock_extract_page, mock_logger, mock_genai_client_class, mock_settings
    ):
        """Test extraction logs start and completion messages."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Test content with 123 characters total"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai_client_class.return_value = mock_client

        mock_extract_page.return_value = b'pdf'

        # Execute
        client = GeminiDocumentClient()
        client.extract_page_content(b'test', page_number=3)

        # Assert logging calls
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]

        # Should log start
        self.assertTrue(any("Extracting page 3" in msg and "Gemini" in msg for msg in log_calls))

        # Should log completion with character count
        self.assertTrue(any("Successfully extracted page 3" in msg for msg in log_calls))
        self.assertTrue(any("chars" in msg for msg in log_calls))

    @patch('src.services.gemini_client.settings')
    @patch('src.services.gemini_client.genai.Client')
    @patch('src.services.gemini_client.logger')
    @patch.object(GeminiDocumentClient, '_extract_single_page_pdf')
    def test_extract_page_content_api_failure(
        self, mock_extract_page, mock_logger, mock_genai_client_class, mock_settings
    ):
        """Test error handling when Gemini API fails."""
        # Setup
        mock_settings.GEMINI_API_KEY = self.test_api_key

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Gemini API error")
        mock_genai_client_class.return_value = mock_client

        mock_extract_page.return_value = b'pdf'

        # Execute & Assert
        client = GeminiDocumentClient()
        with self.assertRaises(Exception) as context:
            client.extract_page_content(b'test', page_number=2)

        self.assertIn("Gemini API error", str(context.exception))

        # Should log error with page number
        mock_logger.error.assert_called_once()
        error_msg = mock_logger.error.call_args[0][0]
        self.assertIn("Failed to extract page 2", error_msg)
        self.assertIn("Gemini", error_msg)


if __name__ == '__main__':
    unittest.main()
