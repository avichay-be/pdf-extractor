"""
Unit tests for Azure OpenAI Document Client.

This file tests the OpenAIDocumentClient class which integrates with Azure OpenAI GPT-4o
for PDF content extraction. Tests cover:
- Client initialization and configuration
- API version detection (Chat API vs Responses API)
- PDF page to image conversion
- Content extraction via both API formats
- Error handling and fallback behaviors
"""
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import base64
from pathlib import Path

from src.services.openai_client import OpenAIDocumentClient


class TestOpenAIDocumentClient(unittest.TestCase):
    """Test cases for OpenAI Document Client initialization and configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_api_key = "test_api_key_12345"
        self.test_endpoint = "https://test.openai.azure.com"
        self.test_deployment = "gpt-4o"
        self.test_api_version = "2024-02-15-preview"

    # ========== Initialization Tests ==========

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_init_with_default_settings(self, mock_azure_openai, mock_settings):
        """Test client initializes with settings from config."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint
        mock_settings.AZURE_OPENAI_DEPLOYMENT = self.test_deployment
        mock_settings.AZURE_OPENAI_API_VERSION = self.test_api_version

        # Execute
        client = OpenAIDocumentClient()

        # Assert
        self.assertEqual(client.api_key, self.test_api_key)
        self.assertEqual(client.endpoint, self.test_endpoint)
        self.assertEqual(client.deployment, self.test_deployment)
        self.assertEqual(client.api_version, self.test_api_version)

        # Verify AzureOpenAI client was created with correct params
        mock_azure_openai.assert_called_once_with(
            api_key=self.test_api_key,
            azure_endpoint=self.test_endpoint,
            api_version=self.test_api_version
        )

    @patch('src.services.openai_client.AzureOpenAI')
    def test_init_with_custom_parameters(self, mock_azure_openai):
        """Test client initializes with custom parameters overriding settings."""
        # Execute
        client = OpenAIDocumentClient(
            api_key=self.test_api_key,
            endpoint=self.test_endpoint,
            deployment=self.test_deployment,
            api_version=self.test_api_version
        )

        # Assert
        self.assertEqual(client.api_key, self.test_api_key)
        self.assertEqual(client.endpoint, self.test_endpoint)
        self.assertEqual(client.deployment, self.test_deployment)
        self.assertEqual(client.api_version, self.test_api_version)

    @patch('src.services.openai_client.settings')
    def test_init_missing_credentials_raises_error(self, mock_settings):
        """Test initialization fails when API key or endpoint is missing."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = None
        mock_settings.AZURE_OPENAI_ENDPOINT = None

        # Execute & Assert
        with self.assertRaises(ValueError) as context:
            OpenAIDocumentClient()

        self.assertIn("API key and endpoint must be provided", str(context.exception))

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    @patch('src.services.openai_client.logger')
    def test_init_logs_deployment_info(self, mock_logger, mock_azure_openai, mock_settings):
        """Test initialization logs deployment information."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint
        mock_settings.AZURE_OPENAI_DEPLOYMENT = self.test_deployment

        # Execute
        OpenAIDocumentClient()

        # Assert
        mock_logger.info.assert_called_with(f"Initialized OpenAI client with deployment: {self.test_deployment}")

    # ========== API Version Detection Tests ==========

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_is_responses_api_with_new_version(self, mock_azure_openai, mock_settings):
        """Test Responses API detection with 2025-02-01-preview or later."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        # Execute
        client = OpenAIDocumentClient(api_version="2025-02-01-preview")

        # Assert
        self.assertTrue(client._is_responses_api())

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_is_responses_api_with_old_version(self, mock_azure_openai, mock_settings):
        """Test Chat API detection with versions before 2025-02-01."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        # Execute
        client = OpenAIDocumentClient(api_version="2024-02-15-preview")

        # Assert
        self.assertFalse(client._is_responses_api())

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    @patch('src.services.openai_client.logger')
    def test_is_responses_api_with_invalid_version(self, mock_logger, mock_azure_openai, mock_settings):
        """Test API detection with invalid version format."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        # Execute
        client = OpenAIDocumentClient(api_version="invalid-format")

        # Assert
        self.assertFalse(client._is_responses_api())
        mock_logger.warning.assert_called_once()
        self.assertIn("Could not parse API version", mock_logger.warning.call_args[0][0])

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_is_responses_api_with_none_version(self, mock_azure_openai, mock_settings):
        """Test API detection with None version."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        # Execute
        client = OpenAIDocumentClient(api_version=None)

        # Assert
        self.assertFalse(client._is_responses_api())

    # ========== PDF to Image Conversion Tests ==========

    @patch('src.services.openai_client.fitz')
    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_pdf_page_to_images_success(self, mock_azure_openai, mock_settings, mock_fitz):
        """Test successful PDF page to image conversion."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        # Mock PDF document and page
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 3  # 3 pages
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b'fake_png_bytes'
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()  # Mock transformation matrix

        # Execute
        client = OpenAIDocumentClient()
        test_pdf_bytes = b'%PDF-1.4 test'
        images = client._pdf_page_to_images(test_pdf_bytes, page_number=0)

        # Assert
        self.assertEqual(len(images), 1)
        self.assertIsInstance(images[0], str)  # Base64 string

        # Verify can decode base64
        decoded = base64.b64decode(images[0])
        self.assertIsInstance(decoded, bytes)

        # Verify PDF was opened and closed
        mock_fitz.open.assert_called_once_with(stream=test_pdf_bytes, filetype="pdf")
        mock_doc.close.assert_called_once()

    @patch('src.services.openai_client.fitz')
    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_pdf_page_to_images_invalid_page_number(self, mock_azure_openai, mock_settings, mock_fitz):
        """Test error handling for invalid page number."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        # Mock PDF with 3 pages
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 3
        mock_fitz.open.return_value = mock_doc

        # Execute & Assert
        client = OpenAIDocumentClient()
        with self.assertRaises(ValueError) as context:
            client._pdf_page_to_images(b'test', page_number=5)

        self.assertIn("Page 5 does not exist", str(context.exception))
        self.assertIn("PDF has 3 pages", str(context.exception))

    @patch('src.services.openai_client.fitz')
    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_pdf_page_to_images_with_custom_dpi(self, mock_azure_openai, mock_settings, mock_fitz):
        """Test PDF conversion with custom DPI setting."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.tobytes.return_value = b'fake_png'
        mock_page.get_pixmap.return_value = mock_pixmap
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc

        # Mock Matrix to capture zoom parameter
        mock_matrix = MagicMock()
        mock_fitz.Matrix.return_value = mock_matrix

        # Execute
        client = OpenAIDocumentClient()
        client._pdf_page_to_images(b'test', page_number=0, dpi=300)

        # Assert - verify zoom factor is dpi/72
        expected_zoom = 300 / 72
        mock_fitz.Matrix.assert_called_once_with(expected_zoom, expected_zoom)

    @patch('src.services.openai_client.fitz')
    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    @patch('src.services.openai_client.logger')
    def test_pdf_page_to_images_conversion_failure(self, mock_logger, mock_azure_openai, mock_settings, mock_fitz):
        """Test error handling when PDF conversion fails."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        # Mock fitz.open to raise exception
        mock_fitz.open.side_effect = Exception("PDF conversion error")

        # Execute & Assert
        client = OpenAIDocumentClient()
        with self.assertRaises(Exception) as context:
            client._pdf_page_to_images(b'corrupt_pdf', page_number=0)

        self.assertIn("PDF conversion error", str(context.exception))
        mock_logger.error.assert_called_once()

    # ========== Chat API Extraction Tests ==========

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_extract_with_chat_api_success(self, mock_azure_openai_class, mock_settings):
        """Test successful extraction using Chat Completions API."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint
        mock_settings.get_system_prompt.return_value = "You are a helpful assistant."
        mock_settings.get_user_prompt_template.return_value = "Extract content from page {page_number}"

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "# Test Markdown\n\nTest content"
        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_openai_class.return_value = mock_client

        # Execute
        client = OpenAIDocumentClient(api_version="2024-02-15-preview")  # Use old version for Chat API
        result = client._extract_with_chat_api(["fake_base64_image"], page_number=0)

        # Assert
        self.assertEqual(result, "# Test Markdown\n\nTest content")

        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args

        # Check messages structure
        self.assertIn('messages', call_args.kwargs)
        messages = call_args.kwargs['messages']
        self.assertEqual(len(messages), 2)  # system + user
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')

        # Check temperature and max_tokens
        self.assertEqual(call_args.kwargs['temperature'], 0.0)
        self.assertEqual(call_args.kwargs['max_tokens'], 4096)

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_extract_with_chat_api_multiple_images(self, mock_azure_openai_class, mock_settings):
        """Test Chat API with multiple images."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint
        mock_settings.get_system_prompt.return_value = "Test system prompt"
        mock_settings.get_user_prompt_template.return_value = "Extract page {page_number}"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test content"
        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_openai_class.return_value = mock_client

        # Execute
        client = OpenAIDocumentClient(api_version="2024-02-15-preview")
        images = ["image1_base64", "image2_base64", "image3_base64"]
        client._extract_with_chat_api(images, page_number=0)

        # Assert
        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args.kwargs['messages'][1]
        user_content = user_message['content']

        # Should have 1 text + 3 images = 4 items
        self.assertEqual(len(user_content), 4)

        # First should be text
        self.assertEqual(user_content[0]['type'], 'text')

        # Rest should be images
        for i in range(1, 4):
            self.assertEqual(user_content[i]['type'], 'image_url')
            self.assertIn('data:image/png;base64,', user_content[i]['image_url']['url'])

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_extract_with_chat_api_uses_settings_prompts(self, mock_azure_openai_class, mock_settings):
        """Test Chat API uses prompts from settings."""
        # Setup
        custom_system = "Custom system prompt"
        custom_user = "Custom user prompt for page {page_number}"
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint
        mock_settings.get_system_prompt.return_value = custom_system
        mock_settings.get_user_prompt_template.return_value = custom_user

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_client.chat.completions.create.return_value = mock_response
        mock_azure_openai_class.return_value = mock_client

        # Execute
        client = OpenAIDocumentClient(api_version="2024-02-15-preview")
        client._extract_with_chat_api(["test_image"], page_number=5)

        # Assert
        mock_settings.get_system_prompt.assert_called_with("openai")
        mock_settings.get_user_prompt_template.assert_called_with("openai")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs['messages']

        # Check system prompt
        self.assertEqual(messages[0]['content'], custom_system)

        # Check user prompt has page number formatted (6 = 5+1 for 1-based)
        user_content = messages[1]['content']
        self.assertIn("Custom user prompt for page 6", user_content[0]['text'])

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    @patch('src.services.openai_client.logger')
    def test_extract_with_chat_api_failure(self, mock_logger, mock_azure_openai_class, mock_settings):
        """Test error handling when Chat API fails."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_azure_openai_class.return_value = mock_client

        # Execute & Assert
        client = OpenAIDocumentClient(api_version="2024-02-15-preview")
        with self.assertRaises(Exception):
            client._extract_with_chat_api(["test_image"], page_number=0)

    # ========== Responses API Extraction Tests ==========

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    def test_extract_with_responses_api_success(self, mock_azure_openai_class, mock_settings):
        """Test successful extraction using Responses API."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint
        mock_settings.get_system_prompt.return_value = "System prompt"
        mock_settings.get_user_prompt_template.return_value = "Extract page {page_number}"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.output_text = "# Markdown\n\nContent from Responses API"
        mock_client.responses.create.return_value = mock_response
        mock_azure_openai_class.return_value = mock_client

        # Execute
        client = OpenAIDocumentClient(api_version="2025-02-01-preview")  # Use new version for Responses API
        result = client._extract_with_responses_api(["fake_base64"], page_number=0)

        # Assert
        self.assertEqual(result, "# Markdown\n\nContent from Responses API")

        # Verify API call
        mock_client.responses.create.assert_called_once()
        call_args = mock_client.responses.create.call_args

        # Check input structure
        self.assertIn('input', call_args.kwargs)
        input_data = call_args.kwargs['input']
        self.assertEqual(len(input_data), 2)  # system + user
        self.assertEqual(input_data[0]['role'], 'system')
        self.assertEqual(input_data[1]['role'], 'user')

        # Check user content has input_text and input_image types
        user_content = input_data[1]['content']
        self.assertEqual(user_content[0]['type'], 'input_text')
        self.assertEqual(user_content[1]['type'], 'input_image')

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    @patch('src.services.openai_client.logger')
    def test_extract_with_responses_api_fallback_to_choices(self, mock_logger, mock_azure_openai_class, mock_settings):
        """Test Responses API fallback when output_text not available."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint
        mock_settings.get_system_prompt.return_value = "Prompt"
        mock_settings.get_user_prompt_template.return_value = "Extract {page_number}"

        mock_client = MagicMock()
        mock_response = MagicMock()
        # No output_text attribute
        del mock_response.output_text
        # But has choices (fallback)
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Fallback content"
        mock_client.responses.create.return_value = mock_response
        mock_azure_openai_class.return_value = mock_client

        # Execute
        client = OpenAIDocumentClient(api_version="2025-02-01-preview")
        result = client._extract_with_responses_api(["fake_base64"], page_number=0)

        # Assert
        self.assertEqual(result, "Fallback content")

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    @patch('src.services.openai_client.logger')
    def test_extract_with_responses_api_fallback_to_str(self, mock_logger, mock_azure_openai_class, mock_settings):
        """Test Responses API final fallback to str(response)."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint
        mock_settings.get_system_prompt.return_value = "Prompt"
        mock_settings.get_user_prompt_template.return_value = "Extract {page_number}"

        mock_client = MagicMock()
        mock_response = MagicMock()
        # No output_text
        del mock_response.output_text
        # No choices either
        del mock_response.choices
        # str(response) will be used
        mock_response.__str__.return_value = "String representation of response"
        mock_client.responses.create.return_value = mock_response
        mock_azure_openai_class.return_value = mock_client

        # Execute
        client = OpenAIDocumentClient(api_version="2025-02-01-preview")
        result = client._extract_with_responses_api(["fake_base64"], page_number=0)

        # Assert
        self.assertIn("String representation", result)
        # Should log warning about fallback
        mock_logger.warning.assert_called_once()
        self.assertIn("fallback", mock_logger.warning.call_args[0][0].lower())

    # ========== Integration Tests ==========

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    @patch.object(OpenAIDocumentClient, '_pdf_page_to_images')
    @patch.object(OpenAIDocumentClient, '_extract_with_chat_api')
    def test_extract_page_content_routes_to_chat_api(
        self, mock_extract_chat, mock_pdf_to_images, mock_azure_openai, mock_settings
    ):
        """Test extract_page_content routes to Chat API for old versions."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        mock_pdf_to_images.return_value = ["image1"]
        mock_extract_chat.return_value = "Chat API result"

        # Execute
        client = OpenAIDocumentClient(api_version="2024-02-15-preview")
        result = client.extract_page_content(b'test_pdf', page_number=0)

        # Assert
        self.assertEqual(result, "Chat API result")
        mock_extract_chat.assert_called_once_with(["image1"], 0)
        # _extract_with_responses_api should NOT be called
        self.assertFalse(hasattr(client, '_extract_with_responses_api_called'))

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    @patch.object(OpenAIDocumentClient, '_pdf_page_to_images')
    @patch.object(OpenAIDocumentClient, '_extract_with_responses_api')
    def test_extract_page_content_routes_to_responses_api(
        self, mock_extract_responses, mock_pdf_to_images, mock_azure_openai, mock_settings
    ):
        """Test extract_page_content routes to Responses API for new versions."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        mock_pdf_to_images.return_value = ["image1"]
        mock_extract_responses.return_value = "Responses API result"

        # Execute
        client = OpenAIDocumentClient(api_version="2025-02-01-preview")
        result = client.extract_page_content(b'test_pdf', page_number=0)

        # Assert
        self.assertEqual(result, "Responses API result")
        mock_extract_responses.assert_called_once_with(["image1"], 0)

    @patch('src.services.openai_client.settings')
    @patch('src.services.openai_client.AzureOpenAI')
    @patch.object(OpenAIDocumentClient, '_pdf_page_to_images')
    @patch('src.services.openai_client.logger')
    def test_extract_page_content_error_handling(
        self, mock_logger, mock_pdf_to_images, mock_azure_openai, mock_settings
    ):
        """Test error handling in extract_page_content."""
        # Setup
        mock_settings.AZURE_OPENAI_API_KEY = self.test_api_key
        mock_settings.AZURE_OPENAI_ENDPOINT = self.test_endpoint

        mock_pdf_to_images.side_effect = Exception("PDF processing error")

        # Execute & Assert
        client = OpenAIDocumentClient()
        with self.assertRaises(Exception) as context:
            client.extract_page_content(b'test_pdf', page_number=0)

        self.assertIn("PDF processing error", str(context.exception))
        mock_logger.error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
