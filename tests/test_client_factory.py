"""
Unit tests for ClientFactory.

Tests the factory pattern, lazy loading, workflow mapping, and singleton behavior.
"""
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

from src.services.client_factory import ClientFactory, get_client_factory


class TestClientFactory(unittest.TestCase):
    """Test cases for ClientFactory class."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear the global singleton before each test
        import src.services.client_factory
        src.services.client_factory._client_factory = None

    def tearDown(self):
        """Clean up after tests."""
        # Clear the global singleton after each test
        import src.services.client_factory
        src.services.client_factory._client_factory = None

    # ============================================================================
    # Initialization Tests (2 tests)
    # ============================================================================

    def test_factory_initialization(self):
        """Test that factory initializes with all clients set to None."""
        factory = ClientFactory()

        self.assertIsNone(factory._pdf_processor)
        self.assertIsNone(factory._mistral_client)
        self.assertIsNone(factory._openai_client)
        self.assertIsNone(factory._gemini_client)
        self.assertIsNone(factory._azure_di_client)

    def test_singleton_pattern(self):
        """Test that get_client_factory returns the same instance."""
        factory1 = get_client_factory()
        factory2 = get_client_factory()

        self.assertIs(factory1, factory2, "get_client_factory should return same instance")

    # ============================================================================
    # Property Lazy Loading Tests (5 tests)
    # ============================================================================

    @patch('src.services.client_factory.PDFProcessor')
    def test_pdf_processor_lazy_loading(self, mock_pdf_processor_class):
        """Test that pdf_processor is lazily initialized."""
        mock_processor = MagicMock()
        mock_pdf_processor_class.return_value = mock_processor

        factory = ClientFactory()

        # First access should create the instance
        result1 = factory.pdf_processor
        self.assertEqual(result1, mock_processor)
        mock_pdf_processor_class.assert_called_once()

        # Second access should return the same instance
        result2 = factory.pdf_processor
        self.assertEqual(result2, mock_processor)
        mock_pdf_processor_class.assert_called_once()  # Still only called once

    @patch('src.services.client_factory.settings')
    @patch('src.services.client_factory.MistralDocumentClient')
    def test_mistral_client_lazy_loading(self, mock_mistral_class, mock_settings):
        """Test that mistral_client is lazily initialized."""
        mock_settings.AZURE_API_KEY = "test_api_key"
        mock_client = MagicMock()
        mock_mistral_class.return_value = mock_client

        factory = ClientFactory()

        # First access should create the instance
        result1 = factory.mistral_client
        self.assertEqual(result1, mock_client)
        mock_mistral_class.assert_called_once_with(api_key="test_api_key")

        # Second access should return the same instance
        result2 = factory.mistral_client
        self.assertEqual(result2, mock_client)
        mock_mistral_class.assert_called_once()  # Still only called once

    @patch('src.services.client_factory.OpenAIDocumentClient')
    def test_openai_client_success(self, mock_openai_class):
        """Test successful OpenAI client initialization."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        factory = ClientFactory()

        result = factory.openai_client
        self.assertEqual(result, mock_client)
        mock_openai_class.assert_called_once()

    @patch('src.services.client_factory.OpenAIDocumentClient')
    def test_openai_client_failure(self, mock_openai_class):
        """Test OpenAI client initialization failure returns None."""
        mock_openai_class.side_effect = ValueError("Missing API key")

        factory = ClientFactory()

        result = factory.openai_client
        self.assertIsNone(result)

    @patch('src.services.client_factory.GeminiDocumentClient')
    def test_gemini_client_failure(self, mock_gemini_class):
        """Test Gemini client initialization failure returns None."""
        mock_gemini_class.side_effect = ValueError("Missing API key")

        factory = ClientFactory()

        result = factory.gemini_client
        self.assertIsNone(result)

    @patch('src.services.client_factory.AzureDocumentIntelligenceClient')
    def test_azure_di_client_success(self, mock_azure_di_class):
        """Test successful Azure DI client initialization."""
        mock_client = MagicMock()
        mock_azure_di_class.return_value = mock_client

        factory = ClientFactory()

        result = factory.azure_document_intelligence_client
        self.assertEqual(result, mock_client)
        mock_azure_di_class.assert_called_once()

    # ============================================================================
    # Workflow Mapping Tests (6 tests)
    # ============================================================================

    @patch('src.services.client_factory.settings')
    @patch('src.services.client_factory.MistralDocumentClient')
    def test_get_client_for_workflow_mistral(self, mock_mistral_class, mock_settings):
        """Test getting client for mistral workflow."""
        mock_settings.AZURE_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_mistral_class.return_value = mock_client

        factory = ClientFactory()
        result = factory.get_client_for_workflow("mistral")

        self.assertEqual(result, mock_client)

    @patch('src.services.client_factory.OpenAIDocumentClient')
    def test_get_client_for_workflow_openai(self, mock_openai_class):
        """Test getting client for openai workflow."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        factory = ClientFactory()
        result = factory.get_client_for_workflow("openai")

        self.assertEqual(result, mock_client)

    @patch('src.services.client_factory.GeminiDocumentClient')
    def test_get_client_for_workflow_gemini(self, mock_gemini_class):
        """Test getting client for gemini workflow."""
        mock_client = MagicMock()
        mock_gemini_class.return_value = mock_client

        factory = ClientFactory()
        result = factory.get_client_for_workflow("gemini")

        self.assertEqual(result, mock_client)

    @patch('src.services.client_factory.GeminiDocumentClient')
    def test_get_client_for_workflow_gemini_wf(self, mock_gemini_class):
        """Test getting client for gemini-wf workflow (should use same client as gemini)."""
        mock_client = MagicMock()
        mock_gemini_class.return_value = mock_client

        factory = ClientFactory()
        result = factory.get_client_for_workflow("gemini-wf")

        self.assertEqual(result, mock_client)

    @patch('src.services.client_factory.AzureDocumentIntelligenceClient')
    def test_get_client_for_workflow_azure_di(self, mock_azure_di_class):
        """Test getting client for azure_document_intelligence workflow."""
        mock_client = MagicMock()
        mock_azure_di_class.return_value = mock_client

        factory = ClientFactory()
        result = factory.get_client_for_workflow("azure_document_intelligence")

        self.assertEqual(result, mock_client)

    def test_get_client_for_workflow_unknown(self):
        """Test that unknown workflow raises ValueError."""
        factory = ClientFactory()

        with self.assertRaises(ValueError) as context:
            factory.get_client_for_workflow("unknown_workflow")

        self.assertIn("Unknown workflow type", str(context.exception))

    # ============================================================================
    # Caching Tests (2 tests)
    # ============================================================================

    @patch('src.services.client_factory.settings')
    @patch('src.services.client_factory.MistralDocumentClient')
    def test_client_caching(self, mock_mistral_class, mock_settings):
        """Test that clients are cached and not recreated on repeated access."""
        mock_settings.AZURE_API_KEY = "test_key"
        mock_client = MagicMock()
        mock_mistral_class.return_value = mock_client

        factory = ClientFactory()

        # Access the same client multiple times
        result1 = factory.mistral_client
        result2 = factory.mistral_client
        result3 = factory.get_client_for_workflow("mistral")

        # All should be the same instance
        self.assertIs(result1, result2)
        self.assertIs(result2, result3)

        # Client should only be initialized once
        mock_mistral_class.assert_called_once()

    @patch('src.services.client_factory.PDFProcessor')
    @patch('src.services.client_factory.settings')
    @patch('src.services.client_factory.MistralDocumentClient')
    def test_init_only_once(self, mock_mistral_class, mock_settings, mock_pdf_processor_class):
        """Test that each client is only initialized once even with multiple property calls."""
        mock_settings.AZURE_API_KEY = "test_key"
        mock_mistral = MagicMock()
        mock_processor = MagicMock()
        mock_mistral_class.return_value = mock_mistral
        mock_pdf_processor_class.return_value = mock_processor

        factory = ClientFactory()

        # Access each property multiple times
        _ = factory.pdf_processor
        _ = factory.pdf_processor
        _ = factory.mistral_client
        _ = factory.mistral_client

        # Each should only be initialized once
        mock_pdf_processor_class.assert_called_once()
        mock_mistral_class.assert_called_once()


if __name__ == '__main__':
    unittest.main()
