"""
Client factory for initializing PDF extraction clients.

Centralizes client initialization logic and error handling.
"""
import logging
from typing import Optional

from src.core.config import settings
from src.services.mistral_client import MistralDocumentClient
from src.services.openai_client import OpenAIDocumentClient
from src.services.gemini_client import GeminiDocumentClient
from src.services.azure_di import AzureDocumentIntelligenceClient
from src.services.pdf_processor import PDFProcessor

logger = logging.getLogger(__name__)


class ClientFactory:
    """Factory for creating and managing PDF extraction clients."""

    def __init__(self):
        """Initialize the client factory."""
        self._pdf_processor = None
        self._mistral_client = None
        self._openai_client = None
        self._gemini_client = None
        self._azure_di_client = None

    @property
    def pdf_processor(self) -> PDFProcessor:
        """Get or create PDF processor instance."""
        if self._pdf_processor is None:
            self._pdf_processor = PDFProcessor()
            logger.info("PDF processor initialized")
        return self._pdf_processor

    @property
    def mistral_client(self) -> MistralDocumentClient:
        """Get or create Mistral client instance."""
        if self._mistral_client is None:
            self._mistral_client = MistralDocumentClient(api_key=settings.AZURE_API_KEY)
            logger.info("Mistral client initialized")
        return self._mistral_client

    @property
    def openai_client(self) -> Optional[OpenAIDocumentClient]:
        """
        Get or create OpenAI client instance.

        Returns None if initialization fails (e.g., API key not configured).
        """
        if self._openai_client is None:
            try:
                self._openai_client = OpenAIDocumentClient()
                logger.info("OpenAI client initialized")
            except Exception as e:
                logger.warning(f"OpenAI client not available: {e}")
                return None
        return self._openai_client

    @property
    def gemini_client(self) -> Optional[GeminiDocumentClient]:
        """
        Get or create Gemini client instance.

        Returns None if initialization fails (e.g., API key not configured).
        """
        if self._gemini_client is None:
            try:
                self._gemini_client = GeminiDocumentClient()
                logger.info("Gemini client initialized")
            except Exception as e:
                logger.warning(f"Gemini client not available: {e}")
                return None
        return self._gemini_client

    @property
    def azure_document_intelligence_client(self) -> Optional[AzureDocumentIntelligenceClient]:
        """
        Get or create Azure Document Intelligence client instance.

        Returns None if initialization fails (e.g., API key not configured).
        """
        if self._azure_di_client is None:
            try:
                self._azure_di_client = AzureDocumentIntelligenceClient()
                logger.info("Azure Document Intelligence client initialized")
            except Exception as e:
                logger.warning(f"Azure Document Intelligence client not available: {e}")
                return None
        return self._azure_di_client

    def get_client_for_workflow(self, workflow: str):
        """
        Get the appropriate client for a given workflow.

        Args:
            workflow: Workflow type string (e.g., "mistral", "openai", "gemini", "gemini-wf", etc.)

        Returns:
            Client instance for the workflow, or None if not available

        Raises:
            ValueError: If workflow type is unknown
        """
        workflow_map = {
            "mistral": self.mistral_client,
            "openai": self.openai_client,
            "gemini": self.gemini_client,
            "gemini-wf": self.gemini_client,  # Uses same client as gemini
            "azure_document_intelligence": self.azure_document_intelligence_client,
        }

        if workflow not in workflow_map:
            raise ValueError(f"Unknown workflow type: {workflow}")

        return workflow_map[workflow]


# Global singleton instance
_client_factory: Optional[ClientFactory] = None


def get_client_factory() -> ClientFactory:
    """
    Get the global client factory instance.

    Returns:
        Singleton ClientFactory instance
    """
    global _client_factory
    if _client_factory is None:
        _client_factory = ClientFactory()
    return _client_factory
