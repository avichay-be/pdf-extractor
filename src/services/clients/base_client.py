"""
Base client for document processing APIs.

This module provides an abstract base class for all document processing clients,
establishing a consistent interface and shared functionality.
"""
from abc import ABC, abstractmethod
from typing import Optional
import httpx
import logging

from src.core.error_handling import ClientConfigurationError

logger = logging.getLogger(__name__)


class BaseDocumentClient(ABC):
    """Abstract base class for all document processing clients.

    Provides common functionality for API clients including:
    - HTTP client management with connection pooling
    - Async context manager support
    - Credential validation
    - Health check interface
    - Page content extraction interface

    Subclasses must implement:
    - _validate_credentials(): Validate API credentials
    - extract_page_content(): Extract content from a single page
    - health_check(): Check if API is accessible
    """

    def __init__(
        self,
        api_key: str,
        endpoint: Optional[str] = None,
        timeout: float = 120.0
    ):
        """Initialize the document client.

        Args:
            api_key: API key for authentication (required)
            endpoint: API endpoint URL (optional, subclass may have default)
            timeout: Request timeout in seconds (default: 120.0)

        Raises:
            ClientConfigurationError: If credentials are invalid
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

        # Validate credentials (implemented by subclass)
        self._validate_credentials()

        logger.info(f"Initialized {self.__class__.__name__} with timeout={timeout}s")

    @abstractmethod
    def _validate_credentials(self) -> None:
        """Validate API credentials.

        Must be implemented by subclasses to check if required credentials
        (API key, endpoint, etc.) are properly configured.

        Raises:
            ClientConfigurationError: If credentials are missing or invalid
        """
        pass

    async def __aenter__(self):
        """Async context manager entry.

        Creates and initializes the HTTP client.

        Returns:
            Self for use in 'async with' statements

        Example:
            async with client:
                result = await client.extract_page_content(pdf_bytes, 1)
        """
        self._client = self._create_client()
        logger.debug(f"{self.__class__.__name__} context manager entered")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.

        Ensures HTTP client is properly closed, even if exceptions occur.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        await self.close()
        logger.debug(f"{self.__class__.__name__} context manager exited")

    async def close(self):
        """Close the HTTP client and release resources.

        Safe to call multiple times. Should be called when the client
        is no longer needed, or use async context manager instead.
        """
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.debug(f"{self.__class__.__name__} HTTP client closed")

    def _create_client(self) -> httpx.AsyncClient:
        """Create HTTP client with connection pooling.

        Configures connection limits for efficient connection reuse
        and resource management.

        Returns:
            Configured httpx.AsyncClient instance
        """
        return httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20
            )
        )

    @abstractmethod
    async def extract_page_content(
        self,
        pdf_bytes: bytes,
        page_number: int,
        prompt: Optional[str] = None
    ) -> str:
        """Extract content from a single PDF page.

        Must be implemented by subclasses to call their specific API
        and return the extracted content in markdown format.

        Args:
            pdf_bytes: PDF file content as bytes
            page_number: Page number to extract (1-based indexing)
            prompt: Optional custom prompt for extraction

        Returns:
            Extracted content in markdown format

        Raises:
            ValueError: If page_number is invalid
            ClientConfigurationError: If client not properly configured
            Exception: API-specific errors
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the API is accessible and responding.

        Must be implemented by subclasses to verify their specific
        API endpoint is reachable and functioning.

        Returns:
            True if API is healthy, False otherwise

        Note:
            Should not raise exceptions - return False on any error
        """
        pass

    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"{self.__class__.__name__}("
            f"endpoint={self.endpoint}, "
            f"timeout={self.timeout}s"
            ")"
        )
