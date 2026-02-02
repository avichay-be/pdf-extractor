"""
Mistral Document AI client for processing PDFs via Azure OCR endpoint.
"""
import logging
import time
import asyncio
from pathlib import Path
from typing import Optional

import httpx
from pydantic import ValidationError

from src.models.mistral_models import (
    MistralOCRRequest,
    MistralOCRResponse,
    MistralErrorResponse,
    DocumentInput
)
from src.core.config import settings
from src.core.utils import encode_pdf_to_base64
from src.core.http_client import get_async_client, request_with_retry

logger = logging.getLogger(__name__)


# Import validation service (lazy import to avoid circular dependencies)
def _get_validation_service():
    """Lazy import of validation service."""
    from src.services.validation import ValidationService
    return ValidationService()


class MistralDocumentClient:
    """Client for interacting with Mistral Document AI via Azure."""

    def __init__(
        self,
        api_key: str,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0
    ):
        """
        Initialize Mistral client with shared HTTP connection pool.

        Args:
            api_key: Azure API key
            api_url: Optional custom API URL
            model: Optional custom model name
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or settings.AZURE_API_KEY
        self.api_url = api_url or settings.MISTRAL_API_URL
        self.model = model or settings.MISTRAL_MODEL
        self.timeout = timeout

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Shared HTTP client with connection pooling for performance
        # This reuses connections across multiple requests, reducing overhead
        self._client: Optional[httpx.AsyncClient] = None

        # Rate limiting - track last request time to enforce minimum interval
        self._last_request_time: float = 0.0
        self._rate_limit_lock = asyncio.Lock()  # Protect against concurrent rate limit checks

    async def __aenter__(self):
        """Async context manager entry - initialize shared client."""
        self._client = get_async_client(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup shared client."""
        await self.close()

    async def close(self):
        """Close the shared HTTP client and cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client instance."""
        if self._client is None:
            # Create temporary client if not using context manager
            # Note: For best performance, use this class as a context manager
            return get_async_client(timeout=self.timeout)
        return self._client

    async def _enforce_rate_limit(self):
        """
        Enforce rate limiting by waiting if necessary.

        Ensures minimum interval between API requests to respect the
        60 requests/minute limit (1 request per second).
        """
        async with self._rate_limit_lock:
            current_time = time.time()
            time_since_last_request = current_time - self._last_request_time

            if time_since_last_request < settings.MISTRAL_MIN_REQUEST_INTERVAL:
                wait_time = settings.MISTRAL_MIN_REQUEST_INTERVAL - time_since_last_request
                logger.info(f"Rate limiting: waiting {wait_time:.2f}s before next request")
                await asyncio.sleep(wait_time)

            self._last_request_time = time.time()

    async def _make_api_request_with_retry(
        self,
        client: httpx.AsyncClient,
        request: MistralOCRRequest
    ) -> httpx.Response:
        """Make API request with shared retry/backoff helper."""
        await self._enforce_rate_limit()
        return await request_with_retry(
            client,
            "POST",
            self.api_url,
            headers=self.headers,
            json=request.model_dump(),
            max_attempts=settings.MISTRAL_RETRY_ATTEMPTS,
            retry_statuses=settings.HTTP_RETRY_STATUSES,
            timeout=self.timeout,
        )

    async def process_document(
        self,
        pdf_path: Optional[str] = None,
        pdf_base64: Optional[str] = None,
        pdf_bytes: Optional[bytes] = None,

        has_query: bool = False,
        enable_validation: Optional[bool] = None,
        include_images: Optional[bool] = None,
        workflow_name: Optional[str] = None
    ) -> tuple[str, Optional[dict]]:
        """
        Process a PDF document and return markdown content with optional cross-validation.

        Performance optimized: Can accept pre-encoded base64 and/or bytes to avoid blocking I/O.

        Args:
            pdf_path: Path to the PDF file (required if pdf_base64 not provided)
            pdf_base64: Pre-encoded base64 string (optional, improves performance)
            pdf_bytes: Pre-read PDF bytes (optional, prevents file system race conditions during validation)
            has_query: Whether query filtering is active (affects validation sampling)
            enable_validation: Override global ENABLE_CROSS_VALIDATION setting (None=use global, True=force enable, False=force disable)
            workflow_name: Name of the workflow (e.g., "01_Fin_Reports") for workflow-specific validation

        Returns:
            Tuple of (markdown_content, validation_report_dict or None)

        Raises:
            ValueError: If neither pdf_path nor pdf_base64 provided, or if API returns an error
            FileNotFoundError: If PDF file doesn't exist
            httpx.HTTPError: If request fails
        """
        if pdf_path is None and pdf_base64 is None:
            raise ValueError("Either pdf_path or pdf_base64 must be provided")

        # Use provided base64 or encode from file
        if pdf_base64 is None:
            # Read and encode PDF from file
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            logger.info(f"Reading PDF: {pdf_path}")
            pdf_base64 = encode_pdf_to_base64(pdf_path)
        else:
            # Using pre-encoded base64 (performance optimized path)
            logger.debug(f"Using pre-encoded base64 ({len(pdf_base64)} chars)")

        # Create request - only include_image_base64 is supported
        should_include_images = include_images if include_images is not None else settings.INCLUDE_IMAGES
        request = MistralOCRRequest(
            model=self.model,
            document=DocumentInput(
                type="document_url",
                document_url=f"data:application/pdf;base64,{pdf_base64}"
            ),
            include_image_base64=should_include_images
        )

        logger.info(f"Sending request to Mistral API: {self.api_url}")

        # Send request using shared client for connection pooling
        client = self._get_client()
        should_close_client = self._client is None  # Close if temporary client

        try:
            # Make API request with rate limiting and retry logic
            response = await self._make_api_request_with_retry(client, request)

            # Handle response
            if response.status_code == 200:
                try:
                    ocr_response = MistralOCRResponse.model_validate(response.json())
                    logger.info(
                        f"Successfully processed document with model {ocr_response.model}, "
                        f"extracted {len(ocr_response.pages)} pages"
                    )

                    # Cross-validation if enabled
                    validation_report_dict = None
                    # Use enable_validation parameter if provided, otherwise use global setting
                    should_validate = enable_validation if enable_validation is not None else settings.ENABLE_CROSS_VALIDATION

                    if should_validate:
                        # Cross-validation requires pdf_bytes (in-memory bytes prevent file system issues)
                        validation_bytes = pdf_bytes
                        if validation_bytes is None and pdf_path:
                            # Fallback: read from file if bytes not provided
                            try:
                                with open(pdf_path, 'rb') as f:
                                    validation_bytes = f.read()
                            except FileNotFoundError:
                                logger.warning(f"PDF file not found for validation: {pdf_path}, skipping validation")
                                validation_bytes = None

                        if validation_bytes is None:
                            logger.warning("Cross-validation requires pdf_bytes or valid pdf_path, skipping validation")
                        else:
                            try:
                                validation_service = _get_validation_service()
                                validation_report = await validation_service.cross_validate_pages(
                                    ocr_response,
                                    validation_bytes,
                                    has_query=has_query,
                                    workflow_name=workflow_name
                                )

                                # Apply fixes for problem pages
                                for result in validation_report.validation_results:
                                    if result.has_problem_pattern and result.alternative_content:
                                        # Replace problematic page content with GPT-4o result
                                        logger.info(
                                            f"[Page {result.page_number}] Replacing problematic content "
                                            f"with GPT-4o extraction"
                                        )
                                        ocr_response.pages[result.page_number].markdown = result.alternative_content

                                    elif not result.passed:
                                        # Log warning but keep original
                                        logger.warning(
                                            f"[Page {result.page_number}] Failed validation "
                                            f"(similarity: {result.similarity_score:.2%}) - keeping original content"
                                        )

                                # Log summary
                                logger.info(
                                    f"Cross-validation summary: {validation_report.validated_pages}/"
                                    f"{validation_report.total_pages} pages checked, "
                                    f"{len(validation_report.problem_pages)} problems fixed, "
                                    f"{len(validation_report.failed_validations)} warnings"
                                )
                                logger.info(
                                    f"Validation metrics: {validation_report.total_time:.2f}s, "
                                    f"${validation_report.total_cost:.4f} estimated cost"
                                )

                                # Determine simple validation status
                                # All detailed metrics are logged above (lines 198-208)
                                has_problems = len(validation_report.problem_pages) > 0
                                has_warnings = len(validation_report.failed_validations) > 0

                                if has_problems:
                                    status = "problems_fixed"
                                elif has_warnings:
                                    status = "warnings"
                                else:
                                    status = "passed"

                                # Return simple status dict
                                validation_report_dict = {
                                    "enabled": "true",
                                    "status": status
                                }

                            except Exception as e:
                                logger.error(f"Cross-validation failed: {e}")
                                logger.warning("Continuing with original Mistral extraction")
                    
                    # Collect images if requested
                    if should_include_images:
                        all_images = []
                        for page in ocr_response.pages:
                            if page.images:
                                for img in page.images:
                                    # Add page index to image metadata
                                    img['page_index'] = page.index
                                    all_images.append(img)
                        
                        if validation_report_dict is None:
                            validation_report_dict = {}
                        
                        validation_report_dict["images"] = all_images
                        logger.info(f"Collected {len(all_images)} images from Mistral response")

                    return ocr_response.content, validation_report_dict

                except ValidationError as e:
                    logger.error(f"Failed to parse response: {e}")
                    raise ValueError(f"Invalid response format: {e}")

            else:
                # Handle error response
                try:
                    error_response = MistralErrorResponse.model_validate(response.json())
                    error_msg = (
                        f"Mistral API error ({response.status_code}): "
                        f"{error_response.message}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                except (ValidationError, ValueError):
                    # If we can't parse the error, return raw response
                    error_msg = (
                        f"Mistral API error ({response.status_code}): "
                        f"{response.text}"
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

        finally:
            # Close temporary client if not using persistent connection
            if should_close_client and client:
                await client.aclose()

    async def health_check(self) -> bool:
        """
        Check if the Mistral API is accessible using shared client.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Use shared client with shorter timeout for health checks
            client = httpx.AsyncClient(timeout=10.0) if self._client is None else self._client
            should_close = self._client is None

            try:
                response = await client.get(
                    self.api_url.replace('/ocr', '/health'),
                    headers=self.headers
                )
                return response.status_code == 200
            finally:
                if should_close:
                    await client.aclose()

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
