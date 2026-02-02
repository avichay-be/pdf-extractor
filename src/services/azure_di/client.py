"""
Azure Document Intelligence client for extracting tables from PDFs.

Main client for API communication with Azure Document Intelligence service.
"""
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio

import httpx
from pydantic import ValidationError

from src.models.azure_document_intelligence_models import (
    DocumentIntelligenceResponse,
    AnalyzeResult
)
from src.core.config import settings
from src.core.utils import encode_pdf_to_base64
from .table_merger import TableMerger, MergedTable

logger = logging.getLogger(__name__)


class AzureDocumentIntelligenceClient:
    """Client for interacting with Azure Document Intelligence API."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0
    ):
        """
        Initialize Azure Document Intelligence client.

        Args:
            endpoint: Azure Document Intelligence endpoint URL
            api_key: Azure API key
            model: Model to use (default: prebuilt-layout)
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint or settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
        self.api_key = api_key or settings.AZURE_DOCUMENT_INTELLIGENCE_KEY
        self.model = model or settings.AZURE_DOCUMENT_INTELLIGENCE_MODEL
        self.timeout = timeout

        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure Document Intelligence endpoint and key must be provided "
                "either as parameters or in environment variables"
            )

        self.headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.api_key
        }

        # Shared HTTP client with connection pooling
        self._client: Optional[httpx.AsyncClient] = None

        # Initialize table merger (which includes validator)
        self.table_merger = TableMerger()

    async def __aenter__(self):
        """Async context manager entry - initialize shared client."""
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
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
            return httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def extract_tables(
        self,
        pdf_path: Optional[str] = None,
        pdf_base64: Optional[str] = None,
        merge_tables: bool = True
    ) -> tuple[List[str], Optional[Dict[str, Any]]]:
        """
        Extract tables from a PDF document using Azure Document Intelligence.

        Args:
            pdf_path: Path to the PDF file
            pdf_base64: Pre-encoded base64 string (optional)
            merge_tables: Whether to merge tables with same headers across pages

        Returns:
            Tuple of (list of markdown tables, metadata dict)

        Raises:
            ValueError: If neither pdf_path nor pdf_base64 provided
            FileNotFoundError: If PDF file doesn't exist
            httpx.HTTPError: If request fails
        """
        if pdf_path is None and pdf_base64 is None:
            raise ValueError("Either pdf_path or pdf_base64 must be provided")

        # Use provided base64 or encode from file
        if pdf_base64 is None:
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            logger.info(f"Reading PDF for table extraction: {pdf_path}")
            pdf_base64 = encode_pdf_to_base64(pdf_path)
        else:
            logger.debug(f"Using pre-encoded base64 ({len(pdf_base64)} chars)")

        # Start analyze operation
        logger.info(f"Starting document analysis with model: {self.model}")
        operation_location = await self._start_analyze(pdf_base64)

        # Poll for results
        logger.info("Polling for analysis results...")
        analyze_result = await self._poll_analyze_result(operation_location)

        # Extract tables
        if not analyze_result.tables:
            logger.info("No tables found in document")
            return [], {"table_count": 0, "merged": False}

        logger.info(f"Found {len(analyze_result.tables)} tables in document")

        # Group tables by page
        tables_by_page = self.table_merger.group_tables_by_page(analyze_result.tables)

        # Merge or return individual tables
        if merge_tables:
            merged_tables = self.table_merger.merge_tables_across_pages(tables_by_page)
            markdown_tables = [table.to_markdown() for table in merged_tables]
            metadata = {
                "table_count": len(analyze_result.tables),
                "merged_table_count": len(merged_tables),
                "merged": True
            }
        else:
            markdown_tables = []
            for page_num in sorted(tables_by_page.keys()):
                for table in tables_by_page[page_num]:
                    markdown_tables.append(self.table_merger.table_to_markdown(table, page_num))
            metadata = {
                "table_count": len(analyze_result.tables),
                "merged": False
            }

        logger.info(f"Extracted {len(markdown_tables)} table(s) as markdown")
        return markdown_tables, metadata

    async def _start_analyze(self, pdf_base64: str) -> str:
        """
        Start document analysis operation.

        Args:
            pdf_base64: Base64 encoded PDF content

        Returns:
            Operation location URL for polling

        Raises:
            ValueError: If API returns error
        """
        analyze_url = f"{self.endpoint}/documentintelligence/documentModels/{self.model}:analyze?api-version=2024-11-30"

        request_body = {
            "base64Source": pdf_base64
        }

        client = self._get_client()
        should_close_client = self._client is None

        try:
            response = await client.post(
                analyze_url,
                headers=self.headers,
                json=request_body
            )

            if response.status_code == 202:
                # Success - get operation location from header
                operation_location = response.headers.get("Operation-Location")
                if not operation_location:
                    raise ValueError("No Operation-Location header in response")

                logger.info(f"Analysis started, operation ID: {operation_location.split('/')[-1][:8]}...")
                return operation_location

            else:
                error_msg = f"Failed to start analysis ({response.status_code}): {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)

        finally:
            if should_close_client and client:
                await client.aclose()

    async def _poll_analyze_result(
        self,
        operation_location: str,
        max_retries: int = 60,
        poll_interval: float = 2.0
    ) -> AnalyzeResult:
        """
        Poll for analysis results until completion.

        Args:
            operation_location: Operation location URL
            max_retries: Maximum number of polling attempts
            poll_interval: Seconds between polling attempts

        Returns:
            AnalyzeResult object

        Raises:
            ValueError: If operation fails or times out
        """
        client = self._get_client()
        should_close_client = self._client is None

        try:
            for attempt in range(max_retries):
                response = await client.get(
                    operation_location,
                    headers={"Ocp-Apim-Subscription-Key": self.api_key}
                )

                if response.status_code != 200:
                    raise ValueError(f"Polling failed ({response.status_code}): {response.text}")

                result = response.json()
                status = result.get("status")

                if status == "succeeded":
                    logger.info(f"Analysis completed after {attempt + 1} polls")
                    # Parse response
                    doc_response = DocumentIntelligenceResponse.model_validate(result)
                    return doc_response.analyze_result

                elif status == "failed":
                    error = result.get("error", {})
                    raise ValueError(f"Analysis failed: {error}")

                elif status in ["running", "notStarted"]:
                    logger.debug(f"Analysis in progress... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(poll_interval)

                else:
                    raise ValueError(f"Unknown status: {status}")

            raise ValueError(f"Analysis timed out after {max_retries * poll_interval} seconds")

        finally:
            if should_close_client and client:
                await client.aclose()

    async def health_check(self) -> bool:
        """
        Check if the Azure Document Intelligence API is accessible.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple health check - try to access the endpoint
            client = httpx.AsyncClient(timeout=10.0) if self._client is None else self._client
            should_close = self._client is None

            try:
                # Document Intelligence doesn't have a dedicated health endpoint
                # We'll just verify the endpoint is reachable
                info_url = f"{self.endpoint}/documentintelligence/info?api-version=2024-11-30"
                response = await client.get(
                    info_url,
                    headers={"Ocp-Apim-Subscription-Key": self.api_key}
                )
                return response.status_code in [200, 404]  # 404 is ok, means endpoint is reachable
            finally:
                if should_close:
                    await client.aclose()

        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
