"""
Azure Document Intelligence client for extracting tables from PDFs.
"""
import logging
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncio
from io import BytesIO

import httpx
from pydantic import ValidationError

from src.models.azure_document_intelligence_models import (
    Table,
    DocumentIntelligenceResponse,
    AnalyzeResult
)
from src.core.config import settings

logger = logging.getLogger(__name__)


class MergedTable:
    """Represents a merged table from multiple pages."""

    def __init__(self, headers: List[str], page_number: int):
        """
        Initialize a merged table.

        Args:
            headers: Table headers
            page_number: Starting page number
        """
        self.headers = headers
        self.start_page = page_number
        self.end_page = page_number
        self.data_rows: List[List[str]] = []

    def add_rows(self, rows: List[List[str]], page_number: int):
        """Add data rows to the merged table."""
        self.data_rows.extend(rows)
        self.end_page = page_number

    def to_markdown(self) -> str:
        """
        Convert merged table to markdown format.

        Handles varying column counts by determining the maximum column count
        and padding/trimming rows to match.

        Returns:
            Markdown string representation of the table
        """
        if not self.headers and not self.data_rows:
            return ""

        lines = []

        # Determine maximum column count across all rows
        max_cols = len(self.headers) if self.headers else 0
        for row in self.data_rows:
            max_cols = max(max_cols, len(row))

        # Adjust headers to match max column count
        adjusted_headers = self.headers[:] if self.headers else []
        while len(adjusted_headers) < max_cols:
            adjusted_headers.append(f"Col{len(adjusted_headers)+1}")

        # Add page range info
        if self.start_page == self.end_page:
            lines.append(f"**Table from Page {self.start_page}**\n")
        else:
            lines.append(f"**Table from Pages {self.start_page}-{self.end_page}**\n")

        # Add headers
        header_row = "| " + " | ".join(adjusted_headers) + " |"
        lines.append(header_row)

        # Separator row
        separator = "| " + " | ".join(["---"] * len(adjusted_headers)) + " |"
        lines.append(separator)

        # Add data rows
        for row in self.data_rows:
            # Create a copy to avoid modifying original data
            formatted_row = row[:]

            # Pad short rows
            while len(formatted_row) < max_cols:
                formatted_row.append("")

            # Trim long rows (shouldn't happen with max_cols calculation, but safe)
            formatted_row = formatted_row[:max_cols]

            row_str = "| " + " | ".join(formatted_row) + " |"
            lines.append(row_str)

        return "\n".join(lines)


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
            pdf_base64 = self._encode_pdf_to_base64(pdf_path)
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
        tables_by_page = self._group_tables_by_page(analyze_result.tables)

        # Merge or return individual tables
        if merge_tables:
            merged_tables = self._merge_tables_across_pages(tables_by_page)
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
                    markdown_tables.append(self._table_to_markdown(table, page_num))
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

    def _encode_pdf_to_base64(self, pdf_path: str) -> str:
        """
        Encode PDF file to base64 string.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Base64 encoded string
        """
        with open(pdf_path, 'rb') as pdf_file:
            pdf_bytes = pdf_file.read()
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

        logger.debug(f"Encoded PDF to base64 ({len(pdf_base64)} chars)")
        return pdf_base64

    def _group_tables_by_page(self, tables: List[Table]) -> Dict[int, List[Table]]:
        """
        Group tables by page number.

        Args:
            tables: List of Table objects

        Returns:
            Dictionary mapping page number to list of tables
        """
        tables_by_page = {}

        for table in tables:
            if not table.bounding_regions:
                # If no bounding region, skip this table
                logger.warning("Table has no bounding regions, skipping")
                continue

            # Use the first bounding region's page number
            page_number = table.bounding_regions[0].page_number

            if page_number not in tables_by_page:
                tables_by_page[page_number] = []

            tables_by_page[page_number].append(table)

        return tables_by_page

    def _merge_tables_across_pages(
        self,
        tables_by_page: Dict[int, List[Table]]
    ) -> List[MergedTable]:
        """
        Merge tables with same headers across consecutive pages.

        Logic:
        1. If a table on page N+1 has the same headers as a table on page N, merge them
        2. If a table on page N+1 has NO headers but follows a table on page N, merge them
        3. Otherwise, start a new merged table

        Args:
            tables_by_page: Dictionary mapping page number to list of tables

        Returns:
            List of MergedTable objects
        """
        merged_tables: List[MergedTable] = []
        previous_table: Optional[MergedTable] = None

        # Process pages in order
        for page_num in sorted(tables_by_page.keys()):
            for table in tables_by_page[page_num]:
                current_headers = table.get_headers()
                current_data = table.get_data_rows()

                # Case 1: No previous table, start a new merged table
                if previous_table is None:
                    previous_table = MergedTable(current_headers, page_num)
                    previous_table.add_rows(current_data, page_num)
                    continue

                # Case 2: Current table has same headers as previous
                if current_headers and self._headers_match(previous_table.headers, current_headers):
                    logger.info(
                        f"Merging table on page {page_num} with previous table "
                        f"(same headers: {current_headers})"
                    )
                    previous_table.add_rows(current_data, page_num)
                    continue

                # Case 3: Current table has NO headers, merge with previous
                if not table.has_headers() and previous_table is not None:
                    logger.info(
                        f"Merging table on page {page_num} with previous table "
                        f"(no headers, continuation)"
                    )
                    # Add all rows including what would be the "header" row
                    # since it's actually data
                    all_rows = [current_headers] + current_data if current_headers else current_data
                    previous_table.add_rows(all_rows, page_num)
                    continue

                # Case 4: Try numerical validation (for OCR errors causing column mismatches)
                if (previous_table is not None and previous_table.data_rows and current_data and
                    settings.AZURE_DI_USE_NUMERICAL_VALIDATION):
                    # Check if the last row of previous table and first row of current table
                    # are numerically continuous (balance continuity)
                    prev_last_row = previous_table.data_rows[-1]
                    curr_first_row = current_data[0] if current_data else []

                    if curr_first_row and self._validate_numerical_continuity(
                        prev_last_row,
                        curr_first_row,
                        tolerance=settings.AZURE_DI_BALANCE_TOLERANCE
                    ):
                        logger.info(
                            f"Merging table on page {page_num} with previous table "
                            f"(numerical continuity validated despite structure mismatch)"
                        )
                        previous_table.add_rows(current_data, page_num)
                        continue

                # Case 5: Different headers and no numerical continuity, finalize previous and start new
                merged_tables.append(previous_table)
                previous_table = MergedTable(current_headers, page_num)
                previous_table.add_rows(current_data, page_num)

        # Don't forget the last table
        if previous_table is not None:
            merged_tables.append(previous_table)

        logger.info(f"Merged {sum(len(tables) for tables in tables_by_page.values())} tables into {len(merged_tables)} merged table(s)")
        return merged_tables

    def _headers_match(self, headers1: List[str], headers2: List[str]) -> bool:
        """
        Check if two header lists match (case-insensitive, whitespace-normalized).

        Args:
            headers1: First header list
            headers2: Second header list

        Returns:
            True if headers match, False otherwise
        """
        if len(headers1) != len(headers2):
            return False

        # Normalize and compare
        normalized1 = [h.strip().lower() for h in headers1]
        normalized2 = [h.strip().lower() for h in headers2]

        return normalized1 == normalized2

    def _extract_numeric_columns(self, row: List[str]) -> Dict[str, Any]:
        """
        Extract numeric values from a row and identify balance/debit/credit columns.

        Args:
            row: List of cell values as strings

        Returns:
            Dict with keys:
            - 'amounts': List of all numeric values found
            - 'balance': Last numeric value (usually the balance column)
            - 'positions': List of (index, value) tuples for all numbers
        """
        import re

        amounts = []
        positions = []

        for idx, cell in enumerate(row):
            if not cell:
                continue

            # Clean the cell value
            cell_clean = str(cell).strip()

            # Match numbers (including decimals, commas, and negatives)
            # Supports formats: 1,234.56 or 1234.56 or -1234.56
            number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
            matches = re.findall(number_pattern, cell_clean)

            for match in matches:
                try:
                    # Remove commas and convert to float
                    value = float(match.replace(',', ''))
                    amounts.append(value)
                    positions.append((idx, value))
                except ValueError:
                    continue

        result = {
            'amounts': amounts,
            'positions': positions,
            'balance': amounts[-1] if amounts else None,  # Last number is usually balance
            'has_numbers': len(amounts) > 0
        }

        return result

    def _validate_numerical_continuity(
        self,
        previous_row: List[str],
        current_row: List[str],
        tolerance: float = 0.01
    ) -> bool:
        """
        Check if two rows are numerically continuous (validate running balance).

        For bank statements, checks if the balance continues correctly between rows.

        Args:
            previous_row: Last row from previous table
            current_row: First row from current table
            tolerance: Acceptable difference for floating point comparison

        Returns:
            True if rows appear to be continuous, False otherwise
        """
        prev_nums = self._extract_numeric_columns(previous_row)
        curr_nums = self._extract_numeric_columns(current_row)

        # Need at least some numbers in both rows
        if not prev_nums['has_numbers'] or not curr_nums['has_numbers']:
            logger.info("Numerical continuity: No numbers found in rows")
            return False

        # Primary check: Balance continuity
        prev_balance = prev_nums['balance']
        curr_balance = curr_nums['balance']

        if prev_balance is not None and curr_balance is not None:
            # Simple continuity check: current balance should be "close" to previous
            # In a continuous statement, balance changes by transactions
            # We check if the change is reasonable (not a huge jump)

            balance_diff = abs(curr_balance - prev_balance)

            # If balance is exactly the same, definitely continuous
            if balance_diff <= tolerance:
                logger.info(f"Numerical continuity: Same balance ({curr_balance:.2f})")
                return True

            # If balance changed, check if it's a reasonable transaction amount
            # Heuristic: If change is less than 50% of previous balance, likely continuous
            if prev_balance != 0:
                percent_change = balance_diff / abs(prev_balance)
                if percent_change < 0.5:  # Less than 50% change
                    logger.info(
                        f"Numerical continuity: Balance change is reasonable "
                        f"({prev_balance:.2f} → {curr_balance:.2f}, {percent_change*100:.1f}% change)"
                    )
                    return True
                else:
                    logger.info(
                        f"Numerical continuity: Balance change too large "
                        f"({prev_balance:.2f} → {curr_balance:.2f}, {percent_change*100:.1f}% change)"
                    )
                    return False

            # If previous balance was 0, accept any reasonable current balance
            if abs(curr_balance) < 1000000:  # Sanity check: balance < 1M
                logger.info(f"Numerical continuity: Starting from zero balance")
                return True

        # Fallback: Check if any numbers match positions (column alignment)
        prev_positions = set(idx for idx, _ in prev_nums['positions'])
        curr_positions = set(idx for idx, _ in curr_nums['positions'])

        if prev_positions and curr_positions:
            overlap = len(prev_positions & curr_positions)
            total = max(len(prev_positions), len(curr_positions))
            if overlap / total >= 0.5:  # At least 50% of columns have numbers in same positions
                logger.info(f"Numerical continuity: Column positions match ({overlap}/{total})")
                return True

        logger.info("Numerical continuity: Validation check failed (no match criteria met)")
        return False

    def _table_to_markdown(self, table: Table, page_number: int) -> str:
        """
        Convert a single table to markdown format.

        Args:
            table: Table object
            page_number: Page number for context

        Returns:
            Markdown string representation
        """
        lines = [f"**Table from Page {page_number}**\n"]

        headers = table.get_headers()
        data_rows = table.get_data_rows()

        if headers:
            # Header row
            header_row = "| " + " | ".join(headers) + " |"
            lines.append(header_row)

            # Separator row
            separator = "| " + " | ".join(["---"] * len(headers)) + " |"
            lines.append(separator)

        # Data rows
        for row in data_rows:
            if headers:
                # Pad or trim row to match header count
                while len(row) < len(headers):
                    row.append("")
                row = row[:len(headers)]

            row_str = "| " + " | ".join(row) + " |"
            lines.append(row_str)

        return "\n".join(lines)

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
