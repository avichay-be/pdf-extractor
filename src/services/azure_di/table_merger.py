"""
Table merger for combining tables across pages.

Provides logic for merging tables with matching headers across consecutive
pages and converting them to markdown format.
"""
import logging
from typing import List, Dict, Optional

from src.models.azure_document_intelligence_models import Table
from .table_validator import TableValidator
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


class TableMerger:
    """Handles table merging logic across pages."""

    def __init__(self):
        """Initialize table merger with validator."""
        self.validator = TableValidator()

    def merge_tables_across_pages(
        self,
        tables_by_page: Dict[int, List[Table]]
    ) -> List[MergedTable]:
        """
        Merge tables with same headers across consecutive pages.

        Logic:
        1. If a table on page N+1 has the same headers as a table on page N, merge them
        2. If a table on page N+1 has NO headers but follows a table on page N, merge them
        3. With numerical validation enabled, merge if balance is continuous
        4. Otherwise, start a new merged table

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

                    if curr_first_row and self.validator.validate_numerical_continuity(
                        prev_last_row,
                        curr_first_row
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

    def group_tables_by_page(self, tables: List[Table]) -> Dict[int, List[Table]]:
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

    def table_to_markdown(self, table: Table, page_number: int) -> str:
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
