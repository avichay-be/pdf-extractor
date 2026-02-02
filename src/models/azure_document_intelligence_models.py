"""
Pydantic models for Azure Document Intelligence API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class BoundingRegion(BaseModel):
    """Bounding region for a document element."""
    page_number: int = Field(..., alias="pageNumber")
    polygon: List[float]


class Span(BaseModel):
    """Span information for text content."""
    offset: int
    length: int


class TableCell(BaseModel):
    """Table cell information."""
    kind: Optional[str] = None
    row_index: int = Field(..., alias="rowIndex")
    column_index: int = Field(..., alias="columnIndex")
    row_span: Optional[int] = Field(default=1, alias="rowSpan")
    column_span: Optional[int] = Field(default=1, alias="columnSpan")
    content: str
    bounding_regions: Optional[List[BoundingRegion]] = Field(default=None, alias="boundingRegions")
    spans: List[Span]

    model_config = {
        "populate_by_name": True
    }


class Table(BaseModel):
    """Table information from document."""
    row_count: int = Field(..., alias="rowCount")
    column_count: int = Field(..., alias="columnCount")
    cells: List[TableCell]
    bounding_regions: Optional[List[BoundingRegion]] = Field(default=None, alias="boundingRegions")
    spans: List[Span]
    caption: Optional[Dict[str, Any]] = None

    model_config = {
        "populate_by_name": True
    }

    def get_headers(self) -> List[str]:
        """
        Extract table headers from the first row or from cells marked as 'columnHeader'.

        Returns:
            List of header strings. Empty list if no headers found.
        """
        headers = []

        # First, try to find cells marked as columnHeader
        header_cells = [cell for cell in self.cells if cell.kind == "columnHeader"]

        if header_cells:
            # Sort by column index
            header_cells.sort(key=lambda c: c.column_index)
            headers = [cell.content.strip() for cell in header_cells]
        else:
            # Fallback: use first row as headers
            first_row_cells = [cell for cell in self.cells if cell.row_index == 0]
            first_row_cells.sort(key=lambda c: c.column_index)
            headers = [cell.content.strip() for cell in first_row_cells]

        return headers

    def has_headers(self) -> bool:
        """Check if table has header cells."""
        return any(cell.kind == "columnHeader" for cell in self.cells)

    def get_data_rows(self) -> List[List[str]]:
        """
        Extract data rows (excluding headers).

        Returns:
            List of rows, where each row is a list of cell contents.
        """
        rows = []

        # Determine starting row (skip headers if they exist)
        start_row = 1 if self.has_headers() else 0

        for row_idx in range(start_row, self.row_count):
            row_cells = [cell for cell in self.cells if cell.row_index == row_idx]
            row_cells.sort(key=lambda c: c.column_index)
            row_data = [cell.content.strip() for cell in row_cells]
            rows.append(row_data)

        return rows


class DocumentPage(BaseModel):
    """Document page information."""
    page_number: int = Field(..., alias="pageNumber")
    angle: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    unit: Optional[str] = None
    words: Optional[List[Dict[str, Any]]] = None
    lines: Optional[List[Dict[str, Any]]] = None
    spans: Optional[List[Span]] = None

    model_config = {
        "populate_by_name": True
    }


class AnalyzeResult(BaseModel):
    """Result from Azure Document Intelligence analyze operation."""
    api_version: str = Field(..., alias="apiVersion")
    model_id: str = Field(..., alias="modelId")
    content: str
    pages: List[DocumentPage]
    tables: Optional[List[Table]] = None
    styles: Optional[List[Dict[str, Any]]] = None
    content_format: Optional[str] = Field(default=None, alias="contentFormat")

    model_config = {
        "populate_by_name": True
    }


class DocumentIntelligenceResponse(BaseModel):
    """Complete response from Document Intelligence API."""
    status: str
    created_date_time: str = Field(..., alias="createdDateTime")
    last_updated_date_time: str = Field(..., alias="lastUpdatedDateTime")
    analyze_result: Optional[AnalyzeResult] = Field(default=None, alias="analyzeResult")

    model_config = {
        "populate_by_name": True
    }
