"""
Workflow models for PDF extraction operations.

This module provides data structures for workflow execution results and extracted sections.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Any


@dataclass
class ExtractedSection:
    """Represents a single extracted section from a PDF."""

    filename: str
    """Name of the section file."""

    content: str
    """Markdown content of this section."""

    title: str
    """Title of the section (from outline or generated)."""

    page_range: tuple[int, int]
    """Start and end page numbers (inclusive)."""

    def __post_init__(self):
        """Validate page range."""
        if self.page_range[0] > self.page_range[1]:
            raise ValueError(f"Invalid page range: {self.page_range}")


@dataclass
class WorkflowResult:
    """Result from workflow execution.

    Represents the output of a workflow handler after processing a PDF document.
    """

    content: str
    """Combined markdown content or list of sections."""

    metadata: dict[str, Any]
    """Workflow execution metadata (model, timing, pages, etc.)."""

    sections: Optional[List[ExtractedSection]] = None
    """Individual extracted sections (if document was split by outlines)."""

    validation_report: Optional[dict[str, Any]] = None
    """Cross-validation report (if validation was enabled)."""

    def __post_init__(self):
        """Initialize default values."""
        if self.sections is None:
            self.sections = []
        if self.metadata is None:
            self.metadata = {}

    @property
    def has_sections(self) -> bool:
        """Check if result contains multiple sections."""
        return self.sections is not None and len(self.sections) > 0

    @property
    def section_count(self) -> int:
        """Get number of extracted sections."""
        return len(self.sections) if self.sections else 0

    @property
    def was_validated(self) -> bool:
        """Check if validation was performed."""
        return self.validation_report is not None
