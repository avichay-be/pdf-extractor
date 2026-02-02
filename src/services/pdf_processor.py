"""
PDF processing service for splitting PDFs by outlines and combining results.
"""
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile

import pypdf
from pypdf import PdfReader, PdfWriter

from src.core.config import settings

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF splitting and markdown combination."""

    def __init__(self, max_pages_per_chunk: int = None):
        """
        Initialize PDF processor.

        Args:
            max_pages_per_chunk: Maximum pages per chunk (default from settings)
        """
        self.max_pages_per_chunk = max_pages_per_chunk or settings.MAX_PAGES_PER_CHUNK

    def split_with_outline_info(self, pdf_path: str) -> Tuple[List[str], Optional[List[Dict]]]:
        """
        Split PDF by main outlines and return outline metadata.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Tuple of (chunk_paths, outline_info)
            - chunk_paths: List of paths to temporary PDF chunk files
            - outline_info: List of outline metadata dicts with 'title', 'page', 'chunk_indices'
                           None if no outlines found
        """
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        logger.info(f"PDF has {total_pages} pages")

        # If PDF is within limit, return it as-is with no outline info
        if total_pages <= self.max_pages_per_chunk:
            logger.info("PDF within size limit, no splitting needed")
            return [pdf_path], None

        # Try to get main outlines (top-level only) - limit to max 4
        outlines = self._get_main_outlines(reader)

        if outlines:
            # Limit to first 4 outlines
            if len(outlines) > 4:
                logger.info(f"Found {len(outlines)} outlines, limiting to first 4")
                outlines = outlines[:4]
            else:
                logger.info(f"Found {len(outlines)} main outline sections")

            chunks, outline_metadata = self._split_by_outlines(reader, outlines, pdf_path, collect_metadata=True)
            return chunks, outline_metadata
        else:
            logger.info("No outlines found, splitting by page count")
            chunks = self._split_by_page_count(reader, pdf_path)
            return chunks, None

    def split_by_main_outlines(self, pdf_path: str) -> List[str]:
        """
        Split PDF by main outlines (top-level only).

        If the PDF has no outlines or sections are too large,
        falls back to splitting by page count.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of paths to temporary PDF chunk files
        """
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        logger.info(f"PDF has {total_pages} pages")

        # If PDF is within limit, return it as-is
        if total_pages <= self.max_pages_per_chunk:
            logger.info("PDF within size limit, no splitting needed")
            return [pdf_path]

        # Try to get main outlines (top-level only)
        outlines = self._get_main_outlines(reader)

        if outlines:
            logger.info(f"Found {len(outlines)} main outline sections")
            chunks = self._split_by_outlines(reader, outlines, pdf_path)
        else:
            logger.info("No outlines found, splitting by page count")
            chunks = self._split_by_page_count(reader, pdf_path)

        return chunks

    def _get_main_outlines(self, reader: PdfReader) -> List[dict]:
        """
        Extract main (top-level) outlines from PDF.

        Args:
            reader: PdfReader instance

        Returns:
            List of outline dictionaries with 'title' and 'page' keys
        """
        outlines = []

        try:
            outline_items = reader.outline
            if not outline_items:
                return []

            for item in outline_items:
                # Only process top-level outlines (not nested)
                if isinstance(item, list):
                    # Skip nested outlines
                    continue

                if hasattr(item, 'title') and hasattr(item, 'page'):
                    page_num = reader.get_destination_page_number(item)
                    outlines.append({
                        'title': item.title,
                        'page': page_num
                    })

            # Sort by page number
            outlines.sort(key=lambda x: x['page'])

        except Exception as e:
            logger.warning(f"Error extracting outlines: {e}")
            return []

        return outlines

    def _split_by_outlines(
        self,
        reader: PdfReader,
        outlines: List[dict],
        original_path: str,
        collect_metadata: bool = False
    ):
        """
        Split PDF based on outline sections, optionally collecting metadata.

        If a section exceeds max pages, it will be further split.

        Args:
            reader: PdfReader instance
            outlines: List of outline dictionaries (max 4 when collect_metadata=True)
            original_path: Path to original PDF
            collect_metadata: If True, return (chunks, metadata) tuple. If False, return chunks only.

        Returns:
            List[str] if collect_metadata=False
            Tuple[List[str], List[Dict]] if collect_metadata=True

        Code refactoring:
            This function unifies the previously duplicate _split_by_outlines and
            _split_by_outlines_with_metadata methods, eliminating 120 lines of duplication.
        """
        chunks = []
        outline_metadata = [] if collect_metadata else None
        total_pages = len(reader.pages)

        for i, outline in enumerate(outlines):
            start_page = outline['page']

            # Determine end page
            if i + 1 < len(outlines):
                end_page = outlines[i + 1]['page']
            else:
                end_page = total_pages

            section_pages = end_page - start_page
            chunk_start_idx = len(chunks) if collect_metadata else None

            # If section is within limit, create single chunk
            if section_pages <= self.max_pages_per_chunk:
                chunk_path = self._create_chunk(
                    reader,
                    start_page,
                    end_page,
                    f"section_{i}"
                )
                chunks.append(chunk_path)
            else:
                # Split large section into smaller chunks
                logger.info(
                    f"Section '{outline['title']}' has {section_pages} pages, "
                    f"splitting further"
                )
                sub_chunks = self._split_page_range(
                    reader,
                    start_page,
                    end_page,
                    f"section_{i}"
                )
                chunks.extend(sub_chunks)

            # Store outline metadata if requested
            if collect_metadata:
                chunk_end_idx = len(chunks)
                outline_metadata.append({
                    'title': outline['title'],
                    'page': start_page,
                    'chunk_indices': list(range(chunk_start_idx, chunk_end_idx))
                })

        # Return tuple or list depending on collect_metadata flag
        if collect_metadata:
            return chunks, outline_metadata
        return chunks

    def _split_by_page_count(self, reader: PdfReader, original_path: str) -> List[str]:
        """
        Split PDF by page count when no outlines are available.

        Args:
            reader: PdfReader instance
            original_path: Path to original PDF

        Returns:
            List of temporary PDF chunk file paths
        """
        total_pages = len(reader.pages)
        return self._split_page_range(reader, 0, total_pages, "chunk")

    def _split_page_range(
        self,
        reader: PdfReader,
        start_page: int,
        end_page: int,
        prefix: str
    ) -> List[str]:
        """
        Split a page range into chunks of max_pages_per_chunk.

        Args:
            reader: PdfReader instance
            start_page: Starting page index
            end_page: Ending page index (exclusive)
            prefix: Prefix for chunk filenames

        Returns:
            List of temporary PDF chunk file paths
        """
        chunks = []
        current_page = start_page

        chunk_idx = 0
        while current_page < end_page:
            chunk_end = min(current_page + self.max_pages_per_chunk, end_page)

            chunk_path = self._create_chunk(
                reader,
                current_page,
                chunk_end,
                f"{prefix}_{chunk_idx}"
            )
            chunks.append(chunk_path)

            current_page = chunk_end
            chunk_idx += 1

        return chunks

    def _create_chunk(
        self,
        reader: PdfReader,
        start_page: int,
        end_page: int,
        name: str
    ) -> str:
        """
        Create a PDF chunk from a page range.

        Args:
            reader: PdfReader instance
            start_page: Starting page index (inclusive)
            end_page: Ending page index (exclusive)
            name: Name for the chunk file

        Returns:
            Path to the created chunk file
        """
        writer = PdfWriter()

        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])

        # Create temporary file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f'_{name}.pdf',
            prefix='pdf_chunk_'
        ) as tmp_file:
            writer.write(tmp_file)
            chunk_path = tmp_file.name

        logger.info(f"Created chunk: pages {start_page}-{end_page-1} -> {chunk_path}")

        return chunk_path

    async def cleanup_chunks(self, chunk_paths: List[str], original_path: str = None):
        """
        Delete temporary chunk files in parallel for better performance.

        Performance optimized: Uses asyncio.to_thread() for concurrent file deletion.

        Args:
            chunk_paths: List of chunk file paths to delete
            original_path: Optional original PDF path to preserve
        """
        import asyncio

        async def delete_file(chunk_path: str):
            """Helper to delete a single file asynchronously."""
            # Don't delete the original file if it was returned as-is
            if original_path and chunk_path == original_path:
                return

            try:
                await asyncio.to_thread(Path(chunk_path).unlink, missing_ok=True)
                logger.debug(f"Deleted chunk: {chunk_path}")
            except Exception as e:
                logger.warning(f"Failed to delete chunk {chunk_path}: {e}")

        # Delete all chunks in parallel
        if chunk_paths:
            await asyncio.gather(*[delete_file(path) for path in chunk_paths])

    def combine_markdown_results(self, markdown_chunks: List[str]) -> str:
        """
        Combine multiple markdown results into a single document.

        Args:
            markdown_chunks: List of markdown strings

        Returns:
            Combined markdown content
        """
        if not markdown_chunks:
            return ""

        if len(markdown_chunks) == 1:
            return markdown_chunks[0]

        # Combine chunks with separators
        combined = []
        for i, chunk in enumerate(markdown_chunks):
            if i > 0:
                # Add a visual separator between chunks
                combined.append("\n\n---\n\n")

            combined.append(chunk.strip())

        result = "".join(combined)
        logger.info(f"Combined {len(markdown_chunks)} markdown chunks")

        return result
