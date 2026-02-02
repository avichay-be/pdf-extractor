"""
Default workflow handler (Mistral with validation).

Uses Mistral Document AI for extraction with optional cross-validation.
Supports outline-based splitting, query filtering, and parallel processing.
"""
import logging
import time
import asyncio
from typing import Optional, List
from pathlib import Path

from .base_handler import BaseWorkflowHandler
from src.models.workflow_models import WorkflowResult, ExtractedSection
from src.services.client_factory import get_client_factory
from src.services.extraction_service import process_with_model
from src.core.utils import filter_outlines_by_query, encode_chunks_to_base64_async
from src.core.error_handling import WorkflowExecutionError

logger = logging.getLogger(__name__)

# Get clients from factory
client_factory = get_client_factory()
pdf_processor = client_factory.pdf_processor


class DefaultHandler(BaseWorkflowHandler):
    """Handler for default Mistral workflow with validation support."""

    async def execute(
        self,
        pdf_path: str,
        query: str,
        enable_validation: Optional[bool] = None
    ) -> WorkflowResult:
        """Execute default Mistral workflow.

        Supports:
        - Outline-based splitting
        - Query filtering
        - Parallel chunk processing
        - Cross-validation

        Args:
            pdf_path: Path to the PDF file
            query: Query string for filtering sections
            enable_validation: Whether to enable cross-validation

        Returns:
            WorkflowResult with extracted content and optional validation

        Raises:
            WorkflowExecutionError: If extraction fails
        """
        start_time = time.time()
        self._log_execution_start("Mistral", pdf_path, query)

        pdf_chunks = []
        try:
            # 1. Split PDF by outlines with metadata
            pdf_chunks, outline_info = pdf_processor.split_with_outline_info(pdf_path)
            logger.info(f"Split PDF into {len(pdf_chunks)} chunks")

            # 2. Filter outlines by query
            if outline_info:
                filtered_outline_info = filter_outlines_by_query(outline_info, query)
                logger.info(
                    f"Filtered to {len(filtered_outline_info)} outlines matching '{query}'"
                )
            else:
                filtered_outline_info = None

            # 3. Pre-encode all chunks to base64 in parallel (3x speedup)
            logger.info(f"Pre-encoding {len(pdf_chunks)} chunks to base64 in parallel...")
            encoded_chunks = await encode_chunks_to_base64_async(pdf_chunks)

            # 4. Process all chunks in parallel with asyncio.gather
            logger.info(f"Processing {len(pdf_chunks)} chunks in parallel...")
            has_query = bool(query and query.strip())

            # Determine workflow name for validation (e.g., "01_Fin_Reports")
            workflow_name = query if query and query.strip() else None

            tasks = [
                process_with_model(
                    model_name="mistral",
                    chunk_path=chunk_path,
                    chunk_base64=chunk_base64,
                    chunk_bytes=None,  # Memory optimization: read on-demand when validation needed
                    has_query=has_query,
                    enable_validation=enable_validation,
                    workflow_name=workflow_name
                )
                for chunk_path, chunk_base64 in encoded_chunks
            ]
            results = await asyncio.gather(*tasks)

            # 5. Unpack results (content, validation_report)
            markdown_results = [content for content, _ in results]
            validation_reports = [report for _, report in results]

            logger.info(f"Completed parallel processing of {len(pdf_chunks)} chunks")

            # 6. Build sections if we have filtered outlines
            sections = []
            if filtered_outline_info:
                sections = self._build_sections(
                    filtered_outline_info,
                    markdown_results,
                    Path(pdf_path).stem
                )

            # 7. Combine all results
            combined_markdown = pdf_processor.combine_markdown_results(markdown_results)

            # 8. Aggregate validation reports
            aggregated_validation = self._aggregate_validation_reports(validation_reports)

            execution_time = time.time() - start_time

            # 9. Build result
            result = WorkflowResult(
                content=combined_markdown,
                metadata={
                    "workflow": "mistral",
                    "extraction_method": "mistral_document_ai",
                    "model": "mistral-document-ai-2505",
                    "total_chunks": len(pdf_chunks),
                    "has_outlines": outline_info is not None,
                    "filtered_sections": len(filtered_outline_info) if filtered_outline_info else 0,
                    "execution_time": execution_time,
                    "query": query
                },
                sections=sections if sections else None,
                validation_report=aggregated_validation
            )

            self._log_execution_complete("Mistral", result, execution_time)
            return result

        except Exception as e:
            logger.error(f"Default Mistral workflow failed: {e}")
            raise WorkflowExecutionError(f"Mistral extraction failed: {str(e)}")

        finally:
            # Cleanup temporary files
            if pdf_chunks:
                await pdf_processor.cleanup_chunks(pdf_chunks, pdf_path)

    def _build_sections(
        self,
        filtered_outline_info: List[dict],
        markdown_results: List[str],
        base_filename: str
    ) -> List[ExtractedSection]:
        """Build ExtractedSection objects from filtered outlines.

        Args:
            filtered_outline_info: List of filtered outline dictionaries
            markdown_results: List of markdown content for each chunk
            base_filename: Base name for section filenames

        Returns:
            List of ExtractedSection objects
        """
        sections = []
        for outline in filtered_outline_info:
            # Combine markdown for all chunks in this outline section
            section_markdown = []
            for chunk_idx in outline['chunk_indices']:
                section_markdown.append(markdown_results[chunk_idx])

            combined = pdf_processor.combine_markdown_results(section_markdown)

            # Create safe filename from outline title
            safe_title = "".join(
                c if c.isalnum() or c in (' ', '-', '_') else '_'
                for c in outline['title']
            )
            safe_title = safe_title.strip().replace(' ', '_')[:50]  # Limit length

            section_filename = f"{safe_title}_{base_filename}.md"

            # Determine page range (first and last page in chunk_indices)
            start_page = outline.get('page', 1)
            # Estimate end page (last chunk's last page)
            end_page = start_page  # Default to same page if no info

            section = ExtractedSection(
                filename=section_filename,
                content=combined,
                title=outline['title'],
                page_range=(start_page, end_page)
            )
            sections.append(section)

        return sections

    def _aggregate_validation_reports(
        self,
        validation_reports: List[Optional[dict]]
    ) -> Optional[dict]:
        """Aggregate validation reports from multiple chunks.

        Args:
            validation_reports: List of validation report dictionaries (or None)

        Returns:
            Aggregated validation report, or None if no validation was performed
        """
        # Filter out None reports
        valid_reports = [r for r in validation_reports if r is not None]

        if not valid_reports:
            return None

        # Count statuses
        status_counts = {}
        for report in valid_reports:
            status = report.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        # Determine overall status
        if "problems_fixed" in status_counts:
            overall_status = "problems_fixed"
        elif "warnings" in status_counts:
            overall_status = "warnings"
        elif "passed" in status_counts:
            overall_status = "passed"
        else:
            overall_status = "unknown"

        return {
            "enabled": True,
            "status": overall_status,
            "chunks_validated": len(valid_reports),
            "status_breakdown": status_counts
        }
