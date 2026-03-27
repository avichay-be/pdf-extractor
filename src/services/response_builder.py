"""
Response builder for PDF extraction endpoints.

Handles formatting responses for both multipart/form-data and JSON endpoints,
including single file downloads and ZIP file creation for multiple sections.
"""
import json
import logging
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timezone
import zipfile

from fastapi import Response
from fastapi.responses import StreamingResponse

from src.models.api_models import (
    ExtractedContent,
    ExtractionResponseMetadata,
    OutlineExtractionResponse,
    ValidationSummary,
)
from src.models.workflow_models import WorkflowResult

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """Builds responses for extraction endpoints."""

    def build_download_response(
        self,
        result: WorkflowResult,
        original_filename: str,
        workflow_suffix: str = "",
        request_id: str = "",
    ) -> Response | StreamingResponse:
        """Build download response for /extract endpoint.

        Returns either a single markdown file or a ZIP file with multiple sections.

        Args:
            result: WorkflowResult from extraction
            original_filename: Original PDF filename
            workflow_suffix: Optional suffix for filename (for example "_ocr_all_around")

        Returns:
            FastAPI Response (single file) or StreamingResponse (ZIP)
        """
        base_filename = Path(original_filename).stem
        safe_filename = quote(base_filename)

        # If result has sections, create ZIP file
        if result.has_sections:
            logger.info(f"Creating ZIP with {result.section_count} sections")
            return self._create_zip_response(
                result=result,
                original_filename=original_filename,
                safe_filename=safe_filename,
                workflow_suffix=workflow_suffix,
                request_id=request_id,
            )
        else:
            logger.info("Returning single markdown file")
            markdown_content = self._append_metadata_footer(
                content=result.content,
                metadata=self._build_download_metadata(
                    result=result,
                    original_filename=original_filename,
                    request_id=request_id,
                ),
            )
            return self._create_single_file_response(
                markdown_content,
                safe_filename,
                workflow_suffix
            )

    def build_json_response(
        self,
        result: WorkflowResult,
        original_filename: str,
        request_time: datetime,
        request_id: str,
    ) -> OutlineExtractionResponse:
        """Build JSON response for /extract-json endpoint.

        Args:
            result: WorkflowResult from extraction
            original_filename: Original PDF filename
            request_time: Request timestamp

        Returns:
            OutlineExtractionResponse model
        """
        # Build extracted content list
        extracted_content = []

        if result.has_sections:
            # Multiple sections from outlines
            for section in result.sections:
                extracted_content.append(
                    ExtractedContent(
                        filename=section.filename,
                        content=section.content
                    )
                )
        else:
            # Single content (no outlines)
            base_filename = Path(original_filename).stem
            extracted_content.append(
                ExtractedContent(
                    filename=f"{base_filename}.md",
                    content=result.content
                )
            )

        # Build response
        response = OutlineExtractionResponse(
            request_id=request_id,
            file_name=original_filename,
            request_time=request_time,
            timestamp=datetime.now(timezone.utc),
            model=result.metadata.get("model", "pdf-extractor-v2"),
            extracted_content=extracted_content,
            metadata=self._build_response_metadata(result, original_filename),
            validation=self._build_legacy_validation(result.validation_report),
        )

        logger.info(
            f"Built JSON response: {len(extracted_content)} sections, "
            f"model={response.model}"
        )

        return response

    def _create_single_file_response(
        self,
        content: str,
        safe_filename: str,
        workflow_suffix: str = ""
    ) -> Response:
        """Create response for single markdown file.

        Args:
            content: Markdown content
            safe_filename: URL-safe base filename
            workflow_suffix: Optional suffix for filename

        Returns:
            FastAPI Response with markdown content
        """
        markdown_bytes = content.encode('utf-8')
        content_length = len(markdown_bytes)

        return Response(
            content=markdown_bytes,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": (
                    f"attachment; filename*=UTF-8''{safe_filename}{workflow_suffix}.md"
                ),
                "Content-Length": str(content_length)
            }
        )

    def _create_zip_response(
        self,
        result: WorkflowResult,
        original_filename: str,
        safe_filename: str,
        workflow_suffix: str = "",
        request_id: str = "",
    ) -> StreamingResponse:
        """Create ZIP response for multiple sections.

        Args:
            result: WorkflowResult with sections
            safe_filename: URL-safe base filename
            workflow_suffix: Optional suffix for filename

        Returns:
            StreamingResponse with ZIP file
        """
        # Create ZIP file in memory
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for section in result.sections:
                section_content = self._append_metadata_footer(
                    content=section.content,
                    metadata=self._build_download_metadata(
                        result=result,
                        original_filename=original_filename,
                        request_id=request_id,
                        section=section,
                    ),
                )
                # Add each section to ZIP
                zip_file.writestr(
                    section.filename,
                    section_content.encode('utf-8')
                )
                logger.debug(f"Added to ZIP: {section.filename}")

        zip_buffer.seek(0)
        zip_size = zip_buffer.getbuffer().nbytes

        logger.info(
            f"Created ZIP file: {result.section_count} sections, "
            f"size={zip_size} bytes"
        )

        # Return ZIP file
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": (
                    f"attachment; filename*=UTF-8''{safe_filename}{workflow_suffix}_sections.zip"
                ),
                "Content-Length": str(zip_size)
            }
        )

    def _build_response_metadata(
        self,
        result: WorkflowResult,
        original_filename: str,
    ) -> ExtractionResponseMetadata:
        """Build the public metadata object for JSON responses."""
        raw_metadata = result.metadata or {}

        return ExtractionResponseMetadata(
            workflow=raw_metadata.get("workflow"),
            ocr_mode=raw_metadata.get("ocr_mode"),
            routing_strategy=raw_metadata.get("routing_strategy"),
            total_pages=raw_metadata.get("total_pages"),
            page_types=raw_metadata.get("page_types"),
            page_routes=raw_metadata.get("page_routes"),
            timing=raw_metadata.get("timing"),
            validation_summary=self._build_validation_summary(result.validation_report),
            source_file_name=original_filename,
        )

    def _build_validation_summary(
        self,
        validation_report: dict | None,
    ) -> ValidationSummary | None:
        """Normalize validation metadata for API exposure."""
        if not validation_report:
            return None

        enabled = validation_report.get("enabled")
        if isinstance(enabled, str):
            enabled = enabled.lower() == "true"
        elif enabled is None:
            enabled = True

        status = validation_report.get("status")
        if status is None and enabled:
            status = "passed"

        return ValidationSummary(
            enabled=enabled,
            mode=validation_report.get("mode"),
            status=status,
            dual_source_pages=validation_report.get("dual_source_pages"),
            total_results=validation_report.get("total_results"),
            source_distribution=validation_report.get("source_distribution"),
        )

    def _build_legacy_validation(
        self,
        validation_report: dict | None,
    ) -> dict[str, str] | None:
        """Build the backward-compatible top-level validation field."""
        summary = self._build_validation_summary(validation_report)
        if summary is None:
            return None

        payload = {
            "enabled": "true" if summary.enabled else "false",
        }
        if summary.status:
            payload["status"] = summary.status
        return payload

    def _build_download_metadata(
        self,
        result: WorkflowResult,
        original_filename: str,
        request_id: str,
        section=None,
    ) -> dict:
        """Build the metadata JSON appended to markdown output."""
        public_metadata = self._build_response_metadata(result, original_filename).model_dump(
            exclude_none=True
        )
        footer_metadata = {
            "request_id": request_id,
            "file_name": original_filename,
        }

        for key in (
            "workflow",
            "ocr_mode",
            "routing_strategy",
            "total_pages",
            "page_types",
            "page_routes",
            "timing",
            "validation_summary",
        ):
            if key in public_metadata:
                footer_metadata[key] = public_metadata[key]

        if section is not None:
            footer_metadata["section_filename"] = section.filename
            footer_metadata["section_title"] = section.title
            footer_metadata["section_page_range"] = list(section.page_range)

        return footer_metadata

    def _append_metadata_footer(self, content: str, metadata: dict) -> str:
        """Append a visible JSON metadata footer to markdown content."""
        formatted_metadata = json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True)
        return f"{content.rstrip()}\n\n##### Metadata\n{formatted_metadata}"
