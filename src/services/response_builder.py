"""
Response builder for PDF extraction endpoints.

Handles formatting responses for both multipart/form-data and JSON endpoints,
including single file downloads and ZIP file creation for multiple sections.
"""
import logging
from io import BytesIO
from pathlib import Path
from urllib.parse import quote
from datetime import datetime
import zipfile

from fastapi import Response
from fastapi.responses import StreamingResponse

from src.models.api_models import OutlineExtractionResponse, ExtractedContent
from src.models.workflow_models import WorkflowResult

logger = logging.getLogger(__name__)


class ResponseBuilder:
    """Builds responses for extraction endpoints."""

    def build_download_response(
        self,
        result: WorkflowResult,
        original_filename: str,
        workflow_suffix: str = ""
    ) -> Response | StreamingResponse:
        """Build download response for /extract endpoint.

        Returns either a single markdown file or a ZIP file with multiple sections.

        Args:
            result: WorkflowResult from extraction
            original_filename: Original PDF filename
            workflow_suffix: Optional suffix for filename (e.g., "_text", "_azure_di")

        Returns:
            FastAPI Response (single file) or StreamingResponse (ZIP)
        """
        base_filename = Path(original_filename).stem
        safe_filename = quote(base_filename)

        # If result has sections, create ZIP file
        if result.has_sections:
            logger.info(f"Creating ZIP with {result.section_count} sections")
            return self._create_zip_response(result, safe_filename, workflow_suffix)
        else:
            logger.info("Returning single markdown file")
            return self._create_single_file_response(
                result.content,
                safe_filename,
                workflow_suffix
            )

    def build_json_response(
        self,
        result: WorkflowResult,
        original_filename: str,
        request_time: datetime
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
            file_name=original_filename,
            request_time=request_time,
            timestamp=datetime.utcnow(),
            model=result.metadata.get("model", "pdf-extractor-v2"),
            extracted_content=extracted_content
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
        safe_filename: str,
        workflow_suffix: str = ""
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
                # Add each section to ZIP
                zip_file.writestr(
                    section.filename,
                    section.content.encode('utf-8')
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
