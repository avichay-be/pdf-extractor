"""
PDF extraction API endpoints.

Provides the retained page-aware extraction API for file uploads and base64
requests.
"""
from fastapi import APIRouter, File, UploadFile, Depends, Request
from typing import Optional
import logging
from datetime import datetime, timezone

from src.core.security import verify_api_key
from src.core.error_handling import (
    generate_request_id,
    get_request_id,
    handle_extraction_errors,
    set_request_id,
)
from src.core.logging_utils import log_event
from src.models.api_models import Base64FileRequest, OutlineExtractionResponse
from src.services.page_aware_extraction_service import (
    build_page_aware_extraction_service,
)
from src.services.pdf_input_handler import PDFInputHandler
from src.services.response_builder import ResponseBuilder

logger = logging.getLogger(__name__)
router = APIRouter()


def _apply_request_id(request: Request, request_id: str) -> str:
    """Override the request ID for this request and keep middleware in sync."""
    request.state.request_id = request_id
    set_request_id(request_id)
    return request_id

@router.post("/extract", dependencies=[Depends(verify_api_key)])
@handle_extraction_errors("Failed to extract PDF content")
async def extract_pdf_content(
    request: Request,
    file: UploadFile = File(...),
    enable_cross_validation: Optional[bool] = None,
    enable_polishing: Optional[bool] = None,
):
    """
    Extract content from PDF using the default page-aware business workflow.

    Automatically routes pages between digital extraction and OCR:
    - Embedded text pages use text extraction
    - Image/scanned pages use the default OCR mode
    - Mixed pages use OCR as primary output with digital text as validation input

    Args:
        file: PDF file to process
        enable_cross_validation: Enable cross-validation (overrides global setting)
        enable_polishing: Enable the Gemini polish pass (overrides global setting)

    Returns:
        Markdown content as single file or ZIP with multiple sections
    """
    pdf_handler = PDFInputHandler()
    extraction_service = build_page_aware_extraction_service()
    response_builder = ResponseBuilder()
    request_id = _apply_request_id(request, get_request_id() or generate_request_id())

    try:
        # 1. Save uploaded file
        pdf_path = await pdf_handler.save_uploaded_file(file)
        log_event(
            logger,
            logging.INFO,
            "extract_request_started",
            filename=file.filename,
            validation_requested=enable_cross_validation,
            polishing_requested=enable_polishing,
        )

        # 2. Execute the page-aware extraction pipeline
        result = await extraction_service.extract_document(
            pdf_path=pdf_path,
            enable_validation=enable_cross_validation,
            enable_polishing=enable_polishing,
        )

        # 3. Build response
        log_event(
            logger,
            logging.INFO,
            "extract_request_completed",
            filename=file.filename,
            workflow=result.metadata.get("workflow"),
            ocr_mode=result.metadata.get("ocr_mode"),
            sections=result.section_count,
        )

        workflow_suffix = ""
        if result.metadata.get("workflow") == "ocr":
            workflow_suffix = f"_ocr_{result.metadata.get('ocr_mode', 'all_around')}"

        return response_builder.build_download_response(
            result=result,
            original_filename=file.filename,
            workflow_suffix=workflow_suffix,
            request_id=request_id,
        )

    finally:
        # Cleanup temporary files
        await pdf_handler.cleanup()


@router.post(
    "/extract-json",
    response_model=OutlineExtractionResponse,
    dependencies=[Depends(verify_api_key)]
)
@handle_extraction_errors("Failed to extract PDF content from base64")
async def extract_pdf_from_base64(
    http_request: Request,
    request: Base64FileRequest,
):
    """
    Extract content from base64-encoded PDF.

    Returns JSON with structured extraction results using the same default
    page-aware flow as /extract.

    Args:
        request: JSON body with filename, base64 content, and options

    Returns:
        JSON with file metadata and array of extracted content sections
    """
    pdf_handler = PDFInputHandler()
    extraction_service = build_page_aware_extraction_service()
    response_builder = ResponseBuilder()
    request_time = datetime.now(timezone.utc)
    selected_request_id = _apply_request_id(
        request=http_request,
        request_id=request.request_id or get_request_id() or generate_request_id(),
    )

    try:
        # 1. Decode and save base64 file
        pdf_path = await pdf_handler.save_base64_file(
            base64_content=request.file_content,
            filename=request.filename
        )
        log_event(
            logger,
            logging.INFO,
            "extract_json_request_started",
            filename=request.filename,
            validation_requested=request.enable_cross_validation,
            polishing_requested=request.enable_polishing,
        )

        # 2. Execute the page-aware extraction pipeline
        result = await extraction_service.extract_document(
            pdf_path=pdf_path,
            enable_validation=request.enable_cross_validation,
            enable_polishing=request.enable_polishing,
        )

        # 3. Build JSON response
        log_event(
            logger,
            logging.INFO,
            "extract_json_request_completed",
            filename=request.filename,
            workflow=result.metadata.get("workflow"),
            ocr_mode=result.metadata.get("ocr_mode"),
            sections=result.section_count,
        )

        return response_builder.build_json_response(
            result=result,
            original_filename=request.filename,
            request_time=request_time,
            request_id=selected_request_id,
        )

    finally:
        # Cleanup temporary files
        await pdf_handler.cleanup()
