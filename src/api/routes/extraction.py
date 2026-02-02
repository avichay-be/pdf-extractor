"""
PDF extraction API endpoints.

Provides endpoints for PDF content extraction using various workflows.
Supports both multipart file upload and base64-encoded JSON requests.
"""
from fastapi import APIRouter, File, UploadFile, Depends
from typing import Optional
import logging
from datetime import datetime

from src.core.security import verify_api_key
from src.core.error_handling import handle_extraction_errors
from src.models.api_models import Base64FileRequest, OutlineExtractionResponse
from src.services.workflow_orchestrator import get_workflow_orchestrator
from src.services.pdf_input_handler import PDFInputHandler
from src.services.response_builder import ResponseBuilder

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/extract", dependencies=[Depends(verify_api_key)])
@handle_extraction_errors("Failed to extract PDF content")
async def extract_pdf_content(
    file: UploadFile = File(...),
    query: str = "01_Fin_Reports",
    enable_validation: Optional[bool] = None
):
    """
    Extract content from PDF using appropriate workflow based on query.

    Automatically routes to specialized workflows:
    - Text extraction (pdfplumber) for "text_extraction" query
    - Azure Document Intelligence for "azure_document_intelligence" query
    - OCR with Images for "ocr_with_images" query
    - Gemini for "gemini-wf" query
    - Mistral with validation (default)

    Args:
        file: PDF file to process
        query: Query string for workflow selection and outline filtering
        enable_validation: Enable cross-validation (overrides global setting)

    Returns:
        Markdown content as single file or ZIP with multiple sections
    """
    pdf_handler = PDFInputHandler()
    orchestrator = get_workflow_orchestrator()
    response_builder = ResponseBuilder()

    try:
        # 1. Save uploaded file
        pdf_path = await pdf_handler.save_uploaded_file(file)
        logger.info(f"Processing PDF: {file.filename} with query: {query}")

        # 2. Execute workflow via orchestrator
        result = await orchestrator.execute_workflow(
            pdf_path=pdf_path,
            query=query,
            enable_validation=enable_validation
        )

        # 3. Build response
        logger.info(
            f"Successfully processed PDF: {file.filename}, "
            f"workflow={result.metadata.get('workflow')}, "
            f"sections={result.section_count}"
        )

        # Determine workflow suffix for filename
        workflow = result.metadata.get('workflow', '')
        workflow_suffix_map = {
            'text_extraction': '_text',
            'azure_document_intelligence': '_azure_di',
            'ocr_with_images': '_ocr_images',
            'gemini-wf': '_gemini',
            'mistral': ''
        }
        workflow_suffix = workflow_suffix_map.get(workflow, '')

        return response_builder.build_download_response(
            result=result,
            original_filename=file.filename,
            workflow_suffix=workflow_suffix
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
async def extract_pdf_from_base64(request: Base64FileRequest):
    """
    Extract content from base64-encoded PDF.

    Returns JSON with structured extraction results. Supports same workflows
    as /extract endpoint, automatically routing based on query pattern.

    Args:
        request: JSON body with filename, base64 content, query, and options

    Returns:
        JSON with file metadata and array of extracted content sections
    """
    pdf_handler = PDFInputHandler()
    orchestrator = get_workflow_orchestrator()
    response_builder = ResponseBuilder()
    request_time = datetime.utcnow()

    try:
        # 1. Decode and save base64 file
        pdf_path = await pdf_handler.save_base64_file(
            base64_content=request.file_content,
            filename=request.filename
        )
        logger.info(f"Processing base64 PDF: {request.filename} with query: {request.query}")

        # 2. Execute workflow via orchestrator
        result = await orchestrator.execute_workflow(
            pdf_path=pdf_path,
            query=request.query,
            enable_validation=request.enable_validation
        )

        # 3. Build JSON response
        logger.info(
            f"Successfully processed base64 PDF: {request.filename}, "
            f"workflow={result.metadata.get('workflow')}, "
            f"sections={result.section_count}"
        )

        return response_builder.build_json_response(
            result=result,
            original_filename=request.filename,
            request_time=request_time
        )

    finally:
        # Cleanup temporary files
        await pdf_handler.cleanup()
