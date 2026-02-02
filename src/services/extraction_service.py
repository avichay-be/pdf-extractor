import logging
import asyncio
from typing import Optional
from fastapi import HTTPException

import fitz  # PyMuPDF

from src.core.config import settings
from src.core.constants import MARKDOWN_SECTION_SEPARATOR
from src.core.utils import encode_chunks_to_base64_async, combine_markdown_sections, format_page_header
from src.services.client_factory import get_client_factory

logger = logging.getLogger(__name__)

# Initialize client factory (lazy initialization of clients)
client_factory = get_client_factory()

# Convenience aliases for commonly used clients
pdf_processor = client_factory.pdf_processor
mistral_client = client_factory.mistral_client
openai_client = client_factory.openai_client
gemini_client = client_factory.gemini_client
azure_document_intelligence_client = client_factory.azure_document_intelligence_client


def extract_text_from_pdf(pdf_path: str) -> tuple[str, dict]:
    """
    Extract tables from PDF using pdfplumber.

    Strictly extracts tabular data using pdfplumber.
    Does not perform OCR or extract non-tabular text.
    Applies bidirectional text correction for proper Hebrew/Arabic display.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (markdown_content, metadata_dict)

    Raises:
        HTTPException: If PDF cannot be read or processed
    """
    logger.info(f"Extracting tables from PDF using pdfplumber: {pdf_path}")

    try:
        import pdfplumber
        import pandas as pd
        from bidi import get_display
        
        def fix_bidi_text(text: str) -> str:
            """Apply bidirectional text correction for RTL languages (Hebrew/Arabic)."""
            if not text or not isinstance(text, str):
                return text
            try:
                return get_display(text)
            except Exception:
                return text
        
        markdown_parts = []
        markdown_parts.append(f"# Extracted Tables\n\n")
        
        tables_found = 0
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Extract tables from page
                page_tables = page.extract_tables()
                
                if page_tables:
                    tables_found += len(page_tables)
                    logger.info(f"Found {len(page_tables)} tables on page {i+1}")
                    
                    for j, table_data in enumerate(page_tables):
                        # Convert to DataFrame
                        # Assume first row is header if it looks like one, otherwise treat as data
                        # For "as is" extraction, we'll just treat all as data or use first row as header
                        if len(table_data) > 1:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        else:
                            df = pd.DataFrame(table_data)
                        
                        # Apply bidirectional text correction to all cells (and headers)
                        df = df.map(fix_bidi_text)
                        df.columns = [fix_bidi_text(col) if isinstance(col, str) else col for col in df.columns]
                        
                        # Convert to markdown "as is"
                        markdown_table = df.to_markdown(index=False)

                        markdown_parts.append(f"### Table {j+1} (Page {i+1})\n\n")
                        markdown_parts.append(markdown_table)
                        markdown_parts.append(MARKDOWN_SECTION_SEPARATOR)

        if tables_found == 0:
            logger.warning("pdfplumber found no tables in the document")
            markdown_parts.append("> [!NOTE]\n> No tables were detected in this document.\n\n")

        # Combine all parts
        combined_markdown = "".join(markdown_parts)

        # Prepare metadata
        metadata = {
            "extraction_method": "pdfplumber_table_only",
            "library": "pdfplumber",
            "tables_found": tables_found
        }

        return combined_markdown, metadata

    except Exception as e:
        logger.error(f"pdfplumber extraction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract tables from PDF: {str(e)}"
        )


async def process_text_extraction(
    pdf_path: str,
    pdf_base64: Optional[str] = None,
    query: str = "text extraction"
) -> tuple[str, dict]:
    """
    Process PDF by extracting tables using pdfplumber (no OCR/AI).

    For digitally-generated PDFs, simply extract the tables "as is"
    without OCR or AI processing. Uses pdfplumber for table detection.

    Args:
        pdf_path: Path to PDF file
        pdf_base64: Pre-encoded base64 string (not used, kept for compatibility)
        query: Query string for context logging

    Returns:
        Tuple of (markdown_content, metadata_dict)

    Raises:
        HTTPException: If processing fails
    """
    logger.info(f"Processing PDF with text extraction for query: {query}")

    # Use simple text extraction (synchronous, so run in thread)
    markdown_content, metadata = await asyncio.to_thread(
        extract_text_from_pdf,
        pdf_path
    )

    # Add query to metadata
    metadata["query"] = query
    metadata["workflow"] = "text_extraction"

    return markdown_content, metadata


async def process_azure_document_intelligence(
    pdf_path: str,
    pdf_base64: Optional[str] = None,
    query: str = "azure document intelligence"
) -> tuple[str, dict]:
    """
    Process PDF using Azure Document Intelligence API for smart table extraction.

    Uses Azure's Document Intelligence API to detect and extract tables with
    intelligent merging across pages. Ideal for complex financial documents.

    Args:
        pdf_path: Path to PDF file
        pdf_base64: Pre-encoded base64 string (optional, for performance)
        query: Query string for context logging

    Returns:
        Tuple of (markdown_content, metadata_dict)

    Raises:
        HTTPException: If Azure DI client not configured or processing fails
    """
    logger.info(f"Processing PDF with Azure Document Intelligence for query: {query}")

    if azure_document_intelligence_client is None:
        raise HTTPException(
            status_code=500,
            detail="Azure Document Intelligence client not configured. "
                   "Please set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and "
                   "AZURE_DOCUMENT_INTELLIGENCE_KEY in .env"
        )

    try:
        # Extract tables using Azure DI with smart merging
        markdown_tables, metadata = await azure_document_intelligence_client.extract_tables(
            pdf_path=pdf_path,
            pdf_base64=pdf_base64,
            merge_tables=True
        )

        # Combine all tables into markdown content
        markdown_content = combine_markdown_sections(
            markdown_tables,
            empty_message="# No tables found in document\n\n"
        )
        if not markdown_tables:
            logger.warning("Azure DI found no tables in the document")

        # Add query to metadata
        metadata["query"] = query
        metadata["workflow"] = "azure_document_intelligence"
        metadata["extraction_method"] = "azure_document_intelligence_api"

        logger.info(f"Successfully extracted {len(markdown_tables)} tables using Azure DI")

        return markdown_content, metadata

    except Exception as e:
        logger.error(f"Azure Document Intelligence processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF with Azure Document Intelligence: {str(e)}"
        )


async def process_with_model(
    model_name: str,
    chunk_path: str,
    chunk_base64: str,
    chunk_bytes: bytes,
    has_query: bool,
    enable_validation: bool,
    validation_model: str = None,
    workflow_name: str = None
):
    """
    Process a PDF chunk using the specified model.

    Args:
        model_name: Model to use ('mistral', 'openai', 'gemini', 'claude')
        chunk_path: Path to PDF chunk file
        chunk_base64: Base64-encoded PDF content
        chunk_bytes: PDF bytes (in-memory to prevent file system race conditions)
        has_query: Whether query filtering is active
        enable_validation: Whether to enable cross-validation
        validation_model: Model to use for validation (if different from extraction model)
        workflow_name: Name of the workflow (e.g., "01_Fin_Reports") for workflow-specific validation

    Returns:
        Tuple of (markdown_content, validation_report_dict)
    """
    if model_name == "mistral":
        # Use Mistral's batch document processing API
        # If validation is enabled and validation_model is specified, use that model
        return await mistral_client.process_document(
            pdf_path=chunk_path,
            pdf_base64=chunk_base64,
            pdf_bytes=chunk_bytes,
            has_query=has_query,
            enable_validation=enable_validation,
            workflow_name=workflow_name
        )

    else:
        # For OpenAI/Gemini, process page-by-page
        client_map = {
            "openai": openai_client,
            "gemini": gemini_client
        }

        client = client_map.get(model_name)
        if not client:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{model_name}' is not available. Please configure the API key."
            )

        # Open PDF and process each page
        pdf_document = fitz.open(chunk_path)
        page_contents = []

        logger.info(f"Processing {len(pdf_document)} pages with {model_name}...")

        # OPTIMIZATION: Read PDF bytes once before loop (eliminates redundant I/O)
        # For 100-page chunk: saves 99 unnecessary file reads (~50-100ms)
        with open(chunk_path, 'rb') as f:
            pdf_bytes = f.read()

        for page_num in range(len(pdf_document)):
            # Reuse pdf_bytes read above (no file I/O in loop)

            # Extract content using the selected model (synchronous, run in thread)
            content = await asyncio.to_thread(
                client.extract_page_content,
                pdf_bytes,
                page_num
            )

            # Add page number header (1-based indexing for user display)
            page_contents.append(format_page_header(page_num) + content)

        pdf_document.close()

        # Combine all pages
        combined_markdown = combine_markdown_sections(page_contents)

        # If validation is enabled, use a different model to validate
        validation_report_dict = None
        if enable_validation and validation_model:
            logger.info(f"Cross-validating {model_name} extraction with {validation_model}...")
            try:
                # Use another model to validate the extraction
                # Create a mock MistralOCRResponse to match expected format
                from src.models.mistral_models import MistralOCRResponse, Page, Dimensions, UsageInfo
                from src.services.validation import ValidationService

                # Create mock page objects
                pages = [
                    Page(
                        index=i,
                        markdown=content,
                        dimensions=Dimensions(dpi=72, height=1000, width=800)
                    )
                    for i, content in enumerate(page_contents)
                ]

                mock_response = MistralOCRResponse(
                    model=model_name,
                    pages=pages,
                    usage_info=UsageInfo(
                        pages_processed=len(pages),
                        doc_size_bytes=0,
                        pages_processed_annotation=0
                    )
                )

                # Initialize validation service with OpenAI validator
                validation_service = ValidationService()

                # Validate using the specified validator
                validation_report = await validation_service.cross_validate_pages(
                    mock_response,
                    chunk_path,
                    has_query=has_query
                )

                # Apply fixes for problem pages
                for result in validation_report.validation_results:
                    if result.has_problem_pattern and result.alternative_content:
                        logger.info(
                            f"[Page {result.page_number}] Replacing problematic {model_name} content "
                            f"with {validation_model} extraction"
                        )
                        page_contents[result.page_number] = result.alternative_content

                # Recombine if any pages were fixed
                combined_markdown = combine_markdown_sections(page_contents)

                # Create validation status
                has_problems = len(validation_report.problem_pages) > 0
                has_warnings = len(validation_report.failed_validations) > 0

                if has_problems:
                    status = "problems_fixed"
                elif has_warnings:
                    status = "warnings"
                else:
                    status = "passed"

                validation_report_dict = {
                    "enabled": "true",
                    "status": status
                }

                logger.info(f"Validation complete: status={status}")

            except Exception as e:
                logger.error(f"Cross-validation failed: {e}")
                validation_report_dict = None

        return combined_markdown, validation_report_dict



async def process_ocr_with_images(
    pdf_path: str,
    pdf_base64: Optional[str] = None,
    query: str = None
) -> tuple[str, dict]:
    """
    Process PDF with Mistral OCR and extract data from detected images using OpenAI.

    Supports large PDFs by automatically chunking them to respect Mistral's 30-page limit.

    Args:
        pdf_path: Path to PDF file
        pdf_base64: Pre-encoded base64 string (ignored, chunks are re-encoded)
        query: Query string (used as prompt for image extraction if provided)

    Returns:
        Tuple of (markdown_content, metadata_dict)
    """
    logger.info(f"Processing PDF with OCR with Images workflow for query: {query}")

    pdf_chunks = []
    try:
        # 1. Split PDF into chunks (respects MAX_PAGES_PER_CHUNK to stay under Mistral's 30-page limit)
        pdf_chunks, _ = pdf_processor.split_with_outline_info(pdf_path)
        logger.info(f"Split PDF into {len(pdf_chunks)} chunks for OCR with images processing")

        # 2. Pre-encode all chunks to base64 in parallel
        logger.info(f"Pre-encoding {len(pdf_chunks)} chunks to base64 in parallel...")
        encoded_chunks = await encode_chunks_to_base64_async(pdf_chunks)

        # 3. Process all chunks with Mistral in parallel (include_images=True)
        logger.info(f"Processing {len(pdf_chunks)} chunks with Mistral OCR in parallel...")

        async def process_chunk_with_images(chunk_path: str, chunk_base64: str) -> tuple[str, list]:
            """Process a single chunk and return (content, images)."""
            content, chunk_metadata = await mistral_client.process_document(
                pdf_path=chunk_path,
                pdf_base64=chunk_base64,
                include_images=True,
                enable_validation=False
            )
            images = chunk_metadata.get("images", []) if chunk_metadata else []
            return content, images

        # Process all chunks concurrently
        tasks = [process_chunk_with_images(chunk_path, chunk_base64)
                 for chunk_path, chunk_base64 in encoded_chunks]
        chunk_results = await asyncio.gather(*tasks)

        # 4. Collect all content and images from all chunks
        all_content = [content for content, _ in chunk_results]
        all_images = []
        for _, images in chunk_results:
            all_images.extend(images)

        logger.info(f"Processed {len(pdf_chunks)} chunks, found {len(all_images)} total images")

        # Combine Mistral content from all chunks
        mistral_content = pdf_processor.combine_markdown_results(all_content)

        # 5. Process images with OpenAI if any were found
        if not all_images:
            logger.info("No images detected by Mistral in any chunk.")
            metadata = {
                "workflow": "ocr_with_images",
                "chunks_processed": len(pdf_chunks),
                "images_processed": 0
            }
            return mistral_content, metadata

        logger.info(f"Found {len(all_images)} images across all chunks. Processing with OpenAI...")

        # Determine prompt
        prompt = query if query else settings.OCR_WITH_IMAGES_DEFAULT_PROMPT

        # Process each image with OpenAI
        image_extractions = []

        for i, img in enumerate(all_images):
            try:
                img_base64 = img.get('image_base64') or img.get('base64')

                if not img_base64:
                    logger.warning(f"Image {i} has no base64 data. Keys: {img.keys()}")
                    continue

                extraction = await asyncio.to_thread(
                    openai_client.extract_from_image,
                    img_base64,
                    prompt
                )

                image_extractions.append(f"### Image {i+1} Extraction\n\n{extraction}\n")

            except Exception as e:
                logger.error(f"Failed to process image {i}: {e}")
                image_extractions.append(f"### Image {i+1} Extraction\n\n> [!WARNING]\n> Failed to extract data from this image: {str(e)}\n")

        # 6. Append image extractions to content
        if image_extractions:
            mistral_content += MARKDOWN_SECTION_SEPARATOR + "## Extracted Image Data\n\n" + "\n\n".join(image_extractions)

        # Prepare metadata
        metadata = {
            "workflow": "ocr_with_images",
            "chunks_processed": len(pdf_chunks),
            "images_processed": len(image_extractions),
            "prompt_used": prompt
        }

        return mistral_content, metadata

    finally:
        # Cleanup chunk files
        if pdf_chunks:
            pdf_processor.cleanup_chunks(pdf_chunks, original_path=pdf_path)


async def process_gemini_wf(
    pdf_path: str,
    pdf_base64: Optional[str] = None,
    query: str = None
) -> tuple[str, dict]:
    """
    Process PDF page-by-page using Gemini with async processing.

    This workflow sends each page to Gemini with OCR prompt and combines
    results into a single markdown document. Processes pages asynchronously
    for better performance.

    Args:
        pdf_path: Path to PDF file
        pdf_base64: Pre-encoded base64 string (not used, kept for compatibility)
        query: Query string for context logging

    Returns:
        Tuple of (markdown_content, metadata_dict)

    Raises:
        HTTPException: If Gemini client not configured or processing fails
    """
    logger.info(f"Processing PDF with Gemini page-by-page workflow for query: {query}")

    if gemini_client is None:
        raise HTTPException(
            status_code=500,
            detail="Gemini client not configured. "
                   "Please set GEMINI_API_KEY in .env"
        )

    try:
        # Open PDF to get page count
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        pdf_document.close()

        logger.info(f"Processing {total_pages} pages with Gemini asynchronously...")

        # Read PDF bytes once (reuse for all page extractions)
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        # Process all pages asynchronously
        async def process_page(page_num: int) -> tuple[int, str]:
            """Process a single page and return (page_num, content)."""
            content = await asyncio.to_thread(
                gemini_client.extract_page_content,
                pdf_bytes,
                page_num
            )
            # Add page number header (1-based indexing)
            return page_num, format_page_header(page_num) + content

        # Create tasks for all pages
        tasks = [process_page(page_num) for page_num in range(total_pages)]

        # Process all pages concurrently
        results = await asyncio.gather(*tasks)

        # Sort by page number (should already be sorted, but ensure order)
        results.sort(key=lambda x: x[0])

        # Extract content in order
        page_contents = [content for _, content in results]

        # Combine all pages
        combined_markdown = combine_markdown_sections(page_contents)

        # Prepare metadata
        metadata = {
            "workflow": "gemini-wf",
            "extraction_method": "gemini_page_by_page",
            "model": settings.GEMINI_MODEL,
            "total_pages": total_pages,
            "query": query
        }

        logger.info(f"Successfully processed {total_pages} pages with Gemini")

        return combined_markdown, metadata

    except Exception as e:
        logger.error(f"Gemini page-by-page processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF with Gemini: {str(e)}"
        ) 
