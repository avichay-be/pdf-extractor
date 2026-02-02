"""
PDF input handler for file upload and base64 decoding.

Handles saving uploaded files and decoding base64 content to temporary files.
Provides cleanup functionality for temporary files.
"""
import logging
import tempfile
import base64
from pathlib import Path
from typing import Optional
from fastapi import UploadFile

from src.core.config import settings
from src.core.error_handling import PDFValidationError, FileEncodingError

logger = logging.getLogger(__name__)


class PDFInputHandler:
    """Handles PDF file input operations."""

    def __init__(self):
        """Initialize PDF input handler."""
        self.temp_files: list[str] = []
        logger.debug("PDFInputHandler initialized")

    async def save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to temporary location.

        Args:
            file: UploadFile from FastAPI

        Returns:
            Path to temporary file

        Raises:
            PDFValidationError: If file type is not PDF
            FileEncodingError: If file save fails
        """
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise PDFValidationError("Only PDF files are supported")
        if file.content_type not in {"application/pdf", "application/octet-stream", None}:
            raise PDFValidationError("Invalid content type; only application/pdf is allowed")

        safe_filename = self._sanitize_filename(file.filename)

        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                content = await file.read()
                self._enforce_size_limit(len(content))

                tmp_file.write(content)
                tmp_file_path = tmp_file.name

            self.temp_files.append(tmp_file_path)
            logger.info(f"Saved uploaded file: {safe_filename} ({len(content)} bytes)")

            return tmp_file_path

        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            raise FileEncodingError(f"Failed to save uploaded file: {str(e)}")

    async def save_base64_file(self, base64_content: str, filename: str = "document.pdf") -> str:
        """Decode base64 content and save to temporary file.

        Args:
            base64_content: Base64-encoded PDF content
            filename: Original filename (for logging)

        Returns:
            Path to temporary file

        Raises:
            FileEncodingError: If base64 decoding or file save fails
        """
        try:
            # Decode base64 content
            safe_filename = self._sanitize_filename(filename)
            logger.info(f"Decoding base64 content for: {safe_filename}")

            if len(base64_content) > settings.MAX_BASE64_LENGTH:
                raise PDFValidationError("Base64 payload too large")

            pdf_bytes = base64.b64decode(base64_content)
            self._enforce_size_limit(len(pdf_bytes))

            if not pdf_bytes.startswith(b"%PDF"):
                raise PDFValidationError("Uploaded content is not a PDF")

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_file_path = tmp_file.name

            self.temp_files.append(tmp_file_path)
            logger.info(f"Saved base64 file: {safe_filename} ({len(pdf_bytes)} bytes)")

            return tmp_file_path

        except Exception as e:
            logger.error(f"Failed to decode/save base64 content: {e}")
            raise FileEncodingError(f"Failed to decode PDF from base64: {str(e)}")

    async def cleanup(self):
        """Clean up all temporary files created by this handler.

        Safe to call multiple times.
        """
        if not self.temp_files:
            return

        logger.info(f"Cleaning up {len(self.temp_files)} temporary files")

        for file_path in self.temp_files:
            try:
                Path(file_path).unlink(missing_ok=True)
                logger.debug(f"Deleted temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {file_path}: {e}")

        self.temp_files.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sanitize_filename(self, filename: str) -> str:
        """Strip path components and control characters from filenames."""
        safe_name = Path(filename).name
        safe_name = "".join(ch for ch in safe_name if ch.isprintable())
        if not safe_name.lower().endswith(".pdf"):
            safe_name = f"{safe_name}.pdf"
        return safe_name

    def _enforce_size_limit(self, size_bytes: int):
        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise PDFValidationError(
                f"PDF exceeds max allowed size of {settings.MAX_UPLOAD_MB} MB"
            )
