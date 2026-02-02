"""
Error handling utilities for PDF extraction operations.

This module provides custom exceptions and decorators for consistent error handling
across the application.
"""
import asyncio
import logging
import time
import uuid
from typing import Callable, TypeVar, ParamSpec
from functools import wraps
from contextvars import ContextVar

from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Context variable for request ID tracking across async contexts
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

# Type variables for generic function signatures
P = ParamSpec('P')
T = TypeVar('T')


# ============================================================================
# Custom Exceptions
# ============================================================================

class PDFExtractionError(Exception):
    """Base exception for PDF extraction errors."""
    pass


class PDFValidationError(PDFExtractionError):
    """PDF validation failed."""
    pass


class WorkflowExecutionError(PDFExtractionError):
    """Workflow execution failed."""
    pass


class ClientConfigurationError(PDFExtractionError):
    """Client not properly configured."""
    pass


class TableExtractionError(PDFExtractionError):
    """Table extraction failed."""
    pass


class FileEncodingError(PDFExtractionError):
    """File encoding/decoding failed."""
    pass


# ============================================================================
# Error Handler Decorator
# ============================================================================

def handle_extraction_errors(
    error_message: str = "Operation failed"
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to handle errors in extraction operations.

    Automatically converts extraction errors to appropriate HTTP exceptions
    and logs them. Works with both sync and async functions.

    Args:
        error_message: Custom error message prefix

    Returns:
        Decorated function with error handling

    Example:
        @handle_extraction_errors("Failed to extract PDF")
        async def extract_pdf(pdf_path: str) -> str:
            # Your extraction logic here
            pass
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:

        @wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            """Async wrapper for error handling with request tracking and timing."""
            # Generate request ID if not already set
            if not request_id_var.get():
                request_id_var.set(str(uuid.uuid4()))

            request_id = request_id_var.get()
            start_time = time.time()

            try:
                logger.info(f"[{request_id}] Starting {func.__name__}")
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time

                # Log with warning if response time exceeds threshold
                from src.core.config import settings
                threshold_ms = settings.RESPONSE_TIME_WARNING_THRESHOLD_MS
                elapsed_ms = elapsed * 1000
                if elapsed_ms > threshold_ms:
                    logger.warning(
                        f"[{request_id}] ⚠️  SLOW RESPONSE: {func.__name__} took {elapsed:.2f}s "
                        f"({elapsed_ms:.0f}ms > {threshold_ms}ms threshold)"
                    )
                else:
                    logger.info(f"[{request_id}] Completed {func.__name__} in {elapsed:.2f}s")

                return result
            except PDFValidationError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Validation error after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF validation failed: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except ClientConfigurationError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Configuration error after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Service configuration error: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except WorkflowExecutionError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Workflow error after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Workflow execution failed: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except PDFExtractionError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Extraction error after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=str(e),
                    headers={"X-Request-ID": request_id}
                )
            except FileNotFoundError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - File not found after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except ValueError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Invalid value after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid input: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception(f"[{request_id}] {error_message} - Unexpected error in {func.__name__} after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{error_message}: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            """Sync wrapper for error handling with request tracking and timing."""
            # Generate request ID if not already set
            if not request_id_var.get():
                request_id_var.set(str(uuid.uuid4()))

            request_id = request_id_var.get()
            start_time = time.time()

            try:
                logger.info(f"[{request_id}] Starting {func.__name__}")
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                # Log with warning if response time exceeds threshold
                from src.core.config import settings
                threshold_ms = settings.RESPONSE_TIME_WARNING_THRESHOLD_MS
                elapsed_ms = elapsed * 1000
                if elapsed_ms > threshold_ms:
                    logger.warning(
                        f"[{request_id}] ⚠️  SLOW RESPONSE: {func.__name__} took {elapsed:.2f}s "
                        f"({elapsed_ms:.0f}ms > {threshold_ms}ms threshold)"
                    )
                else:
                    logger.info(f"[{request_id}] Completed {func.__name__} in {elapsed:.2f}s")

                return result
            except PDFValidationError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Validation error after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF validation failed: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except ClientConfigurationError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Configuration error after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Service configuration error: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except WorkflowExecutionError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Workflow error after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Workflow execution failed: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except PDFExtractionError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Extraction error after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=str(e),
                    headers={"X-Request-ID": request_id}
                )
            except FileNotFoundError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - File not found after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except ValueError as e:
                elapsed = time.time() - start_time
                logger.error(f"[{request_id}] {error_message} - Invalid value after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid input: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )
            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception(f"[{request_id}] {error_message} - Unexpected error in {func.__name__} after {elapsed:.2f}s: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"{error_message}: {str(e)}",
                    headers={"X-Request-ID": request_id}
                )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator
