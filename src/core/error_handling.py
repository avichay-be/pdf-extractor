"""
Error handling utilities for PDF extraction operations.

This module provides custom exceptions and decorators for consistent error handling
across the application.
"""
import asyncio
import inspect
import logging
import time
import uuid
from typing import Callable, TypeVar, ParamSpec
from functools import wraps
from contextvars import ContextVar

from fastapi import HTTPException
from src.core.logging_utils import build_log_extra, log_event

logger = logging.getLogger(__name__)

# Context variable for request ID tracking across async contexts
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

# Type variables for generic function signatures
P = ParamSpec('P')
T = TypeVar('T')


def generate_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())


def get_request_id() -> str:
    """Return the current request ID, if one is set."""
    return request_id_var.get()


def set_request_id(request_id: str) -> str:
    """Override the current request ID for this request context."""
    request_id_var.set(request_id)
    return request_id


def ensure_request_id() -> str:
    """Return the current request ID, creating one when missing."""
    current_request_id = get_request_id()
    if current_request_id:
        return current_request_id

    current_request_id = generate_request_id()
    set_request_id(current_request_id)
    return current_request_id


def resolve_request_id_for_handler(
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> str:
    """Resolve the final request ID for endpoint handlers before logging."""
    bound_arguments = inspect.signature(func).bind_partial(*args, **kwargs).arguments

    request_object = next(
        (
            value for value in bound_arguments.values()
            if hasattr(value, "state") and hasattr(value, "headers")
        ),
        None,
    )

    if func.__name__ == "extract_pdf_content":
        request_id = generate_request_id()
        if request_object is not None:
            request_object.state.request_id = request_id
        return set_request_id(request_id)

    if func.__name__ == "extract_pdf_from_base64":
        request_body = next(
            (
                value for value in bound_arguments.values()
                if hasattr(value, "request_id")
                and hasattr(value, "filename")
                and hasattr(value, "file_content")
            ),
            None,
        )
        request_id = getattr(request_body, "request_id", None) or generate_request_id()
        if request_object is not None:
            request_object.state.request_id = request_id
        return set_request_id(request_id)

    return ensure_request_id()


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
            request_id = resolve_request_id_for_handler(func, args, kwargs)
            start_time = time.time()

            try:
                log_event(
                    logger,
                    logging.INFO,
                    "handler_started",
                    function=func.__name__,
                )
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time

                # Log with warning if response time exceeds threshold
                from src.core.config import settings
                threshold_ms = settings.RESPONSE_TIME_WARNING_THRESHOLD_MS
                elapsed_ms = elapsed * 1000
                if elapsed_ms > threshold_ms:
                    log_event(
                        logger,
                        logging.WARNING,
                        "handler_slow",
                        function=func.__name__,
                        elapsed_ms=round(elapsed_ms, 1),
                        threshold_ms=threshold_ms,
                    )
                else:
                    log_event(
                        logger,
                        logging.INFO,
                        "handler_completed",
                        function=func.__name__,
                        elapsed_ms=round(elapsed_ms, 1),
                    )

                return result
            except PDFValidationError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF validation failed: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except ClientConfigurationError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Service configuration error: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except WorkflowExecutionError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Workflow execution failed: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except PDFExtractionError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=400,
                    detail=str(e),
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except FileNotFoundError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except ValueError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid input: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception(
                    "handler_failed",
                    extra=build_log_extra(
                        function=func.__name__,
                        error_message=error_message,
                        error_type=type(e).__name__,
                        elapsed_ms=round(elapsed * 1000, 1),
                        reason=str(e),
                    ),
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"{error_message}: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            """Sync wrapper for error handling with request tracking and timing."""
            request_id = resolve_request_id_for_handler(func, args, kwargs)
            start_time = time.time()

            try:
                log_event(
                    logger,
                    logging.INFO,
                    "handler_started",
                    function=func.__name__,
                )
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time

                # Log with warning if response time exceeds threshold
                from src.core.config import settings
                threshold_ms = settings.RESPONSE_TIME_WARNING_THRESHOLD_MS
                elapsed_ms = elapsed * 1000
                if elapsed_ms > threshold_ms:
                    log_event(
                        logger,
                        logging.WARNING,
                        "handler_slow",
                        function=func.__name__,
                        elapsed_ms=round(elapsed_ms, 1),
                        threshold_ms=threshold_ms,
                    )
                else:
                    log_event(
                        logger,
                        logging.INFO,
                        "handler_completed",
                        function=func.__name__,
                        elapsed_ms=round(elapsed_ms, 1),
                    )

                return result
            except PDFValidationError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF validation failed: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except ClientConfigurationError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Service configuration error: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except WorkflowExecutionError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Workflow execution failed: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except PDFExtractionError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=400,
                    detail=str(e),
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except FileNotFoundError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except ValueError as e:
                elapsed = time.time() - start_time
                log_event(
                    logger,
                    logging.ERROR,
                    "handler_failed",
                    function=func.__name__,
                    error_message=error_message,
                    error_type=type(e).__name__,
                    elapsed_ms=round(elapsed * 1000, 1),
                    reason=str(e),
                )
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid input: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )
            except Exception as e:
                elapsed = time.time() - start_time
                logger.exception(
                    "handler_failed",
                    extra=build_log_extra(
                        function=func.__name__,
                        error_message=error_message,
                        error_type=type(e).__name__,
                        elapsed_ms=round(elapsed * 1000, 1),
                        reason=str(e),
                    ),
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"{error_message}: {str(e)}",
                    headers={"X-Request-ID": ensure_request_id()}
                )

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator
