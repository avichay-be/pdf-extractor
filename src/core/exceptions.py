import logging

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.core.logging_utils import log_event

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Custom handler for HTTP exceptions to ensure consistent JSON response.
    """
    level = logging.ERROR if exc.status_code >= 500 else logging.WARNING
    log_event(
        logger,
        level,
        "http_exception",
        method=request.method,
        path=str(request.url.path),
        status_code=exc.status_code,
        error_type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom handler for validation errors.
    """
    error_fields = []
    for error in exc.errors():
        location = error.get("loc", ())
        if len(location) > 1:
            error_fields.append(".".join(str(part) for part in location[1:]))
        elif location:
            error_fields.append(str(location[0]))

    log_event(
        logger,
        logging.WARNING,
        "request_validation_failed",
        method=request.method,
        path=str(request.url.path),
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_count=len(exc.errors()),
        error_fields=error_fields[:10],
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )
