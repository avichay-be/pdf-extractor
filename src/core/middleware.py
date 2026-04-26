"""
HTTP middleware utilities.

Adds request ID propagation for structured logging and response headers.
"""
import uuid
from typing import Callable

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.error_handling import get_request_id, request_id_var


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach/propagate request IDs for each incoming request."""

    async def dispatch(self, request: Request, call_next: Callable):
        # Prefer incoming header if present, otherwise generate a new one
        incoming = request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID")
        request_id = incoming or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            final_request_id = getattr(request.state, "request_id", None) or get_request_id() or request_id
            response.headers["X-Request-ID"] = final_request_id
            return response
        finally:
            request_id_var.reset(token)
