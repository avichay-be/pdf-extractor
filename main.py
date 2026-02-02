"""
FastAPI application for processing PDFs with Mistral Document AI.
Splits PDFs by main outlines and combines results into markdown.
"""
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging

from src.api.routes import health, extraction
from src.core.logging import setup_logging
from src.core.exceptions import http_exception_handler, validation_exception_handler
from src.core.middleware import RequestIDMiddleware

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Michman PDF Extractor",
    description="API for extracting content from PDFs using Mistral Document AI",
    version="1.0.0"
)

# Middleware
app.add_middleware(RequestIDMiddleware)

# Add exception handlers
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Include routers
app.include_router(health.router)
app.include_router(extraction.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
