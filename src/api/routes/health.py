import os
import tempfile
from fastapi import APIRouter
from src.core.config import settings

router = APIRouter()

@router.get("/")
async def root():
    """Basic health check endpoint."""
    return {"message": "Michman PDF Extractor API", "status": "healthy"}

@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.

    Verifies:
    - Configuration for active engines (Mistral, Gemini)
    - File system write permissions
    """
    health_status = {
        "status": "healthy",
        "service": "Michman PDF Extractor",
        "version": "2.0",
    }

    # Configuration checks
    config_checks = {
        "mistral_configured": bool(settings.AZURE_API_KEY and settings.MISTRAL_API_URL),
        "gemini_configured": bool(settings.GEMINI_API_KEY),
        "validation_enabled": settings.ENABLE_CROSS_VALIDATION,
        "validation_provider": "gemini" if settings.ENABLE_CROSS_VALIDATION else None,
        "ocr_engine": "mistral",
        "routing_strategy": "page_aware",
    }
    health_status.update(config_checks)

    # File system check
    try:
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, ".health_check_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        health_status["filesystem_writable"] = True
        health_status["temp_directory"] = temp_dir
    except Exception as e:
        health_status["filesystem_writable"] = False
        health_status["filesystem_error"] = str(e)
        health_status["status"] = "degraded"

    # Overall status
    if not config_checks["mistral_configured"]:
        health_status["status"] = "degraded"
        health_status["warning"] = "Mistral (primary provider) not configured"

    return health_status
