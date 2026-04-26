import base64
import pytest
from starlette.requests import Request

from src.api.routes import extraction
from src.core.config import settings
from src.models.api_models import Base64FileRequest
from src.models.workflow_models import WorkflowResult


class _DummyService:
    async def extract_document(
        self,
        pdf_path: str,
        enable_validation=None,
        enable_polishing=None,
    ):
        return WorkflowResult(
            content="# Hello\n\nBody",
            metadata={
                "model": "pdf-extractor-v2",
                "workflow": "ocr",
                "ocr_mode": "all_around",
                "routing_strategy": "page_aware",
                "page_routes": [
                    {
                        "page_number": 1,
                        "page_range": [1, 1],
                        "strategy": "ocr",
                        "source": "mistral",
                        "confidence": 1.0,
                        "ocr_mode": "all_around",
                    }
                ],
            },
            sections=[],
            validation_report={
                "enabled": True,
                "mode": "mixed_page_cross_validation",
                "status": "passed",
                "dual_source_pages": 1,
                "total_results": 1,
                "source_distribution": {"mistral": 1},
            },
        )


@pytest.mark.asyncio
async def test_extract_json_smoke(monkeypatch):
    # Relax auth for test
    prev_require_api_key = settings.REQUIRE_API_KEY
    settings.REQUIRE_API_KEY = False
    try:
        # Stub extraction service to avoid external calls
        monkeypatch.setattr(
            extraction,
            "build_page_aware_extraction_service",
            lambda: _DummyService(),
        )

        # Minimal valid PDF bytes
        sample_pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
        payload = {
            "filename": "sample.pdf",
            "file_content": base64.b64encode(sample_pdf).decode("ascii"),
            "request_id": "body-id",
            "enable_cross_validation": False,
            "enable_polishing": False,
        }

        http_request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/extract-json",
                "headers": [],
                "query_string": b"",
                "scheme": "http",
                "server": ("testserver", 80),
                "client": ("testclient", 123),
                "root_path": "",
                "http_version": "1.1",
            }
        )

        response = await extraction.extract_pdf_from_base64(
            http_request,
            Base64FileRequest(**payload)
        )

        assert response.request_id == "body-id"
        assert response.file_name == "sample.pdf"
        assert response.metadata.workflow == "ocr"
        assert response.metadata.page_routes[0].ocr_mode == "all_around"
        assert response.metadata.validation_summary.status == "passed"
        assert response.extracted_content[0].filename.endswith(".md")
        assert "Hello" in response.extracted_content[0].content
    finally:
        settings.REQUIRE_API_KEY = prev_require_api_key
