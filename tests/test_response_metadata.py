import base64
import io
import json
import uuid
import zipfile

import httpx
import pytest

from main import app
from src.api.routes import extraction
from src.core.config import settings
from src.models.workflow_models import ExtractedSection, WorkflowResult
from src.services.response_builder import ResponseBuilder


class _DummyService:
    def __init__(self):
        self.calls = []

    async def extract_document(
        self,
        pdf_path: str,
        enable_validation=None,
        enable_polishing=None,
    ):
        self.calls.append(
            {
                "pdf_path": pdf_path,
                "enable_validation": enable_validation,
                "enable_polishing": enable_polishing,
            }
        )
        return WorkflowResult(
            content="# Hello\n\nBody",
            metadata={
                "model": "pdf-extractor-v2",
                "workflow": "ocr",
                "ocr_mode": "all_around",
                "routing_strategy": "page_aware",
                "total_pages": 2,
                "page_types": {"text_only": 1, "image_only": 1},
                "page_routes": [
                    {
                        "page_number": 1,
                        "page_range": [1, 1],
                        "strategy": "text_extraction",
                        "source": "pdfplumber",
                        "confidence": 1.0,
                    },
                    {
                        "page_number": 2,
                        "page_range": [2, 2],
                        "strategy": "ocr",
                        "source": "mistral",
                        "confidence": 0.98,
                        "ocr_mode": "all_around",
                        "validation_source": "text_extraction",
                    },
                ],
                "timing": {"analysis_s": 0.1, "total_s": 0.5},
            },
            sections=[],
            validation_report={
                "enabled": True,
                "mode": "mixed_page_cross_validation",
                "status": "passed",
                "dual_source_pages": 1,
                "total_results": 2,
                "source_distribution": {
                    "pdfplumber": 1,
                    "mistral": 1,
                },
            },
        )


def _parse_metadata_footer(markdown: str) -> dict:
    assert "##### Metadata\n" in markdown
    _, metadata_json = markdown.rsplit("##### Metadata\n", 1)
    return json.loads(metadata_json)


@pytest.fixture
async def client(monkeypatch):
    prev_require_api_key = settings.REQUIRE_API_KEY
    settings.REQUIRE_API_KEY = False
    dummy_service = _DummyService()
    monkeypatch.setattr(
        extraction,
        "build_page_aware_extraction_service",
        lambda: dummy_service,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as test_client:
        yield test_client, dummy_service

    settings.REQUIRE_API_KEY = prev_require_api_key


@pytest.mark.asyncio
async def test_extract_json_uses_body_request_id(client):
    client, dummy_service = client
    sample_pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
    response = await client.post(
        "/extract-json",
        json={
            "filename": "sample.pdf",
            "file_content": base64.b64encode(sample_pdf).decode("ascii"),
            "request_id": "body-id",
            "enable_cross_validation": False,
            "enable_polishing": False,
        },
        headers={"X-Request-ID": "header-id"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "body-id"
    assert response.headers["X-Request-ID"] == "body-id"
    assert payload["metadata"]["workflow"] == "ocr"
    assert payload["metadata"]["page_routes"][1]["ocr_mode"] == "all_around"
    assert payload["metadata"]["validation_summary"]["status"] == "passed"
    assert dummy_service.calls[0]["enable_validation"] is False
    assert dummy_service.calls[0]["enable_polishing"] is False


@pytest.mark.asyncio
async def test_extract_json_generates_request_id_when_missing(client):
    client, _ = client
    sample_pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
    response = await client.post(
        "/extract-json",
        json={
            "filename": "sample.pdf",
            "file_content": base64.b64encode(sample_pdf).decode("ascii"),
            "request_id": None,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    uuid.UUID(payload["request_id"])
    assert response.headers["X-Request-ID"] == payload["request_id"]


@pytest.mark.asyncio
async def test_extract_download_ignores_incoming_request_id_and_appends_metadata(client):
    client, dummy_service = client
    sample_pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
    response = await client.post(
        "/extract?enable_cross_validation=false&enable_polishing=false",
        files={"file": ("sample.pdf", sample_pdf, "application/pdf")},
        headers={"X-Request-ID": "header-id"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "header-id"
    metadata = _parse_metadata_footer(response.text)
    uuid.UUID(response.headers["X-Request-ID"])
    assert metadata["request_id"] == response.headers["X-Request-ID"]
    assert metadata["workflow"] == "ocr"
    assert metadata["validation_summary"]["mode"] == "mixed_page_cross_validation"
    assert dummy_service.calls[0]["enable_validation"] is False
    assert dummy_service.calls[0]["enable_polishing"] is False


@pytest.mark.asyncio
async def test_response_builder_appends_metadata_to_zip_sections():
    builder = ResponseBuilder()
    result = WorkflowResult(
        content="",
        metadata={
            "workflow": "ocr",
            "ocr_mode": "all_around",
            "routing_strategy": "page_aware",
            "total_pages": 2,
            "page_routes": [
                {
                    "page_number": 1,
                    "page_range": [1, 1],
                    "strategy": "ocr_all_around",
                    "source": "mistral",
                    "confidence": 0.99,
                    "ocr_mode": "all_around",
                }
            ],
            "timing": {"total_s": 1.2},
        },
        sections=[
            ExtractedSection(
                filename="section1.md",
                content="# Section 1",
                title="Section 1",
                page_range=(1, 2),
            )
        ],
        validation_report={
            "enabled": True,
            "status": "passed",
            "mode": "full",
            "dual_source_pages": 0,
            "total_results": 2,
            "source_distribution": {"mistral": 2},
        },
    )

    response = builder.build_download_response(
        result=result,
        original_filename="sample.pdf",
        request_id="zip-id",
    )

    body = b""
    async for chunk in response.body_iterator:
        body += chunk

    with zipfile.ZipFile(io.BytesIO(body), "r") as zip_file:
        names = zip_file.namelist()
        assert names == ["section1.md"]
        content = zip_file.read("section1.md").decode("utf-8")

    metadata = _parse_metadata_footer(content)
    assert metadata["request_id"] == "zip-id"
    assert metadata["section_filename"] == "section1.md"
    assert metadata["section_title"] == "Section 1"
    assert metadata["section_page_range"] == [1, 2]
