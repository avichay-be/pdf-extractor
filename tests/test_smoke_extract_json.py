import base64
from fastapi.testclient import TestClient

from main import app
from src.api.routes import extraction
from src.core.config import settings
from src.models.workflow_models import WorkflowResult


class _DummyOrchestrator:
    async def execute_workflow(self, pdf_path: str, query: str, enable_validation=None):
        return WorkflowResult(
            content="# Hello\n\nBody",
            metadata={"model": "pdf-extractor-v2", "workflow": "mistral"},
            sections=[],
            validation_report=None,
        )


def test_extract_json_smoke(monkeypatch):
    # Relax auth for test
    prev_require_api_key = settings.REQUIRE_API_KEY
    settings.REQUIRE_API_KEY = False

    # Stub orchestrator to avoid external calls
    monkeypatch.setattr(extraction, "get_workflow_orchestrator", lambda: _DummyOrchestrator())

    # Minimal valid PDF bytes
    sample_pdf = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF"
    payload = {
        "filename": "sample.pdf",
        "file_content": base64.b64encode(sample_pdf).decode("ascii"),
        "query": "01_Fin_Reports",
    }

    with TestClient(app) as client:
        response = client.post("/extract-json", json=payload)

    settings.REQUIRE_API_KEY = prev_require_api_key

    assert response.status_code == 200
    data = response.json()
    assert data["file_name"] == "sample.pdf"
    assert data["extracted_content"][0]["filename"].endswith(".md")
    assert "Hello" in data["extracted_content"][0]["content"]
