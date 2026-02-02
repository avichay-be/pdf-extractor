"""
Integration tests for API endpoints.

Tests the complete PDF processing pipeline by sending files to the API
and validating the responses.
"""
import pytest
import httpx
from pathlib import Path
import base64
import zipfile
import io
import time
from typing import List, Dict, Any
import json


# Configuration
TEST_PDFS_DIR = Path(__file__).parent.parent.parent / "tests" / "test_pdfs"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "tests" / "integration_output"
TIMEOUT = 300.0  # 5 minutes for large PDFs
from fastapi.testclient import TestClient
from main import app

@pytest.fixture(scope="module")
def api_client():
    """Create TestClient for API calls."""
    from src.core.config import settings
    settings.API_KEY = "test-key"
    settings.REQUIRE_API_KEY = True
    client = TestClient(app)
    client.headers["X-API-Key"] = "test-key"
    return client


@pytest.fixture(scope="module")
def async_api_client():
    """Create async HTTP client for API calls."""
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=TIMEOUT)


@pytest.fixture(scope="module", autouse=True)
def setup_directories():
    """Set up test directories before tests run."""
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create test PDFs directory if it doesn't exist
    TEST_PDFS_DIR.mkdir(parents=True, exist_ok=True)

    yield

    # Cleanup can be added here if needed
    print(f"\nTest outputs saved to: {OUTPUT_DIR}")


@pytest.fixture(scope="module")
def test_pdf_files() -> List[Path]:
    """
    Get all PDF files from test directory.

    Returns:
        List of PDF file paths
    """
    pdf_files = list(TEST_PDFS_DIR.glob("*.pdf"))

    if not pdf_files:
        pytest.skip(f"No PDF files found in {TEST_PDFS_DIR}. Add PDFs to run integration tests.")

    return pdf_files


@pytest.fixture
def test_metadata() -> Dict[str, Any]:
    """Store test metadata for reporting."""
    return {
        "start_time": time.time(),
        "results": []
    }


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root_endpoint(self, api_client):
        """Test GET / endpoint."""
        response = api_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_endpoint(self, api_client):
        """Test GET /health endpoint."""
        response = api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "mistral_configured" in data
        assert data["status"] == "healthy"


class TestExtractEndpoint:
    """Test POST /extract endpoint with file uploads."""

    def test_extract_all_pdfs(self, api_client, test_pdf_files, test_metadata):
        """
        Test /extract endpoint with all PDF files from test directory.

        Sends each PDF to the API and saves the output.
        """
        results = []

        for pdf_path in test_pdf_files:
            print(f"\n{'='*80}")
            print(f"Testing: {pdf_path.name}")
            print(f"{'='*80}")

            start_time = time.time()

            try:
                # Open and send PDF
                with open(pdf_path, "rb") as f:
                    files = {"file": (pdf_path.name, f, "application/pdf")}
                    data = {"query": ""}  # Empty query to get all content

                    response = api_client.post("/extract", files=files, data=data)

                processing_time = time.time() - start_time

                # Check response
                assert response.status_code == 200, f"Failed for {pdf_path.name}: {response.text}"

                # Determine output type and save
                content_type = response.headers.get("content-type", "")
                output_base = OUTPUT_DIR / f"extract_{pdf_path.stem}"

                if "application/zip" in content_type:
                    # Save and extract ZIP
                    zip_path = output_base.with_suffix(".zip")
                    zip_path.write_bytes(response.content)

                    # Extract ZIP
                    extract_dir = output_base
                    extract_dir.mkdir(exist_ok=True)

                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)

                    extracted_files = list(extract_dir.glob("*.md"))

                    result = {
                        "file": pdf_path.name,
                        "status": "success",
                        "type": "zip",
                        "output": str(zip_path),
                        "extracted_files": len(extracted_files),
                        "processing_time": f"{processing_time:.2f}s",
                        "size_bytes": len(response.content)
                    }

                    print(f"✓ Saved ZIP with {len(extracted_files)} sections: {zip_path}")

                else:
                    # Save single markdown file
                    md_path = output_base.with_suffix(".md")
                    md_path.write_text(response.text, encoding="utf-8")

                    result = {
                        "file": pdf_path.name,
                        "status": "success",
                        "type": "markdown",
                        "output": str(md_path),
                        "processing_time": f"{processing_time:.2f}s",
                        "size_bytes": len(response.content)
                    }

                    print(f"✓ Saved markdown: {md_path}")

                results.append(result)

            except Exception as e:
                result = {
                    "file": pdf_path.name,
                    "status": "error",
                    "error": str(e),
                    "processing_time": f"{time.time() - start_time:.2f}s"
                }
                results.append(result)
                print(f"✗ Error: {e}")

        # Save results summary
        self._save_results_summary(results, "extract_endpoint")

        # Assert all succeeded
        failed = [r for r in results if r["status"] == "error"]
        assert len(failed) == 0, f"{len(failed)} files failed: {failed}"

    def test_extract_with_query_filter(self, api_client, test_pdf_files):
        """Test /extract endpoint with query filtering."""
        if not test_pdf_files:
            pytest.skip("No test PDFs available")

        # Test with first PDF
        pdf_path = test_pdf_files[0]

        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            data = {"query": "דוחות כספיים"}  # Hebrew query

            response = api_client.post("/extract", files=files, data=data)

        assert response.status_code == 200

        # Save output
        output_path = OUTPUT_DIR / f"extract_filtered_{pdf_path.stem}.md"

        if "application/zip" in response.headers.get("content-type", ""):
            output_path = output_path.with_suffix(".zip")
            output_path.write_bytes(response.content)
        else:
            output_path.write_text(response.text, encoding="utf-8")

        print(f"Saved filtered output: {output_path}")

    def _save_results_summary(self, results: List[Dict], test_name: str):
        """Save test results summary to JSON file."""
        summary_path = OUTPUT_DIR / f"{test_name}_summary.json"

        summary = {
            "test_name": test_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Results Summary: {test_name}")
        print(f"{'='*80}")
        print(f"Total: {summary['total_files']}")
        print(f"Success: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*80}\n")


class TestExtractJsonEndpoint:
    """Test POST /extract-json endpoint with base64 PDFs."""

    def test_extract_json_all_pdfs(self, api_client, test_pdf_files, test_metadata):
        """
        Test /extract-json endpoint with all PDF files.

        Encodes PDFs to base64 and sends to JSON endpoint.
        """
        results = []

        for pdf_path in test_pdf_files:
            print(f"\n{'='*80}")
            print(f"Testing JSON endpoint: {pdf_path.name}")
            print(f"{'='*80}")

            start_time = time.time()

            try:
                # Read and encode PDF
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

                # Send request
                request_data = {
                    "filename": pdf_path.name,
                    "file_content": pdf_base64,
                    "query": ""  # Empty query for all content
                }

                response = api_client.post(
                    "/extract-json",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )

                processing_time = time.time() - start_time

                # Check response
                assert response.status_code == 200, f"Failed for {pdf_path.name}: {response.text}"

                # Parse response
                data = response.json()

                # Validate response structure
                assert "file_name" in data
                assert "extracted_content" in data
                assert isinstance(data["extracted_content"], list)
                assert len(data["extracted_content"]) > 0

                # Save each section
                output_dir = OUTPUT_DIR / f"json_{pdf_path.stem}"
                output_dir.mkdir(exist_ok=True)

                for section in data["extracted_content"]:
                    section_path = output_dir / section["filename"]
                    section_path.write_text(section["content"], encoding="utf-8")

                # Save full JSON response
                json_path = output_dir / "response.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                result = {
                    "file": pdf_path.name,
                    "status": "success",
                    "sections": len(data["extracted_content"]),
                    "output_dir": str(output_dir),
                    "processing_time": f"{processing_time:.2f}s",
                    "validation_status": data.get("validation", {}).get("status") if data.get("validation") else None
                }

                print(f"✓ Saved {len(data['extracted_content'])} sections to: {output_dir}")

                if data.get("validation"):
                    validation = data["validation"]
                    print(f"  Validation: enabled={validation.get('enabled')}, status={validation.get('status')}")

                results.append(result)

            except Exception as e:
                result = {
                    "file": pdf_path.name,
                    "status": "error",
                    "error": str(e),
                    "processing_time": f"{time.time() - start_time:.2f}s"
                }
                results.append(result)
                print(f"✗ Error: {e}")

        # Save results summary
        self._save_results_summary(results, "extract_json_endpoint")

        # Assert all succeeded
        failed = [r for r in results if r["status"] == "error"]
        assert len(failed) == 0, f"{len(failed)} files failed: {failed}"

    def test_extract_json_with_validation(self, api_client, test_pdf_files):
        """Test /extract-json with cross-validation enabled."""
        if not test_pdf_files:
            pytest.skip("No test PDFs available")

        # Test with first PDF
        pdf_path = test_pdf_files[0]

        # Read and encode PDF
        with open(pdf_path, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode('utf-8')

        # Send request with validation enabled
        request_data = {
            "filename": pdf_path.name,
            "file_content": pdf_base64,
            "query": "",
            "enable_validation": True
        }

        response = api_client.post("/extract-json", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Check validation status
        if "validation" in data and data["validation"]:
            print(f"\nValidation Status:")
            validation = data["validation"]
            print(f"  Enabled: {validation.get('enabled')}")
            print(f"  Status: {validation.get('status')}")
            print(f"  (Detailed metrics are logged by the server)")

    def _save_results_summary(self, results: List[Dict], test_name: str):
        """Save test results summary to JSON file."""
        summary_path = OUTPUT_DIR / f"{test_name}_summary.json"

        summary = {
            "test_name": test_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(results),
            "successful": len([r for r in results if r["status"] == "success"]),
            "failed": len([r for r in results if r["status"] == "error"]),
            "results": results
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Results Summary: {test_name}")
        print(f"{'='*80}")
        print(f"Total: {summary['total_files']}")
        print(f"Success: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'='*80}\n")


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test endpoints with async client for performance testing."""

    async def test_concurrent_requests(self, async_api_client, test_pdf_files):
        """Test multiple concurrent requests to measure performance."""
        if len(test_pdf_files) < 2:
            pytest.skip("Need at least 2 PDFs for concurrent testing")

        import asyncio

        # Take first 2 PDFs for concurrent test
        test_files = test_pdf_files[:2]

        async def send_request(pdf_path: Path):
            """Send a single request."""
            start = time.time()

            with open(pdf_path, "rb") as f:
                files = {"file": (pdf_path.name, f, "application/pdf")}
                data = {"query": ""}

                response = await async_api_client.post("/extract", files=files, data=data)

            elapsed = time.time() - start
            return {
                "file": pdf_path.name,
                "status_code": response.status_code,
                "time": elapsed
            }

        # Send requests concurrently
        print("\nSending concurrent requests...")
        start_time = time.time()

        tasks = [send_request(pdf_path) for pdf_path in test_files]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        print(f"\nConcurrent processing results:")
        for result in results:
            print(f"  {result['file']}: {result['status_code']} ({result['time']:.2f}s)")
        print(f"Total time: {total_time:.2f}s")

        # All should succeed
        assert all(r["status_code"] == 200 for r in results)


if __name__ == "__main__":
    """Run tests directly with pytest."""
    pytest.main([__file__, "-v", "--tb=short"])
