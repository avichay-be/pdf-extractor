"""
Example client script for testing the Michman PDF Extractor API.
"""
import sys
from pathlib import Path
import httpx


def extract_pdf(pdf_path: str, output_path: str = None, api_url: str = "http://localhost:8000", query: str = None):
    """
    Extract content from a PDF file using the API.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path for output markdown (default: same name with .md)
        api_url: API base URL
        query: Filter outline sections by name (default: "דוחות כספיים")
                Examples: "דוחות כספיים", "דוח דירקטוריון", "financial", etc.
    """
    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"Error: File not found: {pdf_path}")
        return

    if not pdf_file.suffix.lower() == '.pdf':
        print(f"Error: File must be a PDF: {pdf_path}")
        return

    # Set output path
    if output_path is None:
        output_path = pdf_file.with_suffix('.md')
    else:
        output_path = Path(output_path)

    print(f"Processing PDF: {pdf_path}")
    print(f"API URL: {api_url}/extract")
    print(f"Query filter: {query if query else 'דוחות כספיים (default)'}")

    try:
        # Prepare form data
        files = {"file": (pdf_file.name, open(pdf_path, 'rb'), "application/pdf")}
        data = {}
        if query is not None:
            data["query"] = query

        # Send request
        response = httpx.post(
            f"{api_url}/extract",
            files=files,
            data=data,
            timeout=300.0  # 5 minutes timeout
        )

        # Close the file
        files["file"][1].close()

        response.raise_for_status()

        # Save result
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"✓ Success! Markdown saved to: {output_path}")
        print(f"  Size: {len(response.text)} characters")

    except httpx.HTTPStatusError as e:
        print(f"✗ HTTP Error {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"✗ Request Error: {e}")
    except Exception as e:
        print(f"✗ Error: {e}")


def check_health(api_url: str = "http://localhost:8000"):
    """
    Check API health status.

    Args:
        api_url: API base URL
    """
    try:
        response = httpx.get(f"{api_url}/health", timeout=10.0)
        response.raise_for_status()

        data = response.json()
        print("API Health Check:")
        print(f"  Status: {data.get('status', 'unknown')}")
        print(f"  Mistral Configured: {data.get('mistral_configured', False)}")

    except Exception as e:
        print(f"✗ Health check failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python example_client.py <pdf_file> [output_file] [query]")
        print("  python example_client.py --health")
        print()
        print("Examples:")
        print("  python example_client.py document.pdf")
        print("  python example_client.py document.pdf output.md")
        print("  python example_client.py document.pdf output.md 'דוח דירקטוריון'")
        print("  python example_client.py document.pdf output.md 'financial'")
        print("  python example_client.py --health")
        print()
        print("Query examples:")
        print("  - 'דוחות כספיים' (Financial Reports) - default")
        print("  - 'דוח דירקטוריון' (Directors' Report)")
        print("  - 'financial' (case-insensitive partial match)")
        print("  - '' (empty string returns all sections)")
        sys.exit(1)

    if sys.argv[1] == "--health":
        check_health()
    else:
        pdf_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        query = sys.argv[3] if len(sys.argv) > 3 else None
        extract_pdf(pdf_path, output_path, query=query)
