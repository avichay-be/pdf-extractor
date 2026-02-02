"""
Example client for testing the /extract-json endpoint.
"""
import requests
import base64
import json
import sys
from pathlib import Path


def extract_pdf_json(pdf_path: str, api_url: str = "http://localhost:8000"):
    """
    Extract PDF content using the /extract-json endpoint with base64 encoding.

    Args:
        pdf_path: Path to the PDF file to extract
        api_url: Base URL of the API (default: http://localhost:8000)

    Returns:
        dict: Response with filename and content
    """
    # Read PDF file and encode to base64
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

    # Get filename
    filename = Path(pdf_path).name

    # Prepare request payload
    payload = {
        "filename": filename,
        "file_content": pdf_base64
    }

    print(f"📄 Extracting content from: {filename}")
    print(f"📦 Base64 size: {len(pdf_base64)} characters")
    print(f"🔄 Sending to {api_url}/extract-json...")

    try:
        # Send POST request
        response = requests.post(
            f"{api_url}/extract-json",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"📝 Filename: {result['filename']}")
            print(f"📊 Content length: {len(result['content'])} characters")
            print(f"\n--- First 500 characters of content ---")
            print(result['content'][:500])
            print("...")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None

    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return None


def save_result(result: dict, output_path: str = None):
    """
    Save the extraction result to files.

    Args:
        result: Result dictionary from the API
        output_path: Optional path for markdown file
    """
    if not result:
        print("❌ No result to save")
        return

    # Save markdown content
    if not output_path:
        output_path = Path(result['filename']).stem + "_extracted.md"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['content'])
    print(f"💾 Saved markdown to: {output_path}")

    # Save JSON response
    json_path = Path(output_path).stem + "_response.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved JSON response to: {json_path}")


def main():
    """Main function to run the example client."""
    if len(sys.argv) < 2:
        print("Usage: python example_json_client.py <pdf_file> [output_file]")
        print("\nExample:")
        print("  python example_json_client.py document.pdf")
        print("  python example_json_client.py document.pdf output.md")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        sys.exit(1)

    # Extract PDF
    result = extract_pdf_json(pdf_path)

    # Save result
    if result:
        save_result(result, output_path)


if __name__ == "__main__":
    main()
