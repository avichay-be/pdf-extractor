"""
Quick test script for bank statement text extraction.
"""
import sys
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.extraction_service import extract_text_from_pdf

async def test_extraction():
    """Test text extraction from bank statement PDF."""
    # Path to bank statement PDF
    pdf_path = "data/bank_statements.pdf"

    if not Path(pdf_path).exists():
        print(f"Skipping test: PDF not found at {pdf_path}")
        return

    print(f"Testing text extraction from: {pdf_path}")
    print("=" * 80)

    # Extract text
    markdown_content, metadata = await asyncio.to_thread(
        extract_text_from_pdf,
        pdf_path
    )

    print(f"\nExtraction successful!")
    print(f"Method: {metadata.get('extraction_method')}")
    print(f"Total Pages: {metadata.get('total_pages')}")
    print(f"Library: {metadata.get('library')}")
    print("=" * 80)

    # Save to file
    output_file = "/tmp/bank_statement_test.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"\nSaved extracted content to: {output_file}")

    # Show first 50 lines
    lines = markdown_content.split('\n')
    print("\n=== First 50 lines of extracted content ===\n")
    print('\n'.join(lines[:50]))

    print(f"\n\nTotal lines extracted: {len(lines)}")

if __name__ == "__main__":
    asyncio.run(test_extraction())
