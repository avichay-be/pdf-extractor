"""
Test bank_statements.pdf with Azure DI to see numerical validation logs.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.extraction_service import process_azure_document_intelligence


async def test_bank_statements():
    """Test bank statements PDF with numerical validation."""
    pdf_path = "data/bank_statements.pdf"

    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return

    print("=" * 80)
    print("TESTING BANK STATEMENTS WITH NUMERICAL VALIDATION")
    print("=" * 80)
    print(f"\nProcessing: {pdf_path}")
    print("-" * 80)

    try:
        markdown_content, metadata = await process_azure_document_intelligence(
            pdf_path=pdf_path,
            query="bank statements"
        )

        print(f"\n✅ SUCCESS!")
        print("-" * 80)
        print(f"Workflow: {metadata.get('workflow')}")
        print(f"Method: {metadata.get('extraction_method')}")
        print(f"Table Count: {metadata.get('table_count', 'N/A')}")
        print(f"Merged: {metadata.get('merged', 'N/A')}")
        print("-" * 80)

        # Count tables in output
        lines = markdown_content.split('\n')
        table_headers = [line for line in lines if line.startswith("**Table from")]

        print(f"\n📊 RESULTS:")
        print(f"   Total tables in output: {len(table_headers)}")
        for header in table_headers:
            print(f"   - {header}")

        # Show sample
        print(f"\n📝 SAMPLE OUTPUT (first 30 lines):")
        print("-" * 80)
        for i, line in enumerate(lines[:30]):
            print(line)
        print("-" * 80)

        print(f"\n✅ Test completed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_bank_statements())
