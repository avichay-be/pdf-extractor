"""
Test script to verify text_extraction and azure_document_intelligence workflows.
"""
import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workflows.workflow_router import (
    get_workflow_for_query,
    is_text_extraction_query,
    is_azure_document_intelligence_query
)
from src.services.extraction_service import (
    process_text_extraction,
    process_azure_document_intelligence
)


from src.workflows.workflow_types import WorkflowType

def test_workflow_routing():
    """Test that queries route to the correct workflows."""
    print("=" * 80)
    print("Testing Workflow Routing")
    print("=" * 80)

    test_cases = [
        ("05_Esna", WorkflowType.AZURE_DOCUMENT_INTELLIGENCE),
        ("04_Bank_Statements", WorkflowType.TEXT_EXTRACTION),
        ("01_Fin_Reports", WorkflowType.MISTRAL),
        ("random query", WorkflowType.MISTRAL),  # Should default to mistral
    ]

    all_passed = True
    for query, expected_workflow in test_cases:
        actual_workflow = get_workflow_for_query(query)
        status = "✓" if actual_workflow == expected_workflow else "✗"
        if actual_workflow != expected_workflow:
            all_passed = False
        print(f"{status} Query: '{query}' → {actual_workflow} (expected: {expected_workflow})")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All routing tests passed!")
    else:
        print("✗ Some routing tests failed!")
    print("=" * 80)

    return all_passed


async def test_text_extraction_workflow():
    """Test text extraction workflow with sample PDF."""
    print("\n" + "=" * 80)
    print("Testing Text Extraction Workflow (Camelot)")
    print("=" * 80)

    # Path to test PDF
    pdf_path = str(Path(__file__).parent.parent / "data/bank_statements.pdf")

    if not Path(pdf_path).exists():
        print(f"✗ Test PDF not found: {pdf_path}")
        print("  Skipping text extraction test")
        return False

    try:
        print(f"Processing: {pdf_path}")
        markdown_content, metadata = await process_text_extraction(
            pdf_path=pdf_path,
            query="esna"
        )

        print(f"\n✓ Text extraction successful!")
        print(f"  Workflow: {metadata.get('workflow')}")
        print(f"  Method: {metadata.get('extraction_method')}")
        print(f"  Tables Found: {metadata.get('tables_found')}")
        print(f"  Content Length: {len(markdown_content)} characters")
        print(f"\nFirst 500 characters:")
        print("-" * 80)
        print(markdown_content[:500])
        print("-" * 80)

        return True

    except Exception as e:
        print(f"✗ Text extraction failed: {e}")
        return False


async def test_azure_di_workflow():
    """Test Azure Document Intelligence workflow."""
    print("\n" + "=" * 80)
    print("Testing Azure Document Intelligence Workflow")
    print("=" * 80)

    # Path to test PDF
    pdf_path = str(Path(__file__).parent.parent / "data/bank_statements.pdf")

    if not Path(pdf_path).exists():
        print(f"✗ Test PDF not found: {pdf_path}")
        print("  Skipping Azure DI test")
        return False

    try:
        print(f"Processing: {pdf_path}")
        print("  (This may take longer as it calls Azure API...)")

        markdown_content, metadata = await process_azure_document_intelligence(
            pdf_path=pdf_path,
            query="דפי בנק"
        )

        print(f"\n✓ Azure DI extraction successful!")
        print(f"  Workflow: {metadata.get('workflow')}")
        print(f"  Method: {metadata.get('extraction_method')}")
        print(f"  Table Count: {metadata.get('table_count', 0)}")
        print(f"  Content Length: {len(markdown_content)} characters")
        print(f"\nFirst 500 characters:")
        print("-" * 80)
        print(markdown_content[:500])
        print("-" * 80)

        return True

    except Exception as e:
        print(f"✗ Azure DI extraction failed: {e}")
        print(f"  This is expected if Azure DI credentials are not configured")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("WORKFLOW TESTS")
    print("=" * 80 + "\n")

    # Test 1: Routing
    routing_passed = test_workflow_routing()

    # Test 2: Text Extraction
    text_extraction_passed = await test_text_extraction_workflow()

    # Test 3: Azure DI (may fail if not configured)
    azure_di_passed = await test_azure_di_workflow()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"{'✓' if routing_passed else '✗'} Workflow Routing: {'PASSED' if routing_passed else 'FAILED'}")
    print(f"{'✓' if text_extraction_passed else '✗'} Text Extraction: {'PASSED' if text_extraction_passed else 'FAILED'}")
    print(f"{'✓' if azure_di_passed else '⚠'} Azure DI: {'PASSED' if azure_di_passed else 'FAILED/NOT CONFIGURED'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
