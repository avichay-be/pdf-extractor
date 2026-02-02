"""
Test rate limiting with multiple chunks.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.client_factory import get_client_factory
from src.services.pdf_processor import PDFProcessor


async def test_rate_limiting():
    """Test rate limiting with multiple quick requests."""
    print("=" * 80)
    print("TESTING RATE LIMITING")
    print("=" * 80)
    print("\nThis test sends 3 parallel requests to verify rate limiting")
    print("Expected behavior: Requests should be spaced at least 1 second apart")
    print("-" * 80)

    # Use a simple test PDF
    pdf_path = "data/bank_statements.pdf"

    if not Path(pdf_path).exists():
        print(f"❌ Test PDF not found: {pdf_path}")
        return

    # Send the same request 3 times in parallel to test rate limiting
    num_requests = 3
    print(f"\nSending {num_requests} parallel requests with same PDF...\n")

    # Measure total time
    start_time = time.time()

    # Get client
    mistral_client = get_client_factory().mistral_client

    # Process same PDF multiple times in parallel
    tasks = [
        mistral_client.process_document(pdf_path=pdf_path)
        for _ in range(num_requests)
    ]

    print(f"Starting {len(tasks)} parallel requests at {time.strftime('%H:%M:%S')}...\n")

    try:
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        total_time = end_time - start_time

        print("\n" + "-" * 80)
        print(f"✅ All {len(results)} requests completed successfully!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per request: {total_time/len(results):.2f} seconds")
        print("-" * 80)

        # Expected: ~1 second spacing between requests due to rate limiting
        # With 3 requests: first starts immediately, second waits 1s, third waits 2s
        expected_min_time = (len(results) - 1) * 1.0  # Rate limit delay

        if total_time >= expected_min_time:
            print(f"\n✅ Rate limiting is working!")
            print(f"   Expected minimum spacing: {expected_min_time:.1f}s")
            print(f"   Actual time: {total_time:.1f}s")
            print(f"   Rate limit overhead: {total_time - expected_min_time:.1f}s (API processing time)")
        else:
            print(f"\n⚠️  Rate limiting may not be working correctly")
            print(f"   Expected minimum spacing: {expected_min_time:.1f}s")
            print(f"   Actual time: {total_time:.1f}s")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_rate_limiting())
