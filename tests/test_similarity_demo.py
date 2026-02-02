#!/usr/bin/env python3
"""
Demonstration of the improved similarity calculation.
Shows how the alphanumeric-only comparison ignores formatting differences.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.validation import ValidationService


def print_comparison(title: str, text1: str, text2: str, service: ValidationService):
    """Print a comparison between two texts."""
    print(f"\n{'='*80}")
    print(f"TEST: {title}")
    print(f"{'='*80}")

    print(f"\nText 1 ({len(text1)} chars):")
    print(f"  {repr(text1[:100])}")

    print(f"\nText 2 ({len(text2)} chars):")
    print(f"  {repr(text2[:100])}")

    norm1 = service._normalize_for_comparison(text1)
    norm2 = service._normalize_for_comparison(text2)

    print(f"\nNormalized 1 ({len(norm1)} alphanumeric):")
    print(f"  {repr(norm1[:100])}")

    print(f"\nNormalized 2 ({len(norm2)} alphanumeric):")
    print(f"  {repr(norm2[:100])}")

    similarity = service.calculate_similarity(text1, text2)
    print(f"\nSimilarity Score: {similarity:.2%}")

    if similarity >= 0.95:
        print("✅ PASS - Content matches (≥95% threshold)")
    else:
        print("❌ FAIL - Content differs (<95% threshold)")


def main():
    """Run similarity comparison demonstrations."""
    service = ValidationService(openai_client=None)

    print("\n" + "="*80)
    print("ALPHANUMERIC-ONLY SIMILARITY DEMONSTRATION")
    print("="*80)
    print("\nThis demonstrates how the validation service compares content")
    print("by ignoring formatting, punctuation, and whitespace.")

    # Test 1: Same content, different table formatting
    print_comparison(
        "Same Content, Different Table Formatting",
        "Name Test Value 100 Year 2024",
        "| Name | Test | Value | 100 | Year | 2024 |",
        service
    )

    # Test 2: Hebrew content with different formatting
    print_comparison(
        "Hebrew Content with Different Formatting",
        "סלע קפיטל נדלן דוחות כספיים 2024",
        "סלע | קפיטל | נדל\"ן | דוחות כספיים | 2024",
        service
    )

    # Test 3: Whitespace differences
    print_comparison(
        "Whitespace Differences Ignored",
        "HelloWorld123",
        "Hello   World   1 2 3",
        service
    )

    # Test 4: Complex table with formatting
    table1 = """
סלע קפיטל נדלן בעמ
דוחות מתמצתים מאוחדים
ליום 30 בספטמבר 2025 אלפי שח
"""

    table2 = """
| סלע קפיטל נדל"ן בע"מ |
|---------------------|
| דוחות מתמצתים מאוחדים |
| ליום 30 בספטמבר 2025 |
| אלפי ש"ח |
"""

    print_comparison(
        "Complex Table vs Plain Text (Hebrew)",
        table1,
        table2,
        service
    )

    # Test 5: Actually different content
    print_comparison(
        "Different Content (Should Fail)",
        "Financial report for Q1 2024",
        "Technical specifications for product X",
        service
    )

    # Test 6: Only formatting characters
    print_comparison(
        "Both Only Punctuation (Should Match)",
        "| | | --- |",
        "--- | --- | ---",
        service
    )

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
