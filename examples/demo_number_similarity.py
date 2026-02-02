"""
Demonstration of number-frequency based similarity calculation.

This script shows how the new similarity method works for financial documents.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.validation import ValidationService
from config import settings

def print_comparison(content1: str, content2: str, description: str):
    """Print a comparison between two contents."""
    print(f"\n{'='*80}")
    print(f"Test: {description}")
    print(f"{'='*80}")
    print(f"\nContent 1:")
    print(f"{content1}")
    print(f"\nContent 2:")
    print(f"{content2}")

    validator = ValidationService()

    # Extract numbers
    numbers1 = validator._extract_numbers(content1)
    numbers2 = validator._extract_numbers(content2)

    print(f"\nNumbers from Content 1: {numbers1}")
    print(f"Numbers from Content 2: {numbers2}")

    # Calculate similarity
    similarity = validator.calculate_similarity_number_frequency(content1, content2)

    print(f"\n✓ Similarity: {similarity:.2%}")

    if similarity >= 0.95:
        print("✅ PASS - Contents match!")
    elif similarity >= 0.70:
        print("⚠️  PARTIAL - Some differences detected")
    else:
        print("❌ FAIL - Contents are different")


def main():
    """Run demonstration scenarios."""
    print("\n" + "="*80)
    print("NUMBER-FREQUENCY SIMILARITY DEMONSTRATION")
    print("For Financial PDF Validation")
    print("="*80)

    # Scenario 1: Same numbers, different languages
    print_comparison(
        "הכנסה: 1,234,567 ש״ח בשנת 2024",  # Hebrew
        "Revenue: 1234567 ILS in 2024",     # English
        "Same numbers, different languages (Hebrew vs English)"
    )

    # Scenario 2: Same numbers, different formatting
    print_comparison(
        "Total: 1,234,567.89 for year 2024",
        "Sum: 1.234.567,89 in 2024",  # European format
        "Same numbers, different formatting (US vs European)"
    )

    # Scenario 3: OCR error in number
    print_comparison(
        "Assets: 1,234,567 and Debt: 500,000",
        "Assets: 1,234,557 and Debt: 500,000",  # 567 → 557 (OCR error)
        "OCR error in number (should detect difference)"
    )

    # Scenario 4: Missing number
    print_comparison(
        "Q1: 100, Q2: 200, Q3: 300, Q4: 400",
        "Q1: 100, Q2: 200, Q4: 400",  # Missing Q3
        "Missing number (should detect partial match)"
    )

    # Scenario 5: Same numbers, repeated differently
    print_comparison(
        "Revenue: 1000, Expenses: 1000, Profit: 1000",  # 1000 appears 3 times
        "Total: 1000",  # 1000 appears 1 time
        "Same number with different frequencies"
    )

    # Scenario 6: Only text, no numbers
    print_comparison(
        "This is just text with no numerical data",
        "Another text without any digits at all",
        "No numbers in either content (should match as both empty)"
    )

    print(f"\n{'='*80}")
    print("CONFIGURATION:")
    print(f"{'='*80}")
    print(f"Similarity Method: {settings.VALIDATION_SIMILARITY_METHOD}")
    print(f"Similarity Threshold: {settings.VALIDATION_SIMILARITY_THRESHOLD:.2%}")
    print(f"Validator Provider: {settings.VALIDATION_PROVIDER}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
