"""
Text normalization utilities for content comparison.
"""
import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class ContentNormalizer:
    """Normalize text content for comparison and extract numbers."""

    def normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text by keeping only alphanumeric characters.
        This filters out formatting, punctuation, and whitespace differences.

        Args:
            text: Text to normalize

        Returns:
            Text containing only alphanumeric characters (lowercase)
        """
        # Keep only alphanumeric characters (including Unicode letters and digits)
        # This works with Hebrew, Arabic, Chinese, etc.
        normalized = ''.join(char.lower() for char in text if char.isalnum())
        return normalized

    def extract_numbers(self, text: str) -> List[str]:
        """
        Extract all numbers from text, normalizing format.

        Handles:
        - Thousands separators: 1,234,567 → 1234567
        - Decimals: 123.45 → 123.45
        - Percentages: 15% → 15
        - Negative numbers: -123 → -123
        - European format: 1.234.567,89 → 1234567.89

        Args:
            text: Text to extract numbers from

        Returns:
            List of normalized number strings
        """
        # Remove currency symbols and common non-numeric characters
        # Keep: digits, decimal points, commas, minus signs, spaces between digits
        cleaned = re.sub(r'[₪$€£¥₹\u20aa]', '', text)  # Remove currency symbols

        # Pattern to match numbers with various formats:
        # - Optional minus sign
        # - Digits with optional thousands separators (comma or period)
        # - Optional decimal part (period or comma as decimal separator)
        # This matches: -1,234.56 or 1.234,56 or 1234 or -123 or 12.5 or 15%
        number_pattern = r'-?\d+(?:[,\.\s]\d{3})*(?:[,\.]\d+)?%?'

        matches = re.findall(number_pattern, cleaned)

        normalized_numbers = []
        for match in matches:
            # Remove percentage sign
            num = match.rstrip('%')

            # Detect if this is European format (1.234,56) vs US format (1,234.56)
            # European: period for thousands, comma for decimal
            # US: comma for thousands, period for decimal

            # Count periods and commas
            period_count = num.count('.')
            comma_count = num.count(',')

            if comma_count > 0 and period_count > 0:
                # Both present - determine which is decimal separator
                # The last one is usually the decimal separator
                last_period_pos = num.rfind('.')
                last_comma_pos = num.rfind(',')

                if last_comma_pos > last_period_pos:
                    # European format: 1.234,56
                    num = num.replace('.', '').replace(',', '.')
                else:
                    # US format: 1,234.56
                    num = num.replace(',', '')
            elif comma_count > 0:
                # Only commas - could be thousands separator or decimal
                # If only one comma and it's followed by 1-2 digits, it's likely decimal (European)
                # Otherwise it's thousands separator (US)
                comma_pos = num.rfind(',')
                after_comma = num[comma_pos+1:]
                if comma_count == 1 and len(after_comma) <= 2 and after_comma.isdigit():
                    # Likely European decimal: 123,45
                    num = num.replace(',', '.')
                else:
                    # US thousands separator: 1,234,567
                    num = num.replace(',', '')
            # If only periods, assume US format (thousands separator)
            elif period_count > 1:
                # Multiple periods = thousands separator: 1.234.567
                # Keep last period as decimal if followed by 1-2 digits
                parts = num.split('.')
                if len(parts[-1]) <= 2:
                    # Last part is decimal
                    num = ''.join(parts[:-1]) + '.' + parts[-1]
                else:
                    # All are thousands separators
                    num = num.replace('.', '')

            # Remove any remaining spaces
            num = num.replace(' ', '')

            # Only add if it's a valid number
            try:
                # Test if it's parseable as a number
                float(num)
                normalized_numbers.append(num)
            except ValueError:
                # Skip invalid numbers
                continue

        return normalized_numbers
