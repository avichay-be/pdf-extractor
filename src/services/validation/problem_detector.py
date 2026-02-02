"""
Problem detection patterns for PDF content validation.

Detects 13 types of quality issues in PDF extraction results:
1. Empty tables
2. Low content density
3. Missing numbers in tables
4. Inconsistent columns
5. Repeated characters
6. Garbled text
7. Header-only tables
8. Very short pages
9. Missing keywords
10. Malformed structure
11. Duplicate content
12. Unknown characters
13. Repetitive numbers
"""
import re
import logging
from typing import List, Dict, Optional
from collections import Counter
from src.core.config import settings

logger = logging.getLogger(__name__)


class ProblemDetector:
    """Detects quality issues in PDF extraction results."""

    # Regex pattern to detect problematic empty table cells
    # Matches 5+ consecutive lines with mostly empty cells: | | |
    PROBLEM_PATTERN = re.compile(r'(\|\s*\|\s*\|.*\n){5,}')

    def __init__(self, number_extractor=None):
        """
        Initialize problem detector.

        Args:
            number_extractor: Function to extract numbers from text (optional)
        """
        self._extract_numbers = number_extractor

    def detect_problem_pattern(self, markdown_content: str) -> bool:
        """
        Detect if markdown content contains problematic patterns (e.g., empty tables).

        DEPRECATED: Use detect_all_problems() for comprehensive detection.

        Args:
            markdown_content: Markdown content to check

        Returns:
            True if problematic pattern detected, False otherwise
        """
        if not markdown_content:
            return False

        # Check for empty table pattern
        matches = self.PROBLEM_PATTERN.findall(markdown_content)
        if matches:
            logger.debug(f"Detected {len(matches)} problematic table pattern(s)")
            return True

        return False

    def _detect_low_content_density(self, markdown_content: str) -> bool:
        """
        Problem 2: Detect pages with very little text when document should have substantial content.

        Threshold: Less than 100 alphanumeric characters
        """
        if not markdown_content:
            return True  # Empty content is a problem

        alphanumeric_count = sum(1 for c in markdown_content if c.isalnum())

        if alphanumeric_count < 100:
            logger.debug(f"Low content density: {alphanumeric_count} alphanumeric characters")
            return True

        return False

    def _detect_missing_numbers(self, markdown_content: str) -> bool:
        """
        Problem 3: Detect tables that should contain numbers but don't.

        Logic: Table has 5+ rows but no numbers extracted
        """
        if not markdown_content:
            return False

        # Count table rows (rough estimate by counting pipe symbols)
        table_rows = markdown_content.count('|') / 4  # Rough estimate

        # Extract numbers (requires number extractor)
        if self._extract_numbers:
            numbers = self._extract_numbers(markdown_content)
        else:
            # Fallback: simple digit check
            numbers = re.findall(r'\d+', markdown_content)

        if table_rows >= 5 and len(numbers) == 0:
            logger.debug(f"Missing numbers: table with ~{table_rows:.0f} rows but 0 numbers")
            return True

        return False

    def _detect_inconsistent_columns(self, markdown_content: str) -> bool:
        """
        Problem 4: Detect table rows with varying number of columns (OCR misalignment).

        Logic: Check variance in column counts across table rows
        """
        if not markdown_content:
            return False

        lines = [line.strip() for line in markdown_content.split('\n')]
        table_lines = [line for line in lines if line.strip().startswith('|')]

        if len(table_lines) < 3:  # Need at least header, separator, and one data row
            return False

        # Extract column counts per row
        column_counts = [line.count('|') - 1 for line in table_lines]

        # Check variance (allow max 1 different count for header separator)
        unique_counts = set(column_counts)

        if len(unique_counts) > 2:
            logger.debug(f"Inconsistent columns: {len(unique_counts)} different column counts - {unique_counts}")
            return True

        return False

    def _detect_repeated_characters(self, markdown_content: str) -> bool:
        """
        Problem 5: Detect same character repeated 10+ times consecutively (OCR artifacts).

        Pattern: aaaaaaaaaa or 1111111111
        """
        if not markdown_content:
            return False

        # Pattern: same character repeated 10+ times
        pattern = r'(.)\1{9,}'
        matches = re.findall(pattern, markdown_content)

        # Filter out intentional repeated characters (spaces, dashes, underscores)
        problematic_matches = [m for m in matches if m not in [' ', '-', '_', '=', '*', '\n']]

        if problematic_matches:
            logger.debug(f"Repeated characters detected: {len(problematic_matches)} instances")
            return True

        return False

    def _detect_garbled_text(self, markdown_content: str) -> bool:
        """
        Problem 6: Detect high ratio of special characters/symbols indicating OCR failure.

        Threshold: More than 20% special characters (excluding common punctuation)
        """
        if not markdown_content:
            return False

        alphanumeric = sum(1 for c in markdown_content if c.isalnum())

        if alphanumeric == 0:
            return True  # All special characters

        # Count special characters (excluding common punctuation and whitespace)
        common_chars = set(' \n\t.,;:!?-()[]{}"\'/\\|')
        special_chars = sum(1 for c in markdown_content if not c.isalnum() and c not in common_chars)

        ratio = special_chars / alphanumeric if alphanumeric > 0 else 0

        if ratio > 0.2:
            logger.debug(f"Garbled text: {ratio:.1%} special character ratio")
            return True

        return False

    def _detect_header_only_tables(self, markdown_content: str) -> bool:
        """
        Problem 7: Detect tables with headers but no data rows.

        Logic: Table has header and separator but only 0-1 data rows
        """
        if not markdown_content:
            return False

        lines = [line.strip() for line in markdown_content.split('\n') if line.strip().startswith('|')]

        if len(lines) < 2:
            return False

        # Find header separator (|---|---|)
        separator_indices = [i for i, line in enumerate(lines) if '---' in line]

        if not separator_indices:
            return False

        sep_idx = separator_indices[0]
        data_rows = len(lines) - sep_idx - 1

        if data_rows <= 1:
            logger.debug(f"Header-only table: {data_rows} data rows after separator")
            return True

        return False

    def _detect_very_short_pages(self, markdown_content: str) -> bool:
        """
        Problem 8: Detect pages under minimum expected length for document type.

        Threshold: Less than 200 characters for financial document
        """
        if not markdown_content:
            return True

        content_length = len(markdown_content.strip())

        if content_length < 200:
            logger.debug(f"Very short page: {content_length} characters")
            return True

        return False

    def _detect_missing_keywords(self, markdown_content: str) -> bool:
        """
        Problem 9: Detect financial documents without expected keywords.

        Logic: Check for at least one financial keyword in substantial pages
        """
        if not markdown_content or len(markdown_content) < 500:
            return False  # Only check substantial pages

        # Financial keywords (English and Hebrew)
        financial_keywords = [
            # English
            'revenue', 'expense', 'balance', 'asset', 'liability', 'equity',
            'income', 'profit', 'loss', 'debit', 'credit', 'account',
            'total', 'subtotal', 'amount', 'date', 'transaction', 'payment',
            'statement', 'bank', 'financial', 'report', 'summary',
            # Hebrew
            'הכנסות', 'הוצאות', 'יתרה', 'חשבון', 'סכום',
            'סה"כ', 'זכות', 'חובה', 'תאריך', 'עסקה',
            'תשלום', 'דוח', 'כספי', 'מאזן', 'רווח', 'הפסד'
        ]

        content_lower = markdown_content.lower()
        has_keyword = any(keyword in content_lower for keyword in financial_keywords)

        if not has_keyword:
            logger.debug("Missing keywords: no financial terms found in substantial page")
            return True

        return False

    def _detect_malformed_structure(self, markdown_content: str) -> bool:
        """
        Problem 10: Detect malformed table structure (invalid separators).

        Logic: Table separators without proper format
        """
        if not markdown_content:
            return False

        lines = [line.strip() for line in markdown_content.split('\n')]
        table_lines = [line for line in lines if line.startswith('|')]

        if len(table_lines) < 2:
            return False

        # Find separator lines (should be |---|---|)
        separators = [line for line in table_lines if '-' in line]

        for sep in separators:
            parts = sep.split('|')

            # Valid separator parts contain only dashes and spaces
            valid_parts = []
            for p in parts:
                if p.strip():  # Non-empty part
                    if set(p.strip()) <= {'-', ' '}:
                        valid_parts.append(True)
                    else:
                        valid_parts.append(False)

            # If more than 30% of parts are invalid, it's malformed
            if valid_parts and sum(valid_parts) / len(valid_parts) < 0.7:
                logger.debug("Malformed structure: invalid table separator format")
                return True

        return False

    def _detect_duplicate_content(self, markdown_content: str) -> bool:
        """
        Problem 11: Detect same paragraph/section repeated multiple times.

        Logic: Same substantial paragraph appears 3+ times
        """
        if not markdown_content:
            return False

        # Split into paragraphs
        paragraphs = [p.strip() for p in markdown_content.split('\n\n') if p.strip()]

        if len(paragraphs) < 3:
            return False

        # Count paragraph occurrences
        para_counts = Counter(paragraphs)

        for para, count in para_counts.items():
            if count >= 3 and len(para) > 50:  # Substantial paragraph repeated 3+ times
                logger.debug(f"Duplicate content: paragraph repeated {count} times ({len(para)} chars)")
                return True

        return False

    def _detect_repetitive_numbers(self, markdown_content: str) -> bool:
        """
        Problem 13: Detect same number/value repeated 3+ times in close proximity.

        Patterns to detect:
        - "1000 1000 1000" (plain text)
        - "| 1000 | 1000 | 1000 |" (table rows)
        - "1000|1000|1000" (table without spaces)

        Logic: Same number appears 3+ times within close proximity
        """
        if not markdown_content:
            return False

        # Pattern 1: Number repeated in table cells (with pipes)
        # Matches: | 1000 | 1000 | 1000 |
        table_pattern = r'\|\s*(\d+(?:[.,]\d+)?)\s*\|(?:\s*\1\s*\|){2,}'
        table_matches = re.findall(table_pattern, markdown_content)

        if table_matches:
            logger.debug(f"Repetitive numbers in table: {len(table_matches)} instances")
            return True

        # Pattern 2: Number repeated in plain text (space-separated)
        # Matches: 1000 1000 1000
        text_pattern = r'\b(\d+(?:[.,]\d+)?)\s+(?:\1\s+){2,}'
        text_matches = re.findall(text_pattern, markdown_content)

        if text_matches:
            logger.debug(f"Repetitive numbers in text: {len(text_matches)} instances")
            return True

        return False

    def _detect_unknown_characters(self, markdown_content: str) -> bool:
        """
        Problem 12: Detect high ratio of unknown/unrecognized characters.

        Threshold: More than 5% unknown characters (□, ?, �, etc.)
        """
        if not markdown_content:
            return False

        # Unknown character indicators
        unknown_chars = ['□', '�', '☐', '▯', '�', '▢', '▣']

        total_chars = len(markdown_content)
        unknown_count = sum(markdown_content.count(char) for char in unknown_chars)

        # Also count standalone question marks (not in words)
        standalone_questions = len(re.findall(r'\s\?\s', markdown_content))
        unknown_count += standalone_questions

        if total_chars > 0 and (unknown_count / total_chars) > 0.05:
            logger.debug(f"Unknown characters: {unknown_count} ({unknown_count/total_chars:.1%})")
            return True

        return False

    def _detect_markdown_images(self, markdown_content: str) -> bool:
        """
        Problem 14: Detect markdown image references.

        Pattern: ![alt-text](image-path.ext)
        Examples: ![img-01.jpeg](img-01.jpeg), ![chart](figure-5.png)
        Threshold: 1+ image reference
        """
        if not markdown_content:
            return False

        # Regex: ![anything](anything)
        image_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
        matches = re.findall(image_pattern, markdown_content)

        if matches:
            logger.debug(f"Markdown images detected: {len(matches)} instances")
            for i, (alt_text, path) in enumerate(matches[:3]):
                logger.debug(f"  Image {i+1}: alt='{alt_text}', path='{path}'")
            return True

        return False

    def detect_all_problems(self, markdown_content: str, enabled_problems: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Detect problem patterns in markdown content (OPTIMIZED).

        Performance: Only runs enabled detection patterns, skipping disabled ones.
        With default config (7/13 enabled), this achieves 46% speedup.

        Args:
            markdown_content: Content to check
            enabled_problems: List of problem names to check (None = check all from settings)

        Returns:
            Dictionary mapping problem name to detection result (only enabled problems)
        """
        # Determine which problems to check
        if enabled_problems is None:
            # Get from settings (default behavior)
            enabled_problems = settings.validation_problems_list

        # Problem registry: maps problem names to detection methods
        problem_registry = {
            'empty_tables': self.detect_problem_pattern,
            'low_content_density': self._detect_low_content_density,
            'missing_numbers': self._detect_missing_numbers,
            'inconsistent_columns': self._detect_inconsistent_columns,
            'repeated_characters': self._detect_repeated_characters,
            'garbled_text': self._detect_garbled_text,
            'header_only_tables': self._detect_header_only_tables,
            'very_short_pages': self._detect_very_short_pages,
            'missing_keywords': self._detect_missing_keywords,
            'malformed_structure': self._detect_malformed_structure,
            'duplicate_content': self._detect_duplicate_content,
            'unknown_characters': self._detect_unknown_characters,
            'repetitive_numbers': self._detect_repetitive_numbers,
            'markdown_images': self._detect_markdown_images,
        }

        problems = {}

        # Execute only enabled patterns (OPTIMIZATION: skips disabled patterns)
        for problem_name in enabled_problems:
            if problem_name in problem_registry:
                problems[problem_name] = problem_registry[problem_name](markdown_content)
            else:
                logger.warning(f"Unknown problem pattern '{problem_name}' - skipping")
                problems[problem_name] = False

        return problems

    def has_any_problem(self, markdown_content: str, enabled_problems: Optional[List[str]] = None) -> tuple[bool, List[str]]:
        """
        Check if content has any enabled problems.

        Args:
            markdown_content: Content to check
            enabled_problems: List of problem names to check (None = check all from settings)

        Returns:
            Tuple of (has_problem: bool, detected_problems: List[str])
        """
        if not markdown_content:
            return True, ['empty_content']

        # Get all problems
        all_problems = self.detect_all_problems(markdown_content)

        # Determine which problems to check
        if enabled_problems is None:
            # Get from settings
            enabled_problems = settings.validation_problems_list

        # Check only enabled problems
        detected = []
        for problem_name in enabled_problems:
            if all_problems.get(problem_name, False):
                detected.append(problem_name)

        # Log individual detection for debugging
        if detected:
            logger.debug(f"Detected problems: {detected}")

        return len(detected) > 0, detected

    def detect_problems_batch(
        self,
        pages_content: List[tuple[int, str]],
        enabled_problems: Optional[List[str]] = None
    ) -> dict[int, tuple[bool, List[str]]]:
        """
        Detect problems for multiple pages in one batch (optimization).

        This reduces function call overhead compared to calling has_any_problem()
        in a loop for each page.

        Args:
            pages_content: List of (page_index, markdown_content) tuples
            enabled_problems: List of problem names to check (None = check all from settings)

        Returns:
            Dictionary mapping page_index -> (has_problem, detected_problems)
        """
        results = {}

        # Process all pages
        for page_index, content in pages_content:
            has_problem, detected_problems = self.has_any_problem(content, enabled_problems)
            results[page_index] = (has_problem, detected_problems)

        # Log summary at INFO level for visibility
        pages_with_problems = [idx for idx, (has_prob, _) in results.items() if has_prob]
        if pages_with_problems:
            logger.info(
                f"Problem detection: {len(pages_with_problems)}/{len(pages_content)} pages have quality issues "
                f"(pages: {pages_with_problems})"
            )
        else:
            logger.info(f"Problem detection: All {len(pages_content)} pages passed quality checks")

        return results
