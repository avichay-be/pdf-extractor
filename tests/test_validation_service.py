"""
Unit tests for the validation service.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from src.services.validation import ValidationService, ValidationResult, CrossValidationReport


class TestProblemPatternDetection(unittest.TestCase):
    """Test cases for problem pattern detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.validation_service = ValidationService()

    def test_detect_empty_table_pattern(self):
        """Test detection of problematic empty table cells."""
        # Create markdown with many empty table cells
        problematic_content = """
# Financial Report

| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | |
"""
        result = self.validation_service.detect_problem_pattern(problematic_content)
        self.assertTrue(result, "Should detect problematic empty table pattern")

    def test_normal_content_not_detected(self):
        """Test that normal content is not detected as problematic."""
        normal_content = """
# Financial Report

This is a normal document with text content.

## Tables

| Year | Revenue | Profit |
|------|---------|--------|
| 2022 | $100M   | $20M   |
| 2023 | $120M   | $25M   |
"""
        result = self.validation_service.detect_problem_pattern(normal_content)
        self.assertFalse(result, "Should not detect normal content as problematic")

    def test_small_empty_table_not_detected(self):
        """Test that small empty tables (< 5 rows) are not detected."""
        small_table = """
| | | |
| | | |
| | | |
"""
        result = self.validation_service.detect_problem_pattern(small_table)
        self.assertFalse(result, "Should not detect small empty tables as problematic")

    def test_empty_content_not_detected(self):
        """Test that empty content is not detected as problematic."""
        result = self.validation_service.detect_problem_pattern("")
        self.assertFalse(result, "Should not detect empty content as problematic")


class TestNumberExtraction(unittest.TestCase):
    """Test cases for number extraction and normalization."""

    def setUp(self):
        """Set up test fixtures."""
        self.validation_service = ValidationService()

    def test_extract_simple_numbers(self):
        """Test extraction of simple integers."""
        text = "The company earned 1000 dollars in 2024"
        numbers = self.validation_service._extract_numbers(text)
        self.assertIn("1000", numbers)
        self.assertIn("2024", numbers)
        self.assertEqual(len(numbers), 2)

    def test_extract_numbers_with_thousands_separator(self):
        """Test extraction of numbers with comma separators."""
        text = "Revenue was 1,234,567 in Q1"
        numbers = self.validation_service._extract_numbers(text)
        self.assertIn("1234567", numbers)

    def test_extract_decimal_numbers(self):
        """Test extraction of decimal numbers."""
        text = "The rate was 12.5% and amount was 123.45"
        numbers = self.validation_service._extract_numbers(text)
        self.assertIn("12.5", numbers)
        self.assertIn("123.45", numbers)

    def test_extract_negative_numbers(self):
        """Test extraction of negative numbers."""
        text = "Loss of -500 and deficit of -1,234"
        numbers = self.validation_service._extract_numbers(text)
        self.assertIn("-500", numbers)
        self.assertIn("-1234", numbers)

    def test_extract_currency_symbols(self):
        """Test that currency symbols are removed."""
        text = "$1,000 or ₪5,000 or €3,000"
        numbers = self.validation_service._extract_numbers(text)
        self.assertIn("1000", numbers)
        self.assertIn("5000", numbers)
        self.assertIn("3000", numbers)

    def test_extract_percentages(self):
        """Test that percentages are handled."""
        text = "Growth of 15% and decline of 5%"
        numbers = self.validation_service._extract_numbers(text)
        self.assertIn("15", numbers)
        self.assertIn("5", numbers)

    def test_extract_european_format(self):
        """Test European number format (period for thousands, comma for decimal)."""
        text = "Amount: 1.234.567,89"
        numbers = self.validation_service._extract_numbers(text)
        # Should normalize to 1234567.89
        self.assertIn("1234567.89", numbers)

    def test_extract_mixed_hebrew_content(self):
        """Test extraction from Hebrew text."""
        text = "הכנסה: 1,234,567 ש״ח בשנת 2024"
        numbers = self.validation_service._extract_numbers(text)
        self.assertIn("1234567", numbers)
        self.assertIn("2024", numbers)


class TestNumberFrequencySimilarity(unittest.TestCase):
    """Test cases for number-frequency based similarity calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validation_service = ValidationService()

    def test_identical_numbers_same_frequency(self):
        """Test that identical number distributions score 100%."""
        content1 = "Revenue: 1,000,000 in 2024"
        content2 = "Income: 1000000 for year 2024"
        similarity = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        self.assertAlmostEqual(similarity, 1.0, places=5, msg="Identical number distributions should be ~100% similar")

    def test_same_numbers_different_frequency(self):
        """Test that same numbers with different frequencies score lower."""
        content1 = "Q1: 100, Q2: 100, Q3: 100"  # 100 appears 3 times
        content2 = "Q1: 100, Q3: 150"  # 100 appears 1 time, 150 appears 1 time
        similarity = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        # Should be less than 1.0 because frequencies differ
        self.assertLess(similarity, 1.0)
        self.assertGreater(similarity, 0.0)

    def test_completely_different_numbers(self):
        """Test that completely different numbers score 0%."""
        content1 = "Revenue: 1,000,000"
        content2 = "Expenses: 500,000"
        similarity = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        self.assertEqual(similarity, 0.0, "Completely different numbers should be 0% similar")

    def test_missing_numbers(self):
        """Test detection of missing numbers."""
        content1 = "Assets: 500,000, Debt: 200,000, Net: 300,000"
        content2 = "Assets: 500,000, Debt: 200,000"  # Missing net
        similarity = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        # Should be partial match (2 out of 3 numbers match)
        self.assertGreater(similarity, 0.5)
        self.assertLess(similarity, 1.0)

    def test_extra_numbers(self):
        """Test detection of extra numbers."""
        content1 = "Total: 1000"
        content2 = "Total: 1000, Subtotal: 500, Tax: 500"  # Extra numbers
        similarity = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        # Should be partial match
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)

    def test_format_agnostic(self):
        """Test that number formatting doesn't affect similarity."""
        content1 = "Amount: 1,234,567.89 in year 2024"
        content2 = "Sum: 1.234.567,89 for 2024"  # European format
        similarity = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        self.assertGreater(similarity, 0.95, "Format differences should not significantly affect similarity")

    def test_hebrew_financial_content(self):
        """Test with Hebrew financial content."""
        content1 = "הכנסות: 1,234,567 ש״ח בשנת 2024, רווח: 567,890 ש״ח"
        content2 = "Income: 1234567 ILS in 2024, profit: 567890"
        similarity = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        self.assertEqual(similarity, 1.0, "Hebrew and English with same numbers should match")

    def test_ocr_error_in_number(self):
        """Test that OCR errors in numbers are detected."""
        content1 = "Total: 1,234,567"
        content2 = "Total: 1,234,557"  # OCR error: 567 → 557
        similarity = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        self.assertEqual(similarity, 0.0, "Different numbers should not match")

    def test_both_empty(self):
        """Test similarity when both contents are empty."""
        similarity = self.validation_service.calculate_similarity_number_frequency("", "")
        self.assertEqual(similarity, 1.0, "Both empty should be 100% similar")

    def test_one_empty(self):
        """Test similarity when one content is empty."""
        similarity = self.validation_service.calculate_similarity_number_frequency("Revenue: 1000", "")
        self.assertEqual(similarity, 0.0, "One empty should be 0% similar")

    def test_no_numbers(self):
        """Test similarity when neither content has numbers."""
        content1 = "This is just text with no numbers"
        content2 = "Another text without any digits"
        similarity = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        self.assertEqual(similarity, 1.0, "Both with no numbers should be 100% similar")


class TestLevenshteinSimilarity(unittest.TestCase):
    """Test cases for Levenshtein-based similarity calculation."""

    def setUp(self):
        """Set up test fixtures."""
        self.validation_service = ValidationService()

    def test_normalize_for_comparison_basic(self):
        """Test text normalization removes non-alphanumeric characters."""
        # English text
        result = self.validation_service._normalize_for_comparison("Hello, World! 123")
        self.assertEqual(result, "helloworld123")

        # Remove punctuation and whitespace
        result = self.validation_service._normalize_for_comparison("Test | Table | Format")
        self.assertEqual(result, "testtableformat")

        # Only punctuation
        result = self.validation_service._normalize_for_comparison("| | | --- |")
        self.assertEqual(result, "")

    def test_normalize_for_comparison_hebrew(self):
        """Test normalization works with Hebrew characters."""
        result = self.validation_service._normalize_for_comparison("שלום עולם! 456")
        # Hebrew letters and numbers should remain, spaces and punctuation removed
        self.assertIn("שלום", result.replace(" ", ""))
        self.assertIn("456", result)

    def test_normalize_for_comparison_mixed(self):
        """Test normalization with mixed content."""
        # Mixed English, Hebrew, numbers
        result = self.validation_service._normalize_for_comparison("Test 123 שלום | Format")
        # Should keep alphanumeric from all languages
        self.assertGreater(len(result), 0)
        self.assertNotIn("|", result)
        self.assertNotIn(" ", result)

    def test_identical_content(self):
        """Test that identical content has 100% similarity."""
        content = "This is a test document with some content."
        similarity = self.validation_service.calculate_similarity_levenshtein(content, content)
        self.assertEqual(similarity, 1.0, "Identical content should have 100% similarity")

    def test_identical_content_different_formatting(self):
        """Test that same content with different formatting is identical."""
        # Same content, different markdown table formatting
        content1 = "Name Value Year Test 100 2024"
        content2 = "| Name | Value | Year |\n|------|-------|------|\n| Test | 100   | 2024 |"
        similarity = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        # After normalization: "namevalueyeartest1002024" vs same
        self.assertEqual(similarity, 1.0, "Same content with different formatting should be 100% similar")

    def test_whitespace_and_punctuation_ignored(self):
        """Test that whitespace and punctuation differences are ignored."""
        content1 = "HelloWorld123"
        content2 = "Hello, World! 1-2-3"
        similarity = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        # Both normalize to "helloworld123"
        self.assertEqual(similarity, 1.0, "Whitespace and punctuation should be ignored")

    def test_table_formatting_ignored(self):
        """Test that table pipe separators are ignored."""
        content1 = "Name Test Value 100 Year 2024"
        content2 = "Name | Test | Value | 100 | Year | 2024"
        similarity = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        # Both normalize to same alphanumeric string
        self.assertEqual(similarity, 1.0, "Table formatting should be ignored")

    def test_completely_different_content(self):
        """Test that completely different content has low similarity."""
        content1 = "This is document A"
        content2 = "Completely different text"
        similarity = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        self.assertLess(similarity, 0.5, "Completely different content should have low similarity")

    def test_similar_content(self):
        """Test that similar content has high similarity."""
        content1 = "This is a test document with some content about financial reports."
        content2 = "This is a test document with content about financial reports."
        similarity = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        self.assertGreater(similarity, 0.90, "Similar content should have > 90% similarity")

    def test_hebrew_content_similarity(self):
        """Test similarity with Hebrew content."""
        content1 = "סלע קפיטל נדל\"ן בע\"מ דוחות כספיים"
        content2 = "סלע | קפיטל | נדלן | בעמ | דוחות | כספיים"
        similarity = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        # Most content is the same, formatting differs
        self.assertGreater(similarity, 0.85, "Hebrew content with different formatting should be highly similar")

    def test_empty_content(self):
        """Test similarity with empty content."""
        # Both empty = identical
        similarity1 = self.validation_service.calculate_similarity_levenshtein("", "")
        self.assertEqual(similarity1, 1.0, "Both empty should be 100% similar")

        # One empty, one not = completely different
        similarity2 = self.validation_service.calculate_similarity_levenshtein("", "content")
        self.assertEqual(similarity2, 0.0, "One empty should be 0% similar")

    def test_only_punctuation_content(self):
        """Test similarity when both strings contain only punctuation."""
        content1 = "| | | --- |"
        content2 = "--- | --- | ---"
        similarity = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        # Both normalize to empty, so identical
        self.assertEqual(similarity, 1.0, "Both punctuation-only should be 100% similar")

    def test_punctuation_vs_content(self):
        """Test similarity when one has only punctuation, other has content."""
        content1 = "| | | --- |"
        content2 = "Name Value 123"
        similarity = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        # One normalizes to empty, other has content
        self.assertEqual(similarity, 0.0, "Punctuation vs content should be 0% similar")

    def test_95_percent_threshold(self):
        """Test that 5% character difference is detected."""
        # Create two strings where content2 has ~5% extra characters
        content1 = "a" * 100
        content2 = "a" * 95 + "b" * 5  # 5% different

        similarity = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        # Should be around 95% similar (within tolerance)
        self.assertGreater(similarity, 0.90, "Should be > 90% similar")
        self.assertLess(similarity, 1.0, "Should not be 100% similar")


class TestSimilarityDispatcher(unittest.TestCase):
    """Test cases for the main similarity calculation dispatcher."""

    def setUp(self):
        """Set up test fixtures."""
        self.validation_service = ValidationService()

    def test_default_method_is_number_frequency(self):
        """Test that default method uses number frequency."""
        content1 = "Revenue: 1,000,000 in 2024"
        content2 = "Income: 1000000 for year 2024"
        # Should use number_frequency by default (same numbers = 100%)
        similarity = self.validation_service.calculate_similarity(content1, content2)
        self.assertAlmostEqual(similarity, 1.0, places=5)

    def test_dispatches_correctly(self):
        """Test that dispatcher calls the correct method."""
        content1 = "Revenue: 1000"
        content2 = "Income: 1000"

        # Number frequency method should match (same number)
        sim_num = self.validation_service.calculate_similarity_number_frequency(content1, content2)
        self.assertAlmostEqual(sim_num, 1.0, places=5)

        # Levenshtein method should not match (different words)
        sim_lev = self.validation_service.calculate_similarity_levenshtein(content1, content2)
        self.assertLess(sim_lev, 0.5, "Different words should have low Levenshtein similarity")


class TestSampleValidationLogic(unittest.TestCase):
    """Test cases for sample validation logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.validation_service = ValidationService()

    def test_no_validation_without_query(self):
        """Test that no validation happens without query filtering."""
        for page_idx in range(100):
            result = self.validation_service.should_validate_page(
                page_index=page_idx,
                total_pages=100,
                has_query=False,
                random_offset=0
            )
            self.assertFalse(result, f"Page {page_idx} should not be validated without query")

    def test_sample_rate_with_query(self):
        """Test that every 10th page is validated with query filtering."""
        random_offset = 0
        sample_rate = 10

        validated_pages = []
        for page_idx in range(100):
            if self.validation_service.should_validate_page(
                page_index=page_idx,
                total_pages=100,
                has_query=True,
                random_offset=random_offset
            ):
                validated_pages.append(page_idx)

        # Should validate every 10th page starting from offset 0
        expected_pages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
        self.assertEqual(validated_pages, expected_pages, "Should validate every 10th page")

    def test_random_offset_changes_sampling(self):
        """Test that different random offsets change which pages are sampled."""
        offset1 = 0
        offset2 = 5

        # Get pages validated with offset 0
        pages1 = [
            i for i in range(100)
            if self.validation_service.should_validate_page(i, 100, True, offset1)
        ]

        # Get pages validated with offset 5
        pages2 = [
            i for i in range(100)
            if self.validation_service.should_validate_page(i, 100, True, offset2)
        ]

        # Should be different pages
        self.assertNotEqual(pages1, pages2, "Different offsets should sample different pages")
        # But same number of pages
        self.assertEqual(len(pages1), len(pages2), "Should validate same number of pages")


class TestValidationResult(unittest.TestCase):
    """Test cases for ValidationResult data structure."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult with all fields."""
        result = ValidationResult(
            page_number=5,
            similarity_score=0.95,
            passed=True,
            has_problem_pattern=False,
            alternative_content="Alternative content",
            processing_time=1.5
        )

        self.assertEqual(result.page_number, 5)
        self.assertEqual(result.similarity_score, 0.95)
        self.assertTrue(result.passed)
        self.assertFalse(result.has_problem_pattern)
        self.assertEqual(result.alternative_content, "Alternative content")
        self.assertEqual(result.processing_time, 1.5)
        self.assertIsNone(result.error)

    def test_validation_result_with_error(self):
        """Test creating ValidationResult with error."""
        result = ValidationResult(
            page_number=3,
            similarity_score=0.0,
            passed=False,
            has_problem_pattern=True,
            alternative_content=None,
            processing_time=0.5,
            error="OpenAI API error"
        )

        self.assertEqual(result.error, "OpenAI API error")
        self.assertFalse(result.passed)


class TestCrossValidationReport(unittest.TestCase):
    """Test cases for CrossValidationReport data structure."""

    def test_report_creation(self):
        """Test creating CrossValidationReport."""
        result1 = ValidationResult(
            page_number=5,
            similarity_score=0.45,
            passed=False,
            has_problem_pattern=True,
            alternative_content="Fixed content",
            processing_time=1.2
        )

        result2 = ValidationResult(
            page_number=15,
            similarity_score=0.97,
            passed=True,
            has_problem_pattern=False,
            alternative_content=None,
            processing_time=1.0
        )

        report = CrossValidationReport(
            total_pages=50,
            validated_pages=2,
            problem_pages=[5],
            failed_validations=[],
            validation_results=[result1, result2],
            total_time=2.2,
            total_cost=0.01
        )

        self.assertEqual(report.total_pages, 50)
        self.assertEqual(report.validated_pages, 2)
        self.assertEqual(len(report.problem_pages), 1)
        self.assertEqual(len(report.validation_results), 2)
        self.assertEqual(report.total_time, 2.2)
        self.assertEqual(report.total_cost, 0.01)


if __name__ == '__main__':
    unittest.main()
