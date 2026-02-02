"""
Unit tests for query filtering functionality.
"""
import unittest
from src.core.utils import filter_outlines_by_query


class TestQueryFilter(unittest.TestCase):
    """Test cases for outline filtering by query."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample outline metadata
        self.outline_info = [
            {
                'title': 'דוחות כספיים',
                'page': 0,
                'chunk_indices': [0, 1, 2]
            },
            {
                'title': 'דוח דירקטוריון',
                'page': 30,
                'chunk_indices': [3, 4]
            },
            {
                'title': 'תקציר',
                'page': 50,
                'chunk_indices': [5]
            },
            {
                'title': 'Financial Reports',
                'page': 60,
                'chunk_indices': [6, 7]
            }
        ]

    def test_filter_by_hebrew_financial_reports(self):
        """Test filtering by 'דוחות כספיים' (default query)."""
        result = filter_outlines_by_query(self.outline_info, "דוחות כספיים")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['title'], 'דוחות כספיים')
        self.assertEqual(result[0]['chunk_indices'], [0, 1, 2])

    def test_filter_by_hebrew_directors_report(self):
        """Test filtering by 'דוח דירקטוריון'."""
        result = filter_outlines_by_query(self.outline_info, "דוח דירקטוריון")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['title'], 'דוח דירקטוריון')
        self.assertEqual(result[0]['chunk_indices'], [3, 4])

    def test_filter_partial_match(self):
        """Test filtering with partial match."""
        result = filter_outlines_by_query(self.outline_info, "דוח")

        # Should match both 'דוחות כספיים' and 'דוח דירקטוריון'
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['title'], 'דוחות כספיים')
        self.assertEqual(result[1]['title'], 'דוח דירקטוריון')

    def test_filter_case_insensitive(self):
        """Test that filtering is case-insensitive."""
        result = filter_outlines_by_query(self.outline_info, "financial")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['title'], 'Financial Reports')

    def test_filter_no_match_returns_all(self):
        """Test that no match returns all outlines (fallback)."""
        result = filter_outlines_by_query(self.outline_info, "nonexistent query")

        # Should return all outlines when no match found
        self.assertEqual(len(result), 4)
        self.assertEqual(result, self.outline_info)

    def test_filter_empty_query(self):
        """Test filtering with empty query returns all."""
        result = filter_outlines_by_query(self.outline_info, "")

        self.assertEqual(len(result), 4)
        self.assertEqual(result, self.outline_info)

    def test_filter_none_query(self):
        """Test filtering with None query returns all."""
        result = filter_outlines_by_query(self.outline_info, None)

        self.assertEqual(len(result), 4)
        self.assertEqual(result, self.outline_info)

    def test_filter_none_outline_info(self):
        """Test filtering with None outline_info returns None."""
        result = filter_outlines_by_query(None, "query")

        self.assertIsNone(result)

    def test_filter_empty_outline_info(self):
        """Test filtering with empty outline_info returns empty list."""
        result = filter_outlines_by_query([], "query")

        self.assertEqual(result, [])

    def test_filter_multiple_matches(self):
        """Test filtering returns multiple matches in order."""
        outline_info = [
            {'title': 'Introduction to Reports', 'page': 0, 'chunk_indices': [0]},
            {'title': 'Financial Report 2024', 'page': 10, 'chunk_indices': [1]},
            {'title': 'Summary', 'page': 20, 'chunk_indices': [2]},
            {'title': 'Annual Report', 'page': 30, 'chunk_indices': [3]}
        ]

        result = filter_outlines_by_query(outline_info, "report")

        # Should match all titles containing "report"
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]['title'], 'Introduction to Reports')
        self.assertEqual(result[1]['title'], 'Financial Report 2024')
        self.assertEqual(result[2]['title'], 'Annual Report')


class TestQueryFilterIntegration(unittest.TestCase):
    """Integration tests for query filtering with API models."""

    def test_default_query_value(self):
        """Test that default query value is 'דוחות כספיים'."""
        from src.models.api_models import Base64FileRequest
        import base64

        # Create request without query field
        pdf_content = base64.b64encode(b"dummy pdf content").decode('utf-8')
        request = Base64FileRequest(
            filename="test.pdf",
            file_content=pdf_content
        )

        # Should have default value
        self.assertEqual(request.query, "דוחות כספיים")

    def test_custom_query_value(self):
        """Test setting custom query value."""
        from src.models.api_models import Base64FileRequest
        import base64

        pdf_content = base64.b64encode(b"dummy pdf content").decode('utf-8')
        request = Base64FileRequest(
            filename="test.pdf",
            file_content=pdf_content,
            query="דוח דירקטוריון"
        )

        self.assertEqual(request.query, "דוח דירקטוריון")


if __name__ == '__main__':
    unittest.main()
