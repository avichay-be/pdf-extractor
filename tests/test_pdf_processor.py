"""
Unit tests for PDF processor service.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import asyncio

from pypdf import PdfReader, PdfWriter

from src.services.pdf_processor import PDFProcessor


class TestPDFProcessor(unittest.IsolatedAsyncioTestCase):
    """Test cases for PDFProcessor class."""

    @patch('src.services.pdf_processor.settings')
    def setUp(self, mock_settings):
        """Set up test fixtures."""
        mock_settings.MAX_PAGES_PER_CHUNK = 15
        self.processor = PDFProcessor(max_pages_per_chunk=10)

    @patch('src.services.pdf_processor.settings')
    def test_init_default_max_pages(self, mock_settings):
        """Test initialization with default max pages."""
        mock_settings.MAX_PAGES_PER_CHUNK = 15
        processor = PDFProcessor()
        # Default in config.py is 15
        self.assertEqual(processor.max_pages_per_chunk, 15)

    def test_init_custom_max_pages(self):
        """Test initialization with custom max pages."""
        processor = PDFProcessor(max_pages_per_chunk=5)
        self.assertEqual(processor.max_pages_per_chunk, 5)

    def test_combine_markdown_results_empty(self):
        """Test combining empty markdown list."""
        result = self.processor.combine_markdown_results([])
        self.assertEqual(result, "")

    def test_combine_markdown_results_single(self):
        """Test combining single markdown chunk."""
        markdown = "# Test Document\n\nContent here."
        result = self.processor.combine_markdown_results([markdown])
        self.assertEqual(result, markdown)

    def test_combine_markdown_results_multiple(self):
        """Test combining multiple markdown chunks."""
        chunks = [
            "# Part 1\n\nFirst section.",
            "# Part 2\n\nSecond section.",
            "# Part 3\n\nThird section."
        ]
        result = self.processor.combine_markdown_results(chunks)

        self.assertIn("# Part 1", result)
        self.assertIn("# Part 2", result)
        self.assertIn("# Part 3", result)
        self.assertIn("---", result)  # Check for separator

    def test_combine_markdown_results_strips_whitespace(self):
        """Test that combining strips extra whitespace."""
        chunks = [
            "  # Part 1  \n\n  ",
            "  # Part 2  \n\n  "
        ]
        result = self.processor.combine_markdown_results(chunks)

        self.assertTrue(result.startswith("# Part 1"))
        self.assertTrue(result.endswith("# Part 2"))

    @patch('src.services.pdf_processor.PdfReader')
    def test_split_by_main_outlines_small_pdf(self, mock_reader_class):
        """Test that small PDFs are not split."""
        # Mock PDF with 5 pages (below threshold)
        mock_reader = Mock()
        mock_reader.pages = [Mock()] * 5
        mock_reader_class.return_value = mock_reader

        result = self.processor.split_by_main_outlines('/fake/path.pdf')

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], '/fake/path.pdf')

    @patch('src.services.pdf_processor.PdfReader')
    def test_get_main_outlines_no_outlines(self, mock_reader_class):
        """Test extraction when PDF has no outlines."""
        mock_reader = Mock()
        mock_reader.outline = []

        outlines = self.processor._get_main_outlines(mock_reader)

        self.assertEqual(len(outlines), 0)

    @patch('src.services.pdf_processor.PdfReader')
    def test_get_main_outlines_with_valid_outlines(self, mock_reader_class):
        """Test extraction of main outlines."""
        mock_reader = Mock()

        # Create mock outline items
        outline_item1 = Mock()
        outline_item1.title = "Chapter 1"
        outline_item1.page = Mock()

        outline_item2 = Mock()
        outline_item2.title = "Chapter 2"
        outline_item2.page = Mock()

        mock_reader.outline = [outline_item1, outline_item2]
        mock_reader.get_destination_page_number.side_effect = [0, 5]

        outlines = self.processor._get_main_outlines(mock_reader)

        self.assertEqual(len(outlines), 2)
        self.assertEqual(outlines[0]['title'], "Chapter 1")
        self.assertEqual(outlines[0]['page'], 0)
        self.assertEqual(outlines[1]['title'], "Chapter 2")
        self.assertEqual(outlines[1]['page'], 5)

    @patch('src.services.pdf_processor.PdfReader')
    def test_get_main_outlines_skips_nested(self, mock_reader_class):
        """Test that nested outlines are skipped."""
        mock_reader = Mock()

        # Mix of outline items and nested lists
        outline_item = Mock()
        outline_item.title = "Chapter 1"
        outline_item.page = Mock()

        mock_reader.outline = [outline_item, ["nested", "items"]]
        mock_reader.get_destination_page_number.return_value = 0

        outlines = self.processor._get_main_outlines(mock_reader)

        self.assertEqual(len(outlines), 1)
        self.assertEqual(outlines[0]['title'], "Chapter 1")

    async def test_cleanup_chunks_removes_files(self):
        """Test cleanup removes temporary chunk files."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False) as tmp1, \
             tempfile.NamedTemporaryFile(delete=False) as tmp2:
            chunk_paths = [tmp1.name, tmp2.name]

        # Verify files exist
        self.assertTrue(Path(chunk_paths[0]).exists())
        self.assertTrue(Path(chunk_paths[1]).exists())

        # Cleanup
        await self.processor.cleanup_chunks(chunk_paths)

        # Verify files removed
        self.assertFalse(Path(chunk_paths[0]).exists())
        self.assertFalse(Path(chunk_paths[1]).exists())

    async def test_cleanup_chunks_preserves_original(self):
        """Test cleanup preserves original file when specified."""
        with tempfile.NamedTemporaryFile(delete=False) as original, \
             tempfile.NamedTemporaryFile(delete=False) as chunk:

            original_path = original.name
            chunk_path = chunk.name

        # Cleanup with original path
        await self.processor.cleanup_chunks([original_path, chunk_path], original_path)

        # Original should still exist, chunk should be removed
        self.assertTrue(Path(original_path).exists())
        self.assertFalse(Path(chunk_path).exists())

        # Cleanup original manually
        Path(original_path).unlink(missing_ok=True)

    def test_encode_pdf_base64_mock(self):
        """Test base64 encoding of PDF (mocked)."""
        # Create a simple PDF for testing
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            writer.write(tmp)
            tmp_path = tmp.name

        try:
            # Use the shared utility function instead of client method
            from src.core.utils import encode_pdf_to_base64
            encoded = encode_pdf_to_base64(tmp_path)

            # Check it's a valid base64 string
            self.assertIsInstance(encoded, str)
            self.assertGreater(len(encoded), 0)

            # Check it starts with expected PDF header when decoded
            import base64
            decoded = base64.b64decode(encoded)
            self.assertTrue(decoded.startswith(b'%PDF'))

        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
