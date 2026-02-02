"""
Unit tests for Azure Document Intelligence Client.

This file tests the AzureDocumentIntelligenceClient class which integrates with
Azure Document Intelligence API for table extraction and merging. Tests cover:
- MergedTable data class functionality
- Client initialization and configuration
- PDF to base64 encoding
- Azure DI API communication (start analyze, polling)
- Table grouping and merging logic
- Integration workflows
- Health check
"""
import unittest
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
import base64
import tempfile
from pathlib import Path
import asyncio

from src.services.azure_document_intelligence_client import (
    AzureDocumentIntelligenceClient,
    MergedTable
)


class TestMergedTable(unittest.TestCase):
    """Test cases for MergedTable data class."""

    def test_merged_table_init(self):
        """Test MergedTable initialization."""
        headers = ["Column1", "Column2", "Column3"]
        page_number = 5

        table = MergedTable(headers, page_number)

        self.assertEqual(table.headers, headers)
        self.assertEqual(table.start_page, page_number)
        self.assertEqual(table.end_page, page_number)
        self.assertEqual(table.data_rows, [])

    def test_merged_table_add_rows_same_page(self):
        """Test adding rows from the same page."""
        table = MergedTable(["A", "B"], 1)

        rows = [["val1", "val2"], ["val3", "val4"]]
        table.add_rows(rows, page_number=1)

        self.assertEqual(table.data_rows, rows)
        self.assertEqual(table.end_page, 1)  # Still same page

    def test_merged_table_add_rows_multiple_pages(self):
        """Test adding rows from multiple pages updates end_page."""
        table = MergedTable(["A", "B"], 1)

        table.add_rows([["r1c1", "r1c2"]], page_number=1)
        table.add_rows([["r2c1", "r2c2"]], page_number=2)
        table.add_rows([["r3c1", "r3c2"]], page_number=3)

        self.assertEqual(len(table.data_rows), 3)
        self.assertEqual(table.start_page, 1)
        self.assertEqual(table.end_page, 3)

    def test_merged_table_to_markdown_basic(self):
        """Test markdown conversion with basic table."""
        table = MergedTable(["Name", "Value"], 1)
        table.add_rows([["Item1", "100"], ["Item2", "200"]], page_number=1)

        markdown = table.to_markdown()

        # Should contain header
        self.assertIn("Table from Page 1", markdown)
        # Should contain column headers
        self.assertIn("Name", markdown)
        self.assertIn("Value", markdown)
        # Should contain data
        self.assertIn("Item1", markdown)
        self.assertIn("100", markdown)
        # Should have markdown table format (pipes and dashes)
        self.assertIn("|", markdown)
        self.assertIn("---", markdown)

    def test_merged_table_to_markdown_varying_columns(self):
        """Test markdown handles rows with varying column counts."""
        table = MergedTable(["A", "B", "C"], 1)

        # Add rows with different column counts
        table.add_rows([["1", "2"]], page_number=1)  # 2 columns
        table.add_rows([["3", "4", "5", "6"]], page_number=1)  # 4 columns

        markdown = table.to_markdown()

        # Should not crash and should produce valid markdown
        self.assertIsInstance(markdown, str)
        self.assertIn("|", markdown)

    def test_merged_table_to_markdown_page_range(self):
        """Test markdown header shows page range for multi-page tables."""
        table = MergedTable(["Col1"], 5)
        table.add_rows([["data1"]], page_number=5)
        table.add_rows([["data2"]], page_number=8)

        markdown = table.to_markdown()

        # Should show page range
        self.assertIn("Pages 5-8", markdown)


class TestAzureDocumentIntelligenceClient(unittest.IsolatedAsyncioTestCase):
    """Test cases for Azure DI Client."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_endpoint = "https://test.cognitiveservices.azure.com/"
        self.test_api_key = "test_azure_di_key_12345"
        self.test_model = "prebuilt-layout"

    # ========== Client Initialization Tests ==========

    @patch('src.services.azure_document_intelligence_client.settings')
    def test_client_init_with_settings(self, mock_settings):
        """Test client initialization with settings."""
        mock_settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = self.test_endpoint
        mock_settings.AZURE_DOCUMENT_INTELLIGENCE_KEY = self.test_api_key
        mock_settings.AZURE_DOCUMENT_INTELLIGENCE_MODEL = self.test_model

        client = AzureDocumentIntelligenceClient()

        self.assertEqual(client.endpoint, self.test_endpoint)
        self.assertEqual(client.api_key, self.test_api_key)
        self.assertEqual(client.model, self.test_model)
        self.assertIn("Ocp-Apim-Subscription-Key", client.headers)
        self.assertEqual(client.headers["Ocp-Apim-Subscription-Key"], self.test_api_key)

    def test_client_init_with_custom_params(self):
        """Test client initialization with custom parameters."""
        custom_endpoint = "https://custom.endpoint.com/"
        custom_key = "custom_key"
        custom_model = "custom-model"
        custom_timeout = 60.0

        client = AzureDocumentIntelligenceClient(
            endpoint=custom_endpoint,
            api_key=custom_key,
            model=custom_model,
            timeout=custom_timeout
        )

        self.assertEqual(client.endpoint, custom_endpoint)
        self.assertEqual(client.api_key, custom_key)
        self.assertEqual(client.model, custom_model)
        self.assertEqual(client.timeout, custom_timeout)

    @patch('src.services.azure_document_intelligence_client.settings')
    def test_client_init_missing_credentials(self, mock_settings):
        """Test initialization fails when credentials are missing."""
        mock_settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT = None
        mock_settings.AZURE_DOCUMENT_INTELLIGENCE_KEY = None

        with self.assertRaises(ValueError) as context:
            AzureDocumentIntelligenceClient()

        self.assertIn("Azure Document Intelligence endpoint and key must be provided", str(context.exception))

    async def test_client_async_context_manager(self):
        """Test client works as async context manager."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        self.assertIsNone(client._client)

        async with client:
            # Client should be initialized
            self.assertIsNotNone(client._client)

        # Client should be closed after exit
        self.assertIsNone(client._client)

    # ========== Base64 Encoding Tests ==========

    def test_encode_pdf_to_base64(self):
        """Test PDF to base64 encoding."""
        test_content = b"%PDF-1.4\ntest PDF content"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(test_content)
            tmp_path = tmp.name

        try:
            client = AzureDocumentIntelligenceClient(
                endpoint=self.test_endpoint,
                api_key=self.test_api_key
            )
            encoded = client._encode_pdf_to_base64(tmp_path)

            # Verify it's valid base64
            decoded = base64.b64decode(encoded)
            self.assertEqual(decoded, test_content)

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_encode_pdf_to_base64_file_not_found(self):
        """Test encoding raises error for non-existent file."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        with self.assertRaises((FileNotFoundError, IOError)):
            client._encode_pdf_to_base64('/fake/path/document.pdf')

    # ========== Start Analyze Tests ==========

    @patch('httpx.AsyncClient')
    async def test_start_analyze_success(self, mock_client_class):
        """Test successful start of analyze operation."""
        # Setup mock
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {
            'Operation-Location': 'https://test.com/operations/12345'
        }
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        # Execute
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        async with client:
            operation_url = await client._start_analyze("fake_base64_content")

        # Assert
        self.assertEqual(operation_url, 'https://test.com/operations/12345')

        # Verify POST request was made
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args

        # Check URL contains model
        self.assertIn(self.test_model, call_args[0][0])

        # Check request body has base64Source
        self.assertIn('json', call_args.kwargs)
        self.assertIn('base64Source', call_args.kwargs['json'])

    @patch('httpx.AsyncClient')
    async def test_start_analyze_missing_operation_location(self, mock_client_class):
        """Test error when Operation-Location header is missing."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {}  # Missing Operation-Location
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        with self.assertRaises(ValueError) as context:
            async with client:
                await client._start_analyze("fake_base64")

        self.assertIn("Operation-Location", str(context.exception))

    @patch('httpx.AsyncClient')
    async def test_start_analyze_api_error(self, mock_client_class):
        """Test error handling when API returns error status."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request error"
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        with self.assertRaises(ValueError) as context:
            async with client:
                await client._start_analyze("fake_base64")

        self.assertIn("400", str(context.exception))

    @patch('httpx.AsyncClient')
    async def test_start_analyze_uses_shared_client(self, mock_client_class):
        """Test start_analyze reuses client from context manager."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 202
        mock_response.headers = {'Operation-Location': 'https://test.com/op/1'}
        mock_client.post.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        async with client:
            await client._start_analyze("base64_1")
            await client._start_analyze("base64_2")

        # Should use same client (not create new one)
        self.assertEqual(mock_client_class.call_count, 1)
        self.assertEqual(mock_client.post.call_count, 2)

    # ========== Poll Analyze Tests ==========

    @patch('httpx.AsyncClient')
    async def test_poll_analyze_success_first_attempt(self, mock_client_class):
        """Test successful polling on first attempt."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'succeeded',
            'createdDateTime': '2024-01-01T00:00:00Z',
            'lastUpdatedDateTime': '2024-01-01T00:00:00Z',
            'analyzeResult': {
                'apiVersion': '2024-11-30',
                'modelId': 'prebuilt-layout',
                'content': 'test content',
                'pages': [{'pageNumber': 1}],
                'tables': [
                    {
                        'rowCount': 2,
                        'columnCount': 2,
                        'cells': [],
                        'spans': [],
                        'boundingRegions': [{'pageNumber': 1, 'polygon': [0, 0, 1, 1]}]
                    }
                ]
            }
        }
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        async with client:
            result = await client._poll_analyze_result('https://test.com/op/123')

        # Assert - result is an AnalyzeResult object
        self.assertIsNotNone(result.tables)
        self.assertEqual(len(result.tables), 1)
        mock_client.get.assert_called_once()

    @patch('httpx.AsyncClient')
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_poll_analyze_success_after_retries(self, mock_sleep, mock_client_class):
        """Test polling succeeds after multiple attempts."""
        mock_client = AsyncMock()

        # First two calls return "running", third returns "succeeded"
        responses = [
            Mock(status_code=200, json=lambda: {'status': 'running'}),
            Mock(status_code=200, json=lambda: {'status': 'running'}),
            Mock(status_code=200, json=lambda: {
                'status': 'succeeded',
                'createdDateTime': '2024-01-01T00:00:00Z',
                'lastUpdatedDateTime': '2024-01-01T00:00:00Z',
                'analyzeResult': {
                    'apiVersion': '2024-11-30',
                    'modelId': 'prebuilt-layout',
                    'content': '',
                    'pages': [],
                    'tables': []
                }
            })
        ]
        mock_client.get.side_effect = responses
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        async with client:
            result = await client._poll_analyze_result('https://test.com/op/123')

        # Assert - result is AnalyzeResult object
        self.assertIsNotNone(result)
        self.assertEqual(len(result.tables), 0)
        self.assertEqual(mock_client.get.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # Slept between attempts

    @patch('httpx.AsyncClient')
    async def test_poll_analyze_failed_status(self, mock_client_class):
        """Test error handling when analyze fails."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'failed',
            'error': {
                'code': 'InvalidDocument',
                'message': 'Document processing failed'
            }
        }
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        with self.assertRaises(ValueError) as context:
            async with client:
                await client._poll_analyze_result('https://test.com/op/123')

        self.assertIn("failed", str(context.exception).lower())

    @patch('httpx.AsyncClient')
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_poll_analyze_timeout(self, mock_sleep, mock_client_class):
        """Test timeout when polling exceeds max retries."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'running'}
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        with self.assertRaises(ValueError) as context:
            async with client:
                await client._poll_analyze_result(
                    'https://test.com/op/123',
                    max_retries=3,
                    poll_interval=0.1
                )

        self.assertIn("timed out", str(context.exception).lower())

    @patch('httpx.AsyncClient')
    async def test_poll_analyze_unknown_status(self, mock_client_class):
        """Test error handling for unknown status."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'status': 'unknownStatus'}
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        with self.assertRaises(ValueError) as context:
            async with client:
                await client._poll_analyze_result('https://test.com/op/123')

        self.assertIn("unknown", str(context.exception).lower())

    # ========== Table Grouping Tests ==========

    def test_group_tables_by_page_single_page(self):
        """Test grouping tables from single page."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        tables = [
            Mock(bounding_regions=[Mock(page_number=1)]),
            Mock(bounding_regions=[Mock(page_number=1)]),
            Mock(bounding_regions=[Mock(page_number=1)])
        ]

        grouped = client._group_tables_by_page(tables)

        self.assertEqual(len(grouped), 1)
        self.assertIn(1, grouped)
        self.assertEqual(len(grouped[1]), 3)

    def test_group_tables_by_page_multiple_pages(self):
        """Test grouping tables from multiple pages."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        tables = [
            Mock(bounding_regions=[Mock(page_number=1)]),
            Mock(bounding_regions=[Mock(page_number=1)]),
            Mock(bounding_regions=[Mock(page_number=2)]),
            Mock(bounding_regions=[Mock(page_number=3)]),
            Mock(bounding_regions=[Mock(page_number=3)]),
            Mock(bounding_regions=[Mock(page_number=3)])
        ]

        grouped = client._group_tables_by_page(tables)

        self.assertEqual(len(grouped), 3)
        self.assertEqual(len(grouped[1]), 2)
        self.assertEqual(len(grouped[2]), 1)
        self.assertEqual(len(grouped[3]), 3)

    @patch('src.services.azure_document_intelligence_client.logger')
    def test_group_tables_by_page_no_bounding_regions(self, mock_logger):
        """Test handling of tables without bounding regions."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        tables = [
            Mock(bounding_regions=[Mock(page_number=1)]),
            Mock(bounding_regions=None),  # No bounding regions
            Mock(bounding_regions=[Mock(page_number=2)])
        ]

        grouped = client._group_tables_by_page(tables)

        # Should skip table without bounding regions
        self.assertEqual(len(grouped), 2)
        mock_logger.warning.assert_called_once()

    # ========== Table Merging Tests ==========

    def test_merge_tables_same_headers(self):
        """Test merging tables with identical headers."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        # Create mock tables with required methods
        table1 = Mock()
        table1.get_headers.return_value = ["Name", "Value"]
        table1.get_data_rows.return_value = [["Item1", "100"]]
        table1.has_headers.return_value = True

        table2 = Mock()
        table2.get_headers.return_value = ["Name", "Value"]
        table2.get_data_rows.return_value = [["Item2", "200"]]
        table2.has_headers.return_value = True

        tables_by_page = {
            1: [table1],
            2: [table2]
        }

        merged = client._merge_tables_across_pages(tables_by_page)

        # Should merge into 1 table
        self.assertEqual(len(merged), 1)
        # Should have 2 data rows (excluding headers)
        self.assertEqual(len(merged[0].data_rows), 2)
        # Should span pages 1-2
        self.assertEqual(merged[0].start_page, 1)
        self.assertEqual(merged[0].end_page, 2)

    def test_merge_tables_different_headers(self):
        """Test tables with different headers are not merged."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        table1 = Mock()
        table1.get_headers.return_value = ["Name", "Value"]
        table1.get_data_rows.return_value = []
        table1.has_headers.return_value = True

        table2 = Mock()
        table2.get_headers.return_value = ["Product", "Price"]
        table2.get_data_rows.return_value = []
        table2.has_headers.return_value = True

        tables_by_page = {
            1: [table1],
            2: [table2]
        }

        merged = client._merge_tables_across_pages(tables_by_page)

        # Should NOT merge - different headers
        self.assertEqual(len(merged), 2)

    def test_headers_match_case_insensitive(self):
        """Test header matching is case insensitive."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        headers1 = ["Name", "Value", "Date"]
        headers2 = ["name", "value", "date"]

        self.assertTrue(client._headers_match(headers1, headers2))

    def test_headers_match_different_count(self):
        """Test headers with different counts don't match."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        headers1 = ["A", "B", "C"]
        headers2 = ["A", "B"]

        self.assertFalse(client._headers_match(headers1, headers2))

    # ========== Extract Tables Integration Tests ==========

    @patch.object(AzureDocumentIntelligenceClient, '_poll_analyze_result')
    @patch.object(AzureDocumentIntelligenceClient, '_start_analyze')
    @patch.object(AzureDocumentIntelligenceClient, '_encode_pdf_to_base64')
    async def test_extract_tables_success_with_merge(
        self, mock_encode, mock_start, mock_poll
    ):
        """Test successful table extraction with merging."""
        # Setup mocks
        mock_encode.return_value = "fake_base64"
        mock_start.return_value = "https://test.com/op/123"

        # Mock analyze result with tables
        mock_table = Mock()
        mock_table.get_headers.return_value = ["Header1", "Header2"]
        mock_table.get_data_rows.return_value = [["Data1", "Data2"]]
        mock_table.has_headers.return_value = True
        mock_table.bounding_regions = [Mock(page_number=1)]

        mock_analyze_result = Mock()
        mock_analyze_result.tables = [mock_table]

        mock_poll.return_value = mock_analyze_result

        # Execute
        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            tmp.write(b'fake pdf')
            tmp.flush()

            client = AzureDocumentIntelligenceClient(
                endpoint=self.test_endpoint,
                api_key=self.test_api_key
            )

            markdown_list, metadata = await client.extract_tables(
                pdf_path=tmp.name,
                merge_tables=True
            )

        # Assert
        self.assertIsInstance(markdown_list, list)
        self.assertGreater(len(markdown_list), 0)
        self.assertIn('table_count', metadata)
        self.assertTrue(metadata['merged'])

    @patch.object(AzureDocumentIntelligenceClient, '_poll_analyze_result')
    @patch.object(AzureDocumentIntelligenceClient, '_start_analyze')
    @patch.object(AzureDocumentIntelligenceClient, '_encode_pdf_to_base64')
    async def test_extract_tables_no_tables_found(
        self, mock_encode, mock_start, mock_poll
    ):
        """Test handling when no tables are found."""
        mock_encode.return_value = "fake_base64"
        mock_start.return_value = "https://test.com/op/123"

        mock_analyze_result = Mock()
        mock_analyze_result.tables = []  # No tables
        mock_poll.return_value = mock_analyze_result

        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            tmp.write(b'fake pdf')
            tmp.flush()

            client = AzureDocumentIntelligenceClient(
                endpoint=self.test_endpoint,
                api_key=self.test_api_key
            )

            markdown_list, metadata = await client.extract_tables(pdf_path=tmp.name)

        self.assertEqual(len(markdown_list), 0)
        self.assertEqual(metadata['table_count'], 0)

    @patch.object(AzureDocumentIntelligenceClient, '_poll_analyze_result')
    @patch.object(AzureDocumentIntelligenceClient, '_start_analyze')
    @patch.object(AzureDocumentIntelligenceClient, '_encode_pdf_to_base64')
    async def test_extract_tables_without_merge(
        self, mock_encode, mock_start, mock_poll
    ):
        """Test table extraction without merging."""
        mock_encode.return_value = "fake_base64"
        mock_start.return_value = "https://test.com/op/123"

        mock_table = Mock()
        mock_table.get_headers.return_value = ["Test"]
        mock_table.get_data_rows.return_value = []
        mock_table.bounding_regions = [Mock(page_number=1)]

        mock_analyze_result = Mock()
        mock_analyze_result.tables = [mock_table]
        mock_poll.return_value = mock_analyze_result

        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            tmp.write(b'fake pdf')
            tmp.flush()

            client = AzureDocumentIntelligenceClient(
                endpoint=self.test_endpoint,
                api_key=self.test_api_key
            )

            markdown_list, metadata = await client.extract_tables(
                pdf_path=tmp.name,
                merge_tables=False
            )

        self.assertFalse(metadata['merged'])

    @patch.object(AzureDocumentIntelligenceClient, '_poll_analyze_result')
    @patch.object(AzureDocumentIntelligenceClient, '_start_analyze')
    @patch.object(AzureDocumentIntelligenceClient, '_encode_pdf_to_base64')
    async def test_extract_tables_with_pdf_path(
        self, mock_encode, mock_start, mock_poll
    ):
        """Test extraction with PDF file path."""
        mock_encode.return_value = "encoded_content"
        mock_start.return_value = "https://test.com/op/123"

        mock_analyze_result = Mock()
        mock_analyze_result.tables = []
        mock_poll.return_value = mock_analyze_result

        with tempfile.NamedTemporaryFile(suffix='.pdf') as tmp:
            tmp.write(b'test')
            tmp.flush()

            client = AzureDocumentIntelligenceClient(
                endpoint=self.test_endpoint,
                api_key=self.test_api_key
            )

            await client.extract_tables(pdf_path=tmp.name)

        # Should call encode
        mock_encode.assert_called_once_with(tmp.name)

    @patch.object(AzureDocumentIntelligenceClient, '_poll_analyze_result')
    @patch.object(AzureDocumentIntelligenceClient, '_start_analyze')
    @patch.object(AzureDocumentIntelligenceClient, '_encode_pdf_to_base64')
    async def test_extract_tables_with_base64(
        self, mock_encode, mock_start, mock_poll
    ):
        """Test extraction with pre-encoded base64."""
        mock_start.return_value = "https://test.com/op/123"

        mock_analyze_result = Mock()
        mock_analyze_result.tables = []
        mock_poll.return_value = mock_analyze_result

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        await client.extract_tables(pdf_base64="preencoded_base64")

        # Should NOT call encode (already provided)
        mock_encode.assert_not_called()

    async def test_extract_tables_missing_both_params(self):
        """Test error when neither pdf_path nor pdf_base64 provided."""
        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        with self.assertRaises(ValueError) as context:
            await client.extract_tables()

        self.assertIn("pdf_path or pdf_base64", str(context.exception).lower())

    # ========== Health Check Tests ==========

    @patch('httpx.AsyncClient')
    async def test_health_check_success(self, mock_client_class):
        """Test successful health check."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        result = await client.health_check()

        self.assertTrue(result)

    @patch('httpx.AsyncClient')
    async def test_health_check_endpoint_reachable_404(self, mock_client_class):
        """Test health check with 404 (endpoint exists but no health route)."""
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 404
        mock_client.get.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        result = await client.health_check()

        # 404 means endpoint is reachable (just no health route)
        self.assertTrue(result)

    @patch('httpx.AsyncClient')
    async def test_health_check_failure(self, mock_client_class):
        """Test health check failure."""
        mock_client = AsyncMock()
        mock_client.get.side_effect = Exception("Connection failed")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client_class.return_value = mock_client

        client = AzureDocumentIntelligenceClient(
            endpoint=self.test_endpoint,
            api_key=self.test_api_key
        )

        result = await client.health_check()

        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
