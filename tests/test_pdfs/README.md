# Test PDFs Directory

Place your test PDF files in this directory for integration testing.

## Usage

1. Copy PDF files here:
   ```bash
   cp your_test_file.pdf tests/test_pdfs/
   ```

2. Run integration tests:
   ```bash
   pytest tests/integration/ -v
   ```

## Supported Files

- Any valid PDF files
- Recommended: PDFs with different characteristics
  - Small PDFs (< 10 pages)
  - Large PDFs (> 10 pages)
  - PDFs with outlines/bookmarks
  - PDFs without outlines
  - Hebrew/English content
  - Financial documents

## Examples

Good test cases:
- `simple_report.pdf` - Small 5-page report
- `financial_statement.pdf` - PDF with Hebrew content and outlines
- `large_document.pdf` - 100+ pages for performance testing
- `no_outlines.pdf` - PDF without bookmarks

## Note

- Test PDFs are NOT included in git (see `.gitignore`)
- Add your own PDFs for testing
- Integration tests will be skipped if no PDFs are found
