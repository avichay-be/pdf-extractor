"""
Extract tables from PDF using Azure Document Intelligence and save as Markdown.
This script uses Azure's Document Intelligence service to extract structured data from PDFs.
"""
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import os
from pathlib import Path
from datetime import datetime


def extract_tables_to_markdown(pdf_path: str, output_path: str = None,
                               endpoint: str = None, key: str = None,
                               model_id: str = "prebuilt-layout"):
    """
    Extract tables from PDF using Azure Document Intelligence and save as markdown.

    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output markdown file (optional)
        endpoint: Azure Document Intelligence endpoint
        key: Azure Document Intelligence API key
        model_id: Model ID to use for extraction
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Generate output path if not provided
    if output_path is None:
        pdf_name = Path(pdf_path).stem
        output_path = f"{pdf_name}_azure_tables.md"

    # Initialize Azure Document Intelligence client
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )

    print(f"Extracting tables from {pdf_path} using Azure Document Intelligence...")
    print(f"Model: {model_id}")

    # Read PDF file
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # Analyze document
    print("Analyzing document...")
    poller = document_intelligence_client.begin_analyze_document(
        model_id=model_id,
        body=AnalyzeDocumentRequest(bytes_source=pdf_bytes)
    )
    result = poller.result()

    print(f"Document analyzed by model: {result.model_id}")
    print(f"Number of pages: {len(result.pages)}")

    # Start building markdown content
    markdown_content = f"# Tables extracted from {Path(pdf_path).name}\n\n"
    markdown_content += f"**Extraction Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    markdown_content += f"**Model:** {result.model_id}\n"
    markdown_content += f"**Pages:** {len(result.pages)}\n\n"
    markdown_content += "---\n\n"

    # Extract structured documents if available
    if result.documents:
        markdown_content += "## Structured Documents\n\n"
        for idx, document in enumerate(result.documents, 1):
            markdown_content += f"### Document {idx}\n\n"
            markdown_content += f"- **Type:** {document.doc_type}\n"
            markdown_content += f"- **Confidence:** {document.confidence:.2%}\n\n"

            if document.fields:
                markdown_content += "**Fields:**\n\n"
                for field_name, field in document.fields.items():
                    content = getattr(field, 'content', getattr(field, 'value', 'N/A'))
                    confidence = getattr(field, 'confidence', 0)
                    markdown_content += f"- **{field_name}:** {content} (confidence: {confidence:.2%})\n"
                markdown_content += "\n"

            markdown_content += "---\n\n"
    else:
        print("No structured documents found (this is normal for custom layout models)")

    # Extract tables
    if result.tables:
        print(f"Found {len(result.tables)} tables")
        markdown_content += "## Tables\n\n"

        for i, table in enumerate(result.tables, 1):
            markdown_content += f"### Table {i}\n\n"

            # Add metadata
            markdown_content += f"- **Rows:** {table.row_count}\n"
            markdown_content += f"- **Columns:** {table.column_count}\n"

            if table.bounding_regions:
                pages = ", ".join(str(region.page_number) for region in table.bounding_regions)
                markdown_content += f"- **Pages:** {pages}\n"

            markdown_content += "\n"

            # Build table structure
            # Create a 2D array to hold cell contents
            table_array = [['' for _ in range(table.column_count)] for _ in range(table.row_count)]

            # Fill in the cells
            for cell in table.cells:
                row = cell.row_index
                col = cell.column_index
                content = cell.content if cell.content else ''

                # Handle cell spans
                row_span = getattr(cell, 'row_span', 1) or 1
                col_span = getattr(cell, 'column_span', 1) or 1

                # Fill the primary cell
                table_array[row][col] = content

                # Mark spanned cells
                for r in range(row, min(row + row_span, table.row_count)):
                    for c in range(col, min(col + col_span, table.column_count)):
                        if r != row or c != col:
                            table_array[r][c] = '↑' if r > row else '←'

            # Convert to markdown table
            # Header row
            markdown_content += "| " + " | ".join(f"Col {i+1}" for i in range(table.column_count)) + " |\n"
            markdown_content += "|" + "|".join("---" for _ in range(table.column_count)) + "|\n"

            # Data rows
            for row in table_array:
                # Escape pipe characters in cell content
                escaped_row = [cell.replace('|', '\\|').replace('\n', ' ') for cell in row]
                markdown_content += "| " + " | ".join(escaped_row) + " |\n"

            markdown_content += "\n---\n\n"
    else:
        print("No tables found in the document")
        markdown_content += "No tables found in the document.\n\n"

    # Extract page-level content (lines and text)
    if result.pages:
        markdown_content += "## Page Content Summary\n\n"
        for page in result.pages:
            line_count = len(page.lines) if page.lines else 0
            word_count = len(page.words) if page.words else 0
            markdown_content += f"- **Page {page.page_number}:** {line_count} lines, {word_count} words\n"
        markdown_content += "\n"

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"\n✓ Tables saved to {output_path}")

    # Print summary
    print("\nSummary:")
    if result.documents:
        print(f"  Structured Documents: {len(result.documents)}")
    if result.tables:
        for i, table in enumerate(result.tables, 1):
            pages = ", ".join(str(region.page_number) for region in table.bounding_regions) if table.bounding_regions else "N/A"
            print(f"  Table {i}: {table.row_count}x{table.column_count} (Pages: {pages})")

    return output_path


if __name__ == "__main__":
    # Configuration - load from environment variables
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    model_id = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-layout")

    # Input/Output paths
    pdf_path = "data/bank_statements.pdf"
    output_path = "data/output/bank_statements_azure_tables.md"

    try:
        # Extract tables
        result_path = extract_tables_to_markdown(
            pdf_path=pdf_path,
            output_path=output_path,
            endpoint=endpoint,
            key=key,
            model_id=model_id 
        )

        print(f"\n✓ Done! Check {result_path} for the extracted tables.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
