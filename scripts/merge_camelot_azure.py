"""
Extract tables from PDF using both Camelot and Azure Document Intelligence,
merge the results with Camelot taking priority on conflicts.

Usage:
    python merge_camelot_azure.py [input_dir] [output_dir]

Environment Variables:
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT - Azure endpoint URL
    AZURE_DOCUMENT_INTELLIGENCE_KEY - Azure API key
"""
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import camelot
import os
import sys
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def extract_with_camelot(pdf_path: str):
    """Extract tables using Camelot."""
    print("  [Camelot] Extracting tables...")
    tables = []

    try:
        # Try lattice method first (for tables with borders)
        camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')

        if len(camelot_tables) == 0:
            # Fallback to stream method
            camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')

        for i, table in enumerate(camelot_tables, 1):
            tables.append({
                'source': 'camelot',
                'table_num': i,
                'page': table.page,
                'accuracy': table.accuracy,
                'df': table.df,
                'shape': table.df.shape
            })

        print(f"  [Camelot] Found {len(tables)} tables")
        return tables

    except Exception as e:
        print(f"  [Camelot] Error: {e}")
        return []


def extract_with_azure(pdf_path: str, endpoint: str, key: str, model_id: str = "prebuilt-layout"):
    """Extract tables using Azure Document Intelligence."""
    print("  [Azure] Extracting tables...")
    tables = []

    try:
        # Initialize client
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

        # Read PDF
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Analyze
        poller = client.begin_analyze_document(
            model_id=model_id,
            body=AnalyzeDocumentRequest(bytes_source=pdf_bytes)
        )
        result = poller.result()

        if result.tables:
            for i, table in enumerate(result.tables, 1):
                # Build dataframe from Azure table
                table_array = [['' for _ in range(table.column_count)] for _ in range(table.row_count)]

                for cell in table.cells:
                    table_array[cell.row_index][cell.column_index] = cell.content if cell.content else ''

                df = pd.DataFrame(table_array)

                # Get page number
                page = table.bounding_regions[0].page_number if table.bounding_regions else 0

                tables.append({
                    'source': 'azure',
                    'table_num': i,
                    'page': page,
                    'df': df,
                    'shape': df.shape
                })

        print(f"  [Azure] Found {len(tables)} tables")
        return tables

    except Exception as e:
        print(f"  [Azure] Error: {e}")
        return []


def merge_tables(camelot_tables, azure_tables):
    """
    Merge Camelot and Azure tables, preferring Camelot on conflicts.
    Returns merged list of tables.
    """
    merged = []

    # Group by page
    camelot_by_page = {}
    for table in camelot_tables:
        page = table['page']
        if page not in camelot_by_page:
            camelot_by_page[page] = []
        camelot_by_page[page].append(table)

    azure_by_page = {}
    for table in azure_tables:
        page = table['page']
        if page not in azure_by_page:
            azure_by_page[page] = []
        azure_by_page[page].append(table)

    all_pages = set(camelot_by_page.keys()) | set(azure_by_page.keys())

    for page in sorted(all_pages):
        camelot_page_tables = camelot_by_page.get(page, [])
        azure_page_tables = azure_by_page.get(page, [])

        # If Camelot found tables on this page, use them
        if camelot_page_tables:
            for table in camelot_page_tables:
                table['used_source'] = 'camelot (preferred)'
                merged.append(table)

        # If no Camelot tables but Azure found some, use Azure
        elif azure_page_tables:
            for table in azure_page_tables:
                table['used_source'] = 'azure (fallback)'
                merged.append(table)

    return merged


def save_merged_to_markdown(merged_tables, output_path, pdf_name):
    """Save merged tables to markdown."""
    markdown_content = f"# Tables extracted from {pdf_name}\n\n"
    markdown_content += f"**Extraction Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    markdown_content += f"**Method:** Camelot (preferred) + Azure (fallback)\n\n"
    markdown_content += "---\n\n"

    if not merged_tables:
        markdown_content += "No tables found.\n"
    else:
        for i, table in enumerate(merged_tables, 1):
            markdown_content += f"## Table {i}\n\n"
            markdown_content += f"- **Source:** {table['used_source']}\n"
            markdown_content += f"- **Page:** {table['page']}\n"
            markdown_content += f"- **Shape:** {table['shape'][0]} rows × {table['shape'][1]} columns\n"

            if 'accuracy' in table:
                markdown_content += f"- **Accuracy:** {table['accuracy']:.2f}%\n"

            markdown_content += "\n"

            # Convert dataframe to markdown
            df = table['df']
            markdown_content += df.to_markdown(index=False)
            markdown_content += "\n\n---\n\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    return output_path


def process_pdf_with_merge(pdf_path, output_path, endpoint, key, model_id="prebuilt-layout"):
    """Process a single PDF with both Camelot and Azure, merge results."""
    print(f"\nProcessing: {pdf_path}")

    # Extract with both methods
    camelot_tables = extract_with_camelot(pdf_path)
    azure_tables = extract_with_azure(pdf_path, endpoint, key, model_id)

    # Merge results
    print("  [Merge] Combining results...")
    merged_tables = merge_tables(camelot_tables, azure_tables)

    # Report
    camelot_count = len([t for t in merged_tables if 'camelot' in t['used_source']])
    azure_count = len([t for t in merged_tables if 'azure' in t['used_source']])

    print(f"  [Merge] Result: {len(merged_tables)} tables total")
    print(f"          - {camelot_count} from Camelot")
    print(f"          - {azure_count} from Azure")

    # Save
    save_merged_to_markdown(merged_tables, output_path, Path(pdf_path).name)
    print(f"  [Save] Saved to: {output_path}")

    return {
        'total': len(merged_tables),
        'camelot': camelot_count,
        'azure': azure_count
    }


if __name__ == "__main__":
    # Configuration from environment or defaults
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    model_id = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-layout")

    # Parse command line arguments
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "data/bank_statements"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output/merged_tables"

    print(f"Configuration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Azure model: {model_id}\n")

    # Find all PDF files
    pdf_files = glob.glob(f"{input_dir}/**/*.pdf", recursive=True)

    if not pdf_files:
        print(f"No PDF files found in {input_dir}/")
        exit(1)

    print(f"Found {len(pdf_files)} PDF files to process\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Statistics
    successful = 0
    failed = 0
    failed_files = []
    total_stats = {'total': 0, 'camelot': 0, 'azure': 0}

    # Process each file
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'='*80}")
        print(f"File {idx}/{len(pdf_files)}")
        print(f"{'='*80}")

        try:
            # Generate output path
            pdf_name = Path(pdf_path).stem
            safe_name = pdf_name.replace(" ", "_").replace("/", "_")
            output_path = os.path.join(output_dir, f"{safe_name}_merged.md")

            # Process
            stats = process_pdf_with_merge(pdf_path, output_path, endpoint, key, model_id)

            # Update totals
            total_stats['total'] += stats['total']
            total_stats['camelot'] += stats['camelot']
            total_stats['azure'] += stats['azure']

            successful += 1
            print(f"✓ Success!")

        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
            failed_files.append((pdf_path, str(e)))

    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {successful}/{len(pdf_files)}")
    print(f"Failed: {failed}")
    print(f"\nTables extracted:")
    print(f"  Total tables: {total_stats['total']}")
    print(f"  From Camelot: {total_stats['camelot']} ({100*total_stats['camelot']/(total_stats['total'] or 1):.1f}%)")
    print(f"  From Azure:   {total_stats['azure']} ({100*total_stats['azure']/(total_stats['total'] or 1):.1f}%)")

    if failed_files:
        print(f"\nFailed files:")
        for file, error in failed_files:
            print(f"  - {file}")
            print(f"    {error[:80]}...")

    print(f"\nAll output saved to: {output_dir}/")
    print(f"✓ Done!")
