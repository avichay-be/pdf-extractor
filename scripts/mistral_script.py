"""
Extract tables from PDF using Mistral Document AI and save to markdown.

This script uses the existing Mistral client to extract content and
concatenates all tables into a single markdown file.

Usage:
    python mistral.py <input_pdf> [output_md]
    python mistral.py data/bank_statements.pdf output/mistral_tables.md
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Import the existing Mistral client
from src.services.mistral_client import MistralDocumentClient
from config import settings

load_dotenv()


def extract_tables_from_markdown(markdown_content: str):
    """
    Extract tables from markdown content.

    Looks for markdown table patterns (pipes and dashes).
    Returns list of tables with metadata.
    """
    tables = []
    lines = markdown_content.split('\n')

    current_table = []
    in_table = False
    table_num = 0

    for i, line in enumerate(lines):
        # Check if line looks like a table row (contains pipes)
        if '|' in line and line.strip():
            if not in_table:
                # Starting a new table
                in_table = True
                table_num += 1
                current_table = [line]
            else:
                # Continue current table
                current_table.append(line)
        else:
            # Not a table line
            if in_table and current_table:
                # End of table - save it
                table_content = '\n'.join(current_table)

                # Count rows and columns
                table_lines = [l for l in current_table if l.strip() and not l.strip().startswith('|---')]
                num_rows = len(table_lines)
                num_cols = len(table_lines[0].split('|')) - 2 if table_lines else 0  # -2 for leading/trailing pipes

                tables.append({
                    'number': table_num,
                    'content': table_content,
                    'rows': num_rows,
                    'columns': num_cols,
                    'line_start': i - len(current_table),
                    'line_end': i
                })

                current_table = []
                in_table = False

    # Handle case where document ends with a table
    if in_table and current_table:
        table_content = '\n'.join(current_table)
        table_lines = [l for l in current_table if l.strip() and not l.strip().startswith('|---')]
        num_rows = len(table_lines)
        num_cols = len(table_lines[0].split('|')) - 2 if table_lines else 0

        tables.append({
            'number': table_num,
            'content': table_content,
            'rows': num_rows,
            'columns': num_cols,
            'line_start': len(lines) - len(current_table),
            'line_end': len(lines)
        })

    return tables


def save_tables_to_markdown(tables, output_path, pdf_name, full_markdown=None):
    """
    Save extracted tables to a markdown file.

    Args:
        tables: List of table dicts with content and metadata
        output_path: Path to output markdown file
        pdf_name: Name of source PDF
        full_markdown: Optional full markdown content for context
    """
    markdown_content = f"# Tables extracted from {pdf_name}\n\n"
    markdown_content += f"**Extraction Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    markdown_content += f"**Extraction Method:** Mistral Document AI\n"
    markdown_content += f"**Model:** {settings.MISTRAL_MODEL}\n"
    markdown_content += f"**Total Tables Found:** {len(tables)}\n\n"
    markdown_content += "---\n\n"

    if not tables:
        markdown_content += "No tables found in the document.\n\n"

        if full_markdown:
            markdown_content += "## Full Document Content\n\n"
            markdown_content += full_markdown
    else:
        for i, table in enumerate(tables, 1):
            markdown_content += f"## Table {i}\n\n"
            markdown_content += f"- **Dimensions:** {table['rows']} rows � {table['columns']} columns\n"
            markdown_content += f"- **Location:** Lines {table['line_start']}-{table['line_end']}\n\n"
            markdown_content += table['content']
            markdown_content += "\n\n---\n\n"

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    return output_path


async def process_pdf_with_mistral(pdf_path: str, output_path: str):
    """
    Process PDF with Mistral and extract tables.

    Args:
        pdf_path: Path to input PDF
        output_path: Path to output markdown file
    """
    print(f"\nProcessing: {pdf_path}")
    print(f"Output: {output_path}\n")

    # Initialize Mistral client
    print("[1/4] Initializing Mistral client...")
    async with MistralDocumentClient(
        api_key=settings.AZURE_API_KEY,
        api_url=settings.MISTRAL_API_URL,
        model=settings.MISTRAL_MODEL
    ) as mistral_client:

        # Extract content from PDF
        print("[2/4] Extracting content with Mistral...")
        try:
            markdown_content, _ = await mistral_client.process_document(
                pdf_path=pdf_path,
                enable_validation=False  # Disable validation for faster extraction
            )

            print(f"   Successfully extracted {len(markdown_content)} characters")

        except Exception as e:
            print(f"   Error extracting content: {e}")
            raise

        # Extract tables from markdown
        print("[3/4] Extracting tables from markdown...")
        tables = extract_tables_from_markdown(markdown_content)
        print(f"   Found {len(tables)} tables")

        if tables:
            for table in tables:
                print(f"    - Table {table['number']}: {table['rows']}�{table['columns']}")

        # Save to markdown file
        print("[4/4] Saving tables to markdown...")
        output_file = save_tables_to_markdown(
            tables,
            output_path,
            Path(pdf_path).name,
            full_markdown=markdown_content if not tables else None
        )
        print(f"   Saved to: {output_file}")

    return {
        'tables_found': len(tables),
        'total_rows': sum(t['rows'] for t in tables),
        'total_cells': sum(t['rows'] * t['columns'] for t in tables)
    }


async def process_multiple_pdfs(input_dir: str, output_dir: str):
    """
    Process multiple PDFs from a directory.

    Args:
        input_dir: Directory containing PDFs
        output_dir: Directory for output markdown files
    """
    from pathlib import Path
    import glob

    # Find all PDFs
    pdf_files = glob.glob(f"{input_dir}/**/*.pdf", recursive=True)

    if not pdf_files:
        print(f"No PDF files found in {input_dir}/")
        return

    print(f"Found {len(pdf_files)} PDF files to process\n")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Process each file
    successful = 0
    failed = 0
    total_stats = {'tables': 0, 'rows': 0, 'cells': 0}

    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n{'='*80}")
        print(f"File {idx}/{len(pdf_files)}")
        print(f"{'='*80}")

        try:
            # Generate output path
            pdf_name = Path(pdf_path).stem
            safe_name = pdf_name.replace(" ", "_").replace("/", "_")
            output_path = Path(output_dir) / f"{safe_name}_mistral.md"

            # Process
            stats = await process_pdf_with_mistral(pdf_path, str(output_path))

            # Update totals
            total_stats['tables'] += stats['tables_found']
            total_stats['rows'] += stats['total_rows']
            total_stats['cells'] += stats['total_cells']

            successful += 1
            print(f" Success!")

        except Exception as e:
            print(f" Error: {e}")
            failed += 1

    # Print summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Files processed: {successful}/{len(pdf_files)}")
    print(f"Failed: {failed}")
    print(f"\nTables extracted:")
    print(f"  Total tables: {total_stats['tables']}")
    print(f"  Total rows: {total_stats['rows']}")
    print(f"  Total cells: {total_stats['cells']}")
    print(f"\nOutput saved to: {output_dir}/")


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file:  python mistral.py <input_pdf> [output_md]")
        print("  Directory:    python mistral.py <input_dir> <output_dir> --batch")
        print("\nExamples:")
        print("  python mistral.py data/bank_statements.pdf output/tables.md")
        print("  python mistral.py data/bank_statements output/mistral_tables --batch")
        sys.exit(1)

    # Check if batch mode
    is_batch = '--batch' in sys.argv or '-b' in sys.argv

    if is_batch:
        # Batch mode - process directory
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else "output/mistral_tables"

        print(f"Batch Mode")
        print(f"  Input directory: {input_dir}")
        print(f"  Output directory: {output_dir}")

        await process_multiple_pdfs(input_dir, output_dir)

    else:
        # Single file mode
        pdf_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else f"{Path(pdf_path).stem}_mistral_tables.md"

        stats = await process_pdf_with_mistral(pdf_path, output_path)

        print(f"\n Done!")
        print(f"  Tables found: {stats['tables_found']}")
        print(f"  Total rows: {stats['total_rows']}")
        print(f"  Total cells: {stats['total_cells']}")


if __name__ == "__main__":
    asyncio.run(main())
