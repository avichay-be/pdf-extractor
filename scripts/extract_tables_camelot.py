"""
Extract tables from PDF using Camelot and save as Markdown.
"""
import camelot
import os
from pathlib import Path


def extract_tables_to_markdown(pdf_path: str, output_path: str = None, method: str = 'lattice'):
    """
    Extract tables from PDF using Camelot and save as markdown.

    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output markdown file (optional)
        method: 'lattice' for tables with borders, 'stream' for tables without borders
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Generate output path if not provided
    if output_path is None:
        pdf_name = Path(pdf_path).stem
        output_path = f"{pdf_name}_tables.md"

    print(f"Extracting tables from {pdf_path} using {method} method...")

    # Extract tables
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor=method)
        print(f"Found {len(tables)} tables")
    except Exception as e:
        print(f"Error with {method} method: {e}")
        # Try alternative method
        alt_method = 'stream' if method == 'lattice' else 'lattice'
        print(f"Trying {alt_method} method...")
        tables = camelot.read_pdf(pdf_path, pages='all', flavor=alt_method)
        print(f"Found {len(tables)} tables with {alt_method} method")

    if len(tables) == 0:
        print("No tables found in the PDF")
        return

    # Convert to markdown
    markdown_content = f"# Tables extracted from {Path(pdf_path).name}\n\n"

    for i, table in enumerate(tables, 1):
        markdown_content += f"## Table {i} (Page {table.page})\n\n"
        markdown_content += f"Accuracy: {table.accuracy:.2f}%\n\n"

        # Convert table to markdown format
        df = table.df
        markdown_content += df.to_markdown(index=False)
        markdown_content += "\n\n---\n\n"

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"Tables saved to {output_path}")

    # Print summary
    print("\nSummary:")
    for i, table in enumerate(tables, 1):
        print(f"  Table {i}: Page {table.page}, Shape {table.df.shape}, Accuracy {table.accuracy:.2f}%")


if __name__ == "__main__":
    # Extract from bank statements
    pdf_path = "data/bank_statements.pdf"
    output_path = "bank_statements_tables.md"

    # Try lattice method first (for tables with borders)
    extract_tables_to_markdown(pdf_path, output_path, method='lattice')

    print(f"\nDone! Check {output_path} for the extracted tables.")
