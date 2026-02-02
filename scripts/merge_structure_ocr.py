"""
Extract table STRUCTURE from Camelot/pdfplumber and fill with DATA from OCR.

This solves the problem where:
- OCR reads text accurately but misaligns columns
- Camelot/pdfplumber identifies table structure correctly but may miss text

Strategy:
1. Use Camelot to get table structure (cell bounding boxes)
2. Use Azure Document Intelligence to get text with coordinates
3. Map OCR text to table cells based on spatial overlap
4. Build final table with correct structure + accurate content

Usage:
    python merge_structure_ocr.py [input_pdf] [output_md]
"""
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
import camelot
import pdfplumber
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def get_bbox_overlap(bbox1, bbox2):
    """
    Calculate intersection area of two bounding boxes.

    Bounding box format: (x1, y1, x2, y2) where:
    - (x1, y1) is top-left corner
    - (x2, y2) is bottom-right corner
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    return 0


def extract_table_structure_camelot(pdf_path: str):
    """
    Extract table structure using Camelot.
    Returns list of tables with cell bounding boxes.
    """
    print("  [Camelot] Extracting table structure...")
    tables = []

    try:
        # Try lattice method first (for tables with borders)
        camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')

        if len(camelot_tables) == 0:
            # Fallback to stream method
            camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')

        for i, table in enumerate(camelot_tables, 1):
            # Get cell coordinates from Camelot
            cells = []
            for cell in table.cells:
                cells.append({
                    'row': cell[0],
                    'col': cell[1],
                    'bbox': cell[2],  # (x1, y1, x2, y2)
                    'text': ''  # Will be filled from OCR
                })

            tables.append({
                'source': 'camelot',
                'table_num': i,
                'page': table.page,
                'accuracy': table.accuracy,
                'cells': cells,
                'shape': table.df.shape
            })

        print(f"  [Camelot] Found {len(tables)} tables with structure")
        return tables

    except Exception as e:
        print(f"  [Camelot] Error: {e}")
        return []


def extract_table_structure_pdfplumber(pdf_path: str):
    """
    Extract table structure using pdfplumber (alternative to Camelot).
    Returns list of tables with cell bounding boxes.
    """
    print("  [pdfplumber] Extracting table structure...")
    tables = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_tables = page.extract_tables()

                if page_tables:
                    for table_num, table_data in enumerate(page_tables, 1):
                        # Get table settings to extract cell positions
                        table_settings = page.find_tables()[table_num - 1]

                        cells = []
                        for row_idx, row in enumerate(table_data):
                            for col_idx, cell_text in enumerate(row):
                                # Estimate cell bbox based on table bbox and cell position
                                # This is an approximation
                                table_bbox = table_settings.bbox
                                cell_width = (table_bbox[2] - table_bbox[0]) / len(row)
                                cell_height = (table_bbox[3] - table_bbox[1]) / len(table_data)

                                cell_bbox = (
                                    table_bbox[0] + col_idx * cell_width,
                                    table_bbox[1] + row_idx * cell_height,
                                    table_bbox[0] + (col_idx + 1) * cell_width,
                                    table_bbox[1] + (row_idx + 1) * cell_height
                                )

                                cells.append({
                                    'row': row_idx,
                                    'col': col_idx,
                                    'bbox': cell_bbox,
                                    'text': cell_text or ''
                                })

                        tables.append({
                            'source': 'pdfplumber',
                            'table_num': len(tables) + 1,
                            'page': page_num,
                            'cells': cells,
                            'shape': (len(table_data), len(table_data[0]) if table_data else 0)
                        })

        print(f"  [pdfplumber] Found {len(tables)} tables with structure")
        return tables

    except Exception as e:
        print(f"  [pdfplumber] Error: {e}")
        return []


def extract_ocr_content_azure(pdf_path: str, endpoint: str, key: str, model_id: str = "prebuilt-layout"):
    """
    Extract text content with coordinates using Azure Document Intelligence.
    Returns list of words/lines with bounding boxes per page.
    """
    print("  [Azure OCR] Extracting text content...")

    try:
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        poller = client.begin_analyze_document(
            model_id=model_id,
            body=AnalyzeDocumentRequest(bytes_source=pdf_bytes)
        )
        result = poller.result()

        # Extract words with coordinates per page
        pages_content = []

        for page in result.pages:
            words = []

            if page.words:
                for word in page.words:
                    if word.polygon and len(word.polygon) >= 4:
                        # Convert polygon to bbox (x1, y1, x2, y2)
                        x_coords = [p.x for p in word.polygon]
                        y_coords = [p.y for p in word.polygon]
                        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

                        words.append({
                            'text': word.content,
                            'bbox': bbox,
                            'confidence': word.confidence if hasattr(word, 'confidence') else 1.0
                        })

            pages_content.append({
                'page': page.page_number,
                'words': words
            })

        print(f"  [Azure OCR] Extracted {len(pages_content)} pages with {sum(len(p['words']) for p in pages_content)} words")
        return pages_content

    except Exception as e:
        print(f"  [Azure OCR] Error: {e}")
        return []


def map_ocr_to_table_structure(table_structure, ocr_content):
    """
    Map OCR words to table cells based on spatial overlap.

    Args:
        table_structure: Table with cell bounding boxes (from Camelot/pdfplumber)
        ocr_content: Words with coordinates (from Azure OCR)

    Returns:
        Table structure filled with OCR content
    """
    print("  [Mapping] Matching OCR content to table structure...")

    # Get OCR words for this page
    page_num = table_structure['page']
    page_words = []

    for page_content in ocr_content:
        if page_content['page'] == page_num:
            page_words = page_content['words']
            break

    if not page_words:
        print(f"    Warning: No OCR content found for page {page_num}")
        return table_structure

    # Map each word to the best matching cell
    for cell in table_structure['cells']:
        cell_bbox = cell['bbox']
        matched_words = []

        for word in page_words:
            word_bbox = word['bbox']
            overlap = get_bbox_overlap(cell_bbox, word_bbox)

            # If there's significant overlap, add this word to the cell
            if overlap > 0:
                word_area = (word_bbox[2] - word_bbox[0]) * (word_bbox[3] - word_bbox[1])
                overlap_ratio = overlap / word_area if word_area > 0 else 0

                # Require at least 50% of word to be in cell
                if overlap_ratio > 0.5:
                    matched_words.append({
                        'text': word['text'],
                        'overlap': overlap,
                        'confidence': word.get('confidence', 1.0)
                    })

        # Sort words by position (left to right, top to bottom)
        matched_words.sort(key=lambda w: (cell_bbox[1], cell_bbox[0]))

        # Combine matched words into cell text
        cell['text'] = ' '.join(w['text'] for w in matched_words)
        cell['word_count'] = len(matched_words)
        cell['avg_confidence'] = sum(w['confidence'] for w in matched_words) / len(matched_words) if matched_words else 0

    # Count how many cells got content
    filled_cells = sum(1 for cell in table_structure['cells'] if cell['text'])
    total_cells = len(table_structure['cells'])

    print(f"    Filled {filled_cells}/{total_cells} cells ({100*filled_cells/total_cells:.1f}%)")

    return table_structure


def build_dataframe_from_mapped_table(table):
    """Convert mapped table structure to pandas DataFrame."""
    if not table['cells']:
        return pd.DataFrame()

    # Get table dimensions
    max_row = max(cell['row'] for cell in table['cells'])
    max_col = max(cell['col'] for cell in table['cells'])

    # Initialize empty table
    data = [['' for _ in range(max_col + 1)] for _ in range(max_row + 1)]

    # Fill with cell content
    for cell in table['cells']:
        data[cell['row']][cell['col']] = cell['text']

    return pd.DataFrame(data)


def save_to_markdown(tables, output_path, pdf_name):
    """Save merged tables to markdown."""
    markdown_content = f"# Tables extracted from {pdf_name}\n\n"
    markdown_content += f"**Extraction Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    markdown_content += f"**Method:** Structure from Camelot/pdfplumber + Content from Azure OCR\n\n"
    markdown_content += "---\n\n"

    if not tables:
        markdown_content += "No tables found.\n"
    else:
        for i, table in enumerate(tables, 1):
            markdown_content += f"## Table {i}\n\n"
            markdown_content += f"- **Source:** {table['source']}\n"
            markdown_content += f"- **Page:** {table['page']}\n"
            markdown_content += f"- **Shape:** {table['shape'][0]} rows × {table['shape'][1]} columns\n"

            if 'accuracy' in table:
                markdown_content += f"- **Structure accuracy:** {table['accuracy']:.2f}%\n"

            # Calculate content metrics
            filled_cells = sum(1 for cell in table['cells'] if cell['text'])
            total_cells = len(table['cells'])
            avg_confidence = sum(cell.get('avg_confidence', 0) for cell in table['cells']) / total_cells if total_cells > 0 else 0

            markdown_content += f"- **Content fill rate:** {100*filled_cells/total_cells:.1f}%\n"
            markdown_content += f"- **Average OCR confidence:** {100*avg_confidence:.1f}%\n"
            markdown_content += "\n"

            # Convert to dataframe and markdown
            df = build_dataframe_from_mapped_table(table)
            markdown_content += df.to_markdown(index=False)
            markdown_content += "\n\n---\n\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    return output_path


def process_pdf(pdf_path, output_path, endpoint, key, use_pdfplumber=False):
    """Process a PDF: extract structure + OCR content, then merge."""
    print(f"\nProcessing: {pdf_path}")

    # Step 1: Extract table structure
    if use_pdfplumber:
        tables_structure = extract_table_structure_pdfplumber(pdf_path)
    else:
        tables_structure = extract_table_structure_camelot(pdf_path)

        # Fallback to pdfplumber if Camelot fails
        if not tables_structure:
            print("  Falling back to pdfplumber...")
            tables_structure = extract_table_structure_pdfplumber(pdf_path)

    if not tables_structure:
        print("  No table structure found!")
        return None

    # Step 2: Extract OCR content
    ocr_content = extract_ocr_content_azure(pdf_path, endpoint, key)

    if not ocr_content:
        print("  No OCR content found!")
        return None

    # Step 3: Map OCR to table structure
    merged_tables = []
    for table in tables_structure:
        mapped_table = map_ocr_to_table_structure(table, ocr_content)
        merged_tables.append(mapped_table)

    # Step 4: Save results
    save_to_markdown(merged_tables, output_path, Path(pdf_path).name)
    print(f"  [Save] Saved to: {output_path}")

    return {
        'tables': len(merged_tables),
        'total_cells': sum(len(t['cells']) for t in merged_tables),
        'filled_cells': sum(sum(1 for c in t['cells'] if c['text']) for t in merged_tables)
    }


if __name__ == "__main__":
    # Configuration
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

    # Parse arguments
    

    pdf_path = "data/bank_statements/פועלים/עוש פועלים י.הווי קפיטל.pdf"
    output_path = "data/output/פועלים_merged_tables.md"

    print(f"Configuration:")
    print(f"  Input PDF: {pdf_path}")
    print(f"  Output: {output_path}")
    print(f"  Structure: Camelot (with pdfplumber fallback)")
    print(f"  OCR: Azure Document Intelligence\n")

    # Process
    try:
        stats = process_pdf(pdf_path, output_path, endpoint, key)

        if stats:
            print(f"\n✓ Success!")
            print(f"  Tables: {stats['tables']}")
            print(f"  Cells filled: {stats['filled_cells']}/{stats['total_cells']} ({100*stats['filled_cells']/stats['total_cells']:.1f}%)")
        else:
            print("\n✗ Failed to process PDF")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
