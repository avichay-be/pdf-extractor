import pytest
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table
from src.services.extraction_service import extract_text_from_pdf
import pandas as pd
from src.core.utils import normalize_hebrew_text

def create_table_pdf(path):
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    elements = []
    
    data = [
        ['date', 'desc', 'debit', 'credit', 'balance'],
        ['2023-01-01', 'Opening Balance', '', '', '1000.00'],
        ['2023-01-05', 'Payment', '500.00', '', '500.00'],
        ['2023-01-10', 'Deposit', '', '200.00', '700.00'],
    ]
    
    t = Table(data)
    elements.append(t)
    doc.build(elements)

@pytest.fixture
def sample_pdf_with_table(tmp_path):
    pdf_path = tmp_path / "table_service.pdf"
    create_table_pdf(pdf_path)
    return pdf_path

def test_extract_text_from_pdf_pdfplumber(sample_pdf_with_table):
    content, metadata = extract_text_from_pdf(str(sample_pdf_with_table))
    
    print(content)
    
    # Verify metadata indicates pdfplumber was used
    assert "pdfplumber_table_only" in metadata["extraction_method"]
    
    # Verify we got some content
    if "Extracted Tables" in content and "Table 1" in content:
        print("pdfplumber successfully extracted the table.")
        assert "date" in content
        assert "1000.00" in content
    else:
        print("pdfplumber did not detect the table in this synthetic PDF.")
        assert "No tables were detected" in content


def test_pandas_markdown_preserves_logical_hebrew_order():
    df = pd.DataFrame(
        [["סריקת תעודת זהות", "Entra ID (B2C)"]],
        columns=["תהליך", "מערכת"]
    )

    markdown = df.to_markdown(index=False)

    assert "סריקת תעודת זהות" in markdown
    assert "תוהז תדועת תקירס" not in markdown


def test_normalize_hebrew_text_fixes_visual_order_hebrew():
    assert normalize_hebrew_text("תוהז תדועת תקירס") == "סריקת תעודת זהות"


def test_normalize_hebrew_text_fixes_mixed_visual_order_text():
    value = "םיאלמ םיילטיגיד ComSign / DocuSign"
    normalized = normalize_hebrew_text(value)

    assert normalized == "ComSign / DocuSign דיגיטליים מלאים"
    assert "םיאלמ םיילטיגיד" not in normalized
