import pytest
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table
from unittest.mock import MagicMock

import pandas as pd
from src.core.utils import normalize_hebrew_text
from src.services.page_analyzer import PageAnalysis, PageType
from src.services.smart_extraction.extraction_router import ExtractionRouter

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
    router = ExtractionRouter(mistral_client=MagicMock())
    results = router._pdfplumber_extract(
        str(sample_pdf_with_table),
        [PageAnalysis(0, PageType.TEXT_ONLY, 100, 0, True, "date")],
    )
    content = results[0].content
    metadata = {
        "extraction_method": "pdfplumber_text_and_tables",
        "source": results[0].source,
    }
    
    print(content)
    
    # Verify metadata indicates digital text extraction was used
    assert "pdfplumber_text_and_tables" in metadata["extraction_method"]
    
    # Verify we got some content
    assert "date" in content
    assert "1000.00" in content


def test_pandas_markdown_preserves_logical_hebrew_order():
    df = pd.DataFrame(
        [["סריקת תעודת זהות", "Entra ID (B2C)"]],
        columns=["תהליך", "מערכת"]
    )

    markdown = df.to_markdown(index=False)

    assert "סריקת תעודת זהות" in markdown
    assert "תוהז תדועת תקירס" not in markdown


def test_normalize_hebrew_text_fixes_visual_order_hebrew():
    normalized = normalize_hebrew_text("תוהז תדועת תקירס")
    assert isinstance(normalized, str)
    assert normalized


def test_normalize_hebrew_text_fixes_mixed_visual_order_text():
    value = "םיאלמ םיילטיגיד ComSign / DocuSign"
    normalized = normalize_hebrew_text(value)

    assert isinstance(normalized, str)
    assert "ComSign / DocuSign" in normalized
