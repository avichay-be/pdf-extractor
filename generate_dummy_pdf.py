from pypdf import PdfWriter
from pathlib import Path

def create_dummy_pdf(path):
    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)  # A4 size
    with open(path, "wb") as f:
        writer.write(f)
    print(f"Created dummy PDF at {path}")

if __name__ == "__main__":
    output_dir = Path("tests/test_pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)
    create_dummy_pdf(output_dir / "dummy.pdf")
