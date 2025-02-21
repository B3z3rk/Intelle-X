import requests
from fpdf import FPDF
from bm25.file_handlers import read_text_file, extract_text_from_pdf, extract_text_from_url

def test_read_text_file(tmp_path):
    file_path = tmp_path / "test.txt"
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write("Hello, world!")
    assert read_text_file(file_path) == "Hello, world!"

def test_extract_text_from_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    
    # Create a valid PDF file
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Hello, world!", ln=True, align="C")
    pdf.output(pdf_path)

    # Extract text and assert
    extracted_text = extract_text_from_pdf(pdf_path)
    assert "Hello, world!" in extracted_text

def test_extract_text_from_url(monkeypatch):
    class MockResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    def mock_get(*args, **kwargs):
        return MockResponse(b"<html><body>Hello, world!</body></html>")

    monkeypatch.setattr(requests, 'get', mock_get)
    assert extract_text_from_url("https://example.com") == "Hello, world!"

