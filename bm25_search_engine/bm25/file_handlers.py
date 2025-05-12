import pdfplumber
import requests
from bs4 import BeautifulSoup

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        if content.startswith('https') or content.endswith(('.com', '.gov', '.org')):
            return extract_text_from_url(content)
        return content


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

