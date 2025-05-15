import pdfplumber
import requests
from bs4 import BeautifulSoup
from collections import Counter, defaultdict

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        if content.startswith('https') or content.endswith(('.com', '.gov', '.org')):
            return extract_text_from_url(content)
        return content


'''def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text'''


def extract_text_from_pdf(pdf_path):
    repeated_lines = Counter()
    position_map = defaultdict(list)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words()
            for word in words:
                top = round(word['top']) 
                text = word['text'].strip()
                if not text:
                    continue
        
                position_map[(top, text)].append(page.page_number)
    

    for (top, text), pages in position_map.items():
        if len(pages) > 1:
            repeated_lines[(top, text)] = len(pages)

    clean_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            lines = page.extract_text().splitlines()
            filtered = []
            for line in lines:
                found = False
                for (top, rep_text), count in repeated_lines.items():
                    if line.strip() == rep_text:
                        found = True
                        break
                if not found:
                    filtered.append(line)
            clean_text += "\n".join(filtered) + "\n"

    return clean_text.strip()


def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

