from bm25.bm25 import BM25
from bm25.preprocess import preprocess_text
from bm25.file_handlers import read_text_file, extract_text_from_pdf, extract_text_from_url

def build_search_engine(documents, pdfs=None, urls=None):
    corpus = []
    for doc_path in documents:
        text = read_text_file(doc_path)
        corpus.append(preprocess_text(text))
    if pdfs:
        for pdf_path in pdfs:
            text = extract_text_from_pdf(pdf_path)
            corpus.append(preprocess_text(text))
    if urls:
        for url in urls:
            text = extract_text_from_url(url)
            corpus.append(preprocess_text(text))
    return BM25(corpus)
