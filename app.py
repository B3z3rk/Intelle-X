from flask import Flask, render_template, request, redirect, url_for
import os
from functools import lru_cache
from bm25_search_engine.bm25.bm25 import BM25
from bm25_search_engine.bm25.preprocess import preprocess_text
from bm25_search_engine.bm25.file_handlers import read_text_file, extract_text_from_pdf
from transformers import BartForConditionalGeneration, BartTokenizer, pipeline
from sentence_transformers import CrossEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Initialize NLTK
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Global Variables ---
search_engine = None
documents_content = []
uploaded_files = []
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Model Initialization ---
model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --- BM25 Improvements ---
SYNONYM_MAP = {
    "study": ["research", "experiment"],
    "result": ["finding", "outcome"],
    "theory": ["model", "framework"]
}

def dynamic_bm25_params(doc_lengths):
    avg_len = sum(doc_lengths) / len(doc_lengths)
    return {
        'k1': max(1.2, min(2.0, 1.8 * (avg_len / 500))),
        'b': 0.75 if avg_len > 1000 else 0.65
    }

def expand_query(query):
    terms = preprocess_text(query).split()
    expanded = []
    for term in terms:
        expanded.append(term)
        expanded.extend(SYNONYM_MAP.get(term, []))
    return " ".join(expanded)

@lru_cache(maxsize=100)
def cached_search(query, corpus_indices):
    corpus = [documents_content[i] for i in corpus_indices]
    doc_lengths = [len(doc.split()) for doc in corpus]
    params = dynamic_bm25_params(doc_lengths)
    engine = BM25(corpus, k1=params['k1'], b=params['b'])
    return engine.rank_documents(expand_query(preprocess_text(query)))

# --- Core Functions ---
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def highlight_matches(text, query):
    query_terms = [stemmer.stem(term) for term in word_tokenize(query.lower()) if term not in stop_words]
    matching_sections = []
    for sentence in text.split('. '):
        stemmed_sentence = [stemmer.stem(word) for word in word_tokenize(sentence.lower())]
        if any(term in stemmed_sentence for term in query_terms):
            for term in query_terms:
                original_term = next((word for word in word_tokenize(sentence.lower()) 
                                    if stemmer.stem(word) == term), None)
                if original_term:
                    sentence = sentence.replace(original_term, f"<strong>{original_term}</strong>")
            matching_sections.append(sentence)
    return matching_sections

def interpret_answer(query, bm25_results):
    context = " ".join(bm25_results)
    inputs = tokenizer.encode(f"question: {query} context: {context}", 
                            return_tensors="pt", 
                            max_length=512, 
                            truncation=True)
    outputs = model.generate(inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    global search_engine, documents_content, uploaded_files

    if request.method == 'POST':
        if 'files' in request.files:
            # File upload handling (unchanged)
            files = request.files.getlist('files')
            corpus = []
            documents_content = []
            uploaded_files = []
            for file in files:
                if file.filename == '':
                    continue
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                text = read_text_file(file_path) if file.filename.endswith('.txt') else extract_text_from_pdf(file_path)
                chunks = chunk_text(text) if len(text.split()) > 1000 else [text]
                for i, chunk in enumerate(chunks):
                    corpus.append(preprocess_text(chunk))
                    documents_content.append(chunk)
                    uploaded_files.append({
                        'name': f"{file.filename} (chunk {i+1})" if len(chunks) > 1 else file.filename,
                        'content': chunk
                    })
            search_engine = True  # Flag that documents are loaded
            return redirect(url_for('index'))

        if 'query' in request.form:
            query = request.form['query']
            selected_files = request.form.getlist('selected_files')
            
            if not search_engine:
                return render_template('index.html', error="No documents uploaded yet.", uploaded_files=uploaded_files)
            
            corpus_indices = tuple(i for i, f in enumerate(uploaded_files) if f['name'] in selected_files)
            if not corpus_indices:
                return render_template('index.html', error="No documents selected.", uploaded_files=uploaded_files)
            
            # Improved search pipeline
            ranked_docs = cached_search(query, corpus_indices)
            filtered_corpus = [documents_content[i] for i in corpus_indices]
            
            # Rerank top 10 results
            if ranked_docs:
                pairs = [(query, filtered_corpus[doc_idx]) for doc_idx, _ in ranked_docs[:10]]
                reranker_scores = reranker.predict(pairs)
                combined_results = []
                for (doc_idx, bm25_score), rerank_score in zip(ranked_docs[:10], reranker_scores):
                    combined_results.append((doc_idx, 0.7*bm25_score + 0.3*rerank_score))
                ranked_docs = sorted(combined_results, key=lambda x: x[1], reverse=True)
            
            # Prepare results
            results = []
            bm25_results = []
            for doc_index, score in ranked_docs[:5]:  # Return top 5 results
                matching_sections = highlight_matches(filtered_corpus[doc_index], query)
                bm25_results.extend(matching_sections)
                results.append({
                    'doc_index': doc_index,
                    'score': score,
                    'content': filtered_corpus[doc_index],
                    'matching_sections': matching_sections,
                    'paraphrased_response': interpret_answer(query, matching_sections)
                })
                
            return render_template('index.html', 
                                results=results,
                                query=query,
                                uploaded_files=uploaded_files,
                                final_answer=interpret_answer(query, bm25_results))

    return render_template('index.html', uploaded_files=uploaded_files)

if __name__ == '__main__':
    app.run(debug=True)
