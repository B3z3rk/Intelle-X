from flask import Flask, render_template, request, redirect, url_for
import os
from bm25_search_engine.bm25.bm25 import BM25
from bm25_search_engine.bm25.preprocess import preprocess_text
from bm25_search_engine.bm25.file_handlers import read_text_file, extract_text_from_pdf
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
search_engine = None
documents_content = []  # Store the content of uploaded documents
uploaded_files = []     # Store metadata of uploaded files (name and content)

@app.route('/', methods=['GET', 'POST'])
def index():
    global search_engine, documents_content, uploaded_files

    if request.method == 'POST':
        # Handle file uploads
        if 'files' in request.files:
            files = request.files.getlist('files')
            corpus = []
            documents_content = []
            for file in files:
                if file.filename == '':
                    continue
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                if file.filename.endswith('.txt'):
                    text = read_text_file(file_path)
                elif file.filename.endswith('.pdf'):
                    text = extract_text_from_pdf(file_path)
                else:
                    continue
                corpus.append(preprocess_text(text))
                documents_content.append(text)  # Store the original content
                uploaded_files.append({'name': file.filename, 'content': text})  # Store file metadata
            search_engine = BM25(corpus)
            return redirect(url_for('index'))

        # Handle search queries
        if 'query' in request.form:
            query = request.form['query']
            selected_files = request.form.getlist('selected_files')  # Get selected files
            if search_engine:
                # Filter documents based on selected files
                filtered_corpus = [documents_content[i] for i in range(len(documents_content)) if uploaded_files[i]['name'] in selected_files]
                if not filtered_corpus:
                    return render_template('index.html', error="No documents selected.", uploaded_files=uploaded_files)
                
                # Create a new BM25 instance with the filtered corpus
                filtered_search_engine = BM25(filtered_corpus)
                ranked_docs = filtered_search_engine.rank_documents(preprocess_text(query))
                results = []
                for doc_index, score in ranked_docs:
                    # Extract matching sections
                    matching_sections = highlight_matches(filtered_corpus[doc_index], query)

                     # Generate paraphrased response
                    paraphrased_response = paraphrase(query, " ".join(matching_sections))

                    results.append({
                        'doc_index': doc_index,
                        'score': score,
                        'content': filtered_corpus[doc_index],
                        'matching_sections': matching_sections,
                        'paraphrased_response': paraphrased_response
                    })
                return render_template('index.html', results=results, query=query, uploaded_files=uploaded_files)
            else:
                return render_template('index.html', error="No documents uploaded yet.", uploaded_files=uploaded_files)

    return render_template('index.html', uploaded_files=uploaded_files)

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def paraphrase(query, context, max_input_length=512, max_output_length=200):
    """
    Generate a paraphrased response using T5.
    :param query: The user's query.
    :param context: The context (matching sections from the documents).
    :param max_input_length: Maximum length of the input sequence (default: 512).
    :param max_output_length: Maximum length of the output sequence (default: 200).
    :return: Paraphrased response.
    """
    input_text = f"paraphrase: {query} based on {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    outputs = model.generate(inputs, max_length=max_output_length, num_return_sequences=1)
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text

def highlight_matches(text, query):
    """
    Highlight sections of the text that contain the query terms.
    """
    query_terms = query.lower().split()
    sentences = text.split('. ')  # Split text into sentences
    matching_sections = []
    for sentence in sentences:
        if any(term in sentence.lower() for term in query_terms):
            # Highlight matching terms in the sentence
            for term in query_terms:
                sentence = sentence.replace(term, f"<strong>{term}</strong>")
            matching_sections.append(sentence)
    return matching_sections

if __name__ == '__main__':
    app.run(debug=True)