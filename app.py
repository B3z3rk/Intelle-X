from flask import Flask, render_template, request, redirect, url_for
import os
from bm25_search_engine.bm25.bm25 import BM25
from bm25_search_engine.bm25.preprocess import preprocess_text
from bm25_search_engine.bm25.file_handlers import read_text_file, extract_text_from_pdf

from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Download NLTK stopwords (run once)
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store processed document chunks and metadata
search_engine = None
documents_content = []  # List of document chunks (original text)
uploaded_files = []     # Metadata: file/chunk names and content

# Stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def chunk_text(text, chunk_size=10000): #Can lower or raise the number
    """
    Splits the text into chunks of a given word count.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

@app.route('/', methods=['GET', 'POST'])
def index():
    global search_engine, documents_content, uploaded_files

    if request.method == 'POST':
        # Handle file uploads
        if 'files' in request.files:
            files = request.files.getlist('files')
            corpus = []
            documents_content = []
            uploaded_files.clear()
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
                
                # Chunk the file if it exceeds a threshold (e.g., 1000 words)
                if len(text.split()) > 1000:
                    text_chunks = chunk_text(text)
                    for i, chunk in enumerate(text_chunks):
                        corpus.append(preprocess_text(chunk))
                        documents_content.append(chunk)
                        uploaded_files.append({'name': f"{file.filename} (chunk {i+1})", 'content': chunk})
                else:
                    corpus.append(preprocess_text(text))
                    documents_content.append(text)
                    uploaded_files.append({'name': file.filename, 'content': text})
            # Instantiate BM25 with tuned parameters (e.g., k1=1.8, b=0.65)
            search_engine = BM25(corpus, k1=1.8, b=0.65)
            return redirect(url_for('index'))

        # Handle search queries
        if 'query' in request.form:
            query = request.form['query']
            selected_files = request.form.getlist('selected_files')
            if search_engine:
                # Filter document chunks based on user-selected files
                filtered_corpus = [documents_content[i] for i in range(len(documents_content))
                                   if uploaded_files[i]['name'] in selected_files]
                if not filtered_corpus:
                    return render_template('index.html', error="No documents selected.", uploaded_files=uploaded_files)
                
                # Create a new BM25 instance for the filtered corpus
                filtered_search_engine = BM25(filtered_corpus, k1=1.8, b=0.65)
                ranked_docs = filtered_search_engine.rank_documents(preprocess_text(query))
                bm25_results = []
                results = []
                for doc_index, score in ranked_docs:
                    # Retrieve matching sections from each document chunk
                    matching_sections = highlight_matches(filtered_corpus[doc_index], query)
                    bm25_results.extend(matching_sections)
                    results.append({
                        'doc_index': doc_index,
                        'score': score,
                        'content': filtered_corpus[doc_index],
                        'matching_sections': matching_sections,
                    })
                # Use the middle layer to interpret and answer the user's question
                final_answer = interpret_answer(query, bm25_results)
                return render_template('index.html', results=results, query=query, 
                                       uploaded_files=uploaded_files, final_answer=final_answer)
            else:
                return render_template('index.html', error="No documents uploaded yet.", uploaded_files=uploaded_files)

    return render_template('index.html', uploaded_files=uploaded_files)

# # Load pre-trained DistilBART model and tokenizer for answer generation
# model_name = "sshleifer/distilbart-cnn-12-6"
# tokenizer = BartTokenizer.from_pretrained(model_name)
# model = BartForConditionalGeneration.from_pretrained(model_name)

def interpret_answer(query, bm25_results, max_input_length=512, max_output_length=200):
    """
    Generate an answer by interpreting the user's question using BM25 matching sections.
    """
    # Combine the matching sections into a single context string
    context = " ".join(bm25_results)
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)
    outputs = model.generate(inputs, max_length=max_output_length, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# Load the pre-trained model and tokenizer
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load or prepare your dataset
# Example: A dataset with "question" and "context" columns
data = {
    "question": ["What is Bandura's last name?", "Who did the research?"],
    "context": ["Albert Bandura is a psychologist known for his work on social learning theory.", 
                "The research was conducted by Dr. Smith and his team."],
    "answer": ["Bandura", "Dr. Smith"]
}
dataset = Dataset.from_dict(data)

# Tokenize the dataset
def preprocess_function(examples):
    inputs = [f"question: {q} context: {c}" for q, c in zip(examples["question"], examples["context"])]
    targets = examples["answer"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into training and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-distilbart")
tokenizer.save_pretrained("./fine-tuned-distilbart")

def highlight_matches(text, query):
    """
    Highlight sections of the text that contain the query terms, ignoring stopwords.
    """
    # Tokenize and stem the query, ignoring stopwords
    query_terms = [stemmer.stem(term) for term in word_tokenize(query.lower()) if term not in stop_words]
    sentences = text.split('. ')
    matching_sections = []
    for sentence in sentences:
        # Stem the sentence for comparison
        stemmed_sentence = [stemmer.stem(word) for word in word_tokenize(sentence.lower())]
        if any(term in stemmed_sentence for term in query_terms):
            # Highlight the original terms (not stemmed) in the sentence
            for term in query_terms:
                original_term = next((word for word in word_tokenize(sentence.lower()) if stemmer.stem(word) == term), None)
                if original_term:
                    sentence = sentence.replace(original_term, f"<strong>{original_term}</strong>")
            matching_sections.append(sentence)
    return matching_sections

if __name__ == '__main__':

    app.run(debug=True)
