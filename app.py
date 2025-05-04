from flask import Flask, render_template, request, redirect, url_for,flash, session
from flask_mail import Mail, Message
from dotenv import load_dotenv
from functools import lru_cache
from bm25_search_engine.bm25.bm25 import BM25
from bm25_search_engine.bm25.preprocess import preprocess_text
from bm25_search_engine.bm25.file_handlers import read_text_file, extract_text_from_pdf
from form import LoginForm, SignupForm
import mysql.connector
from mysql.connector import Error
from itsdangerous import URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import  LoginManager, login_user, logout_user, current_user, login_required
from model import User
import traceback
import os
from openai import OpenAI
from transformers import pipeline
from sentence_transformers import CrossEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from nltk import sent_tokenize


# Download NLTK stopwords (run once)
nltk.download('stopwords')
nltk.download('punkt')

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')

# Flask-Mail Configuration
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587 
app.config["MAIL_USE_TLS"] = True 
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")  
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_USERNAME")


mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# MySQL credentials
db_user = os.getenv('MYSQL_USER')
db_pass = os.getenv('MYSQL_PASSWORD')
db_host = os.getenv('MYSQL_HOST')
db_name = os.getenv('MYSQL_DB')

# Set up OpenAI API Key
#api_key=os.environ.get("OPENAI_API_KEY")
#openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not client:
    raise ValueError("Missing OpenAI API Key in environment variables.")

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # redirect here if not logged in


@login_manager.user_loader
def load_user(user_id):
    try:
        conn = mysql.connector.connect(
            user=db_user,
            password=db_pass,
            host=db_host,
            database=db_name
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM user WHERE userID = %s", (user_id,))
        user_data = cursor.fetchone()
        conn.close()
        if user_data:
            return User(user_data)
    except Exception as e:
        app.logger.error(f"User loader failed: {e}")
    return None


@app.route('/', methods=['GET', 'POST'])
def login_view():
    #session.pop('_flashes', None)
    form = LoginForm()
    if request.method == "POST" and form.validate_on_submit():
        email = form.email.data
        password = str(form.password.data)
        try:
            with mysql.connector.connect(
                user=db_user, 
                password=db_pass,
                host=db_host,
                database=db_name
            ) as conn:
                with conn.cursor(dictionary=True) as cursor:
                    cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
                    result = cursor.fetchone()
                    
                    if result:
                        user = User(result)
                        
                        # Check if email is verified before allowing login
                        if not user.verified:
                            flash("Account not verified. Please check your email.", "danger")
                            return redirect(url_for("login_view"))
                        
                        if check_password_hash(user.password, password): 
                            login_user(user)
                            #flash("Login Successful.", 'success')
                            return redirect(url_for("index"))
                        else:
                            flash("Invalid password.", "danger")
                    else:
                        flash("User not found.", "danger")
        except mysql.connector.Error as e:
            flash("Database connection failed.", "danger")
            app.logger.error(f"MySQL Error: {e}")

    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for('login_view'))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = SignupForm()
    if request.method == "POST" and form.validate_on_submit():
        firstName = form.firstname.data
        lastName = form.lastname.data
        email = form.email.data
        password = form.password.data
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256", salt_length=16)

        try:
            with mysql.connector.connect(
                user=db_user,
                password=db_pass,
                host=db_host,
                database=db_name
            ) as conn:
                with conn.cursor(dictionary=True) as cursor:
                    # Check if user already exists
                    cursor.execute("SELECT * FROM user WHERE email = %s", (email,))
                    if cursor.fetchone():
                        flash("Email already registered.", "danger")
                    else:
                        # Generate verification token
                        token = serializer.dumps(email, salt="email-confirm")

                        # Send verification email
                        verify_url = url_for("verify_email", token=token, _external=True)
                        subject = "Confirm Your Email"
                        body = f"Click the link to verify your email: {verify_url}"

                        msg = Message(subject, recipients=[email], body=body)
                        mail.send(msg)

                        # Insert unverified user into the database
                        cursor.execute(
                            "INSERT INTO user (firstName, lastName, email, verified,  password) VALUES (%s, %s, %s, %s, %s)",
                            (firstName, lastName, email, 0, hashed_password)
                        )
                        conn.commit()

                        flash("Signup successful. Please check your email to verify your account.", "success")
                        return redirect(url_for("login_view"))
        except mysql.connector.Error as e:
            flash("Database error during signup.", "danger")
            app.logger.error(f"MySQL Error: {e}")

    return render_template("signup.html", form=form)


@app.route("/verify-email/<token>")
def verify_email(token):
    try:
        email = serializer.loads(token, salt="email-confirm", max_age=3600)  # Token expires in 1 hour

        with mysql.connector.connect(
            user=db_user,
            password=db_pass,
            host=db_host,
            database=db_name
        ) as conn:
            with conn.cursor(dictionary=True) as cursor:
                cursor.execute("UPDATE user SET verified = %s WHERE email = %s", (1, email))
                conn.commit()

        flash("Email verified successfully! You can now log in.", "success")
    except Exception:
        flash("Invalid or expired token.", "danger")

    return redirect(url_for("login_view"))


# --- Global Variables ---
search_engine = None
documents_content = []
uploaded_files = []
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Model Initialization ---
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
def chunk_text(text, chunk_size=2000):
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

def interpret_answer(query, matching_sections):
    if not matching_sections:
        return "Sorry, I couldn't find any relevant information to answer your question."

    context = " ".join(matching_sections[:3])  # Use top 3 relevant sections

    system_prompt = (
        "You are a document-based assistant. Answer the user's question only using the provided context from the documents and the user's question. "
        "provide a clear, natural, and paraphrased answer in complete sentences. Be concise and informative."
        "Do not use prior knowledge. If the answer is not in the context, say: 'I could not find relevant information in the documents.'"
    )

    user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        traceback.print_exc()
        return "An error occurred while generating the response."

# --- Routes ---
@app.route('/home', methods=['GET', 'POST'])
@login_required
def index():
    global search_engine, documents_content, uploaded_files

    if request.method == 'POST':
        if 'files' in request.files:
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

                corpus.append(preprocess_text(text))  # Full document
                documents_content.append(text)        # Full document text
                uploaded_files.append({
                    'name': file.filename,
                    'content': text
                })
            search_engine = BM25(corpus)
            return redirect(url_for('index'))

        if 'query' in request.form:
            query = request.form['query']
            selected_files = request.form.getlist('selected_files')
            
            if not search_engine:
                return render_template('index.html', error="No documents uploaded yet.", uploaded_files=uploaded_files)
            
            corpus_indices = [i for i, f in enumerate(uploaded_files) if f['name'] in selected_files]
            if not corpus_indices:
                return render_template('index.html', error="No documents selected.", uploaded_files=uploaded_files)

            query_pre = preprocess_text(query)
            ranked_docs = []
            for i in corpus_indices:
                score = search_engine._compute_bm25_score(query_pre.split(), i)
                ranked_docs.append((i, score))
            ranked_docs.sort(key=lambda x: x[1], reverse=True)

            results = []
            for doc_index, score in ranked_docs[:3]:  # Top 3 documents
                full_text = documents_content[doc_index]
                sentences = sent_tokenize(full_text)

                # Score each sentence individually using BM25 scoring logic
                scored_sentences = []
                for sent in sentences:
                    sent_score = BM25([preprocess_text(sent)])._compute_bm25_score(query_pre.split(), 0)
                    scored_sentences.append((sent, sent_score))
                top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:3]  # Top 3 matching sentences

                # Prepare results
                results.append({
                    'doc_index': doc_index,
                    'score': score,
                    'matching_sections': [s[0] for s in top_sentences],
                    'paraphrased_response': interpret_answer(query, [s[0] for s in top_sentences]),
                    'content': uploaded_files[doc_index]['content']
                })

            return render_template('index.html',
                                   results=results,
                                   query=query,
                                   uploaded_files=uploaded_files,
                                   final_answer=interpret_answer(query, [r['content'] for r in results]))

    return render_template('index.html', uploaded_files=uploaded_files)

if __name__ == '__main__':
    app.run(debug=True)
