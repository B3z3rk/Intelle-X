from flask import Flask, render_template, request, redirect, url_for,flash, session,send_from_directory
from flask_mail import Mail, Message
from dotenv import load_dotenv
from functools import lru_cache
from bm25_search_engine.bm25.bm25 import BM25
from bm25_search_engine.bm25.preprocess import preprocess_text
from bm25_search_engine.bm25.file_handlers import read_text_file, extract_text_from_pdf, extract_text_from_url
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
from cachetools import TTLCache
from hashlib import sha256

from heapq import nlargest
from collections import defaultdict




# Download NLTK stopwords (run once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

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

# Cache up to 100 queries, each expires after 50 minutes (3000 seconds)
#query_cache = TTLCache(maxsize=100, ttl=3000)
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity=1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)  # Recently used
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

query_cache = LRUCache(capacity=500)


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
        cursor.execute("SELECT * FROM user WHERE userId = %s", (user_id,))
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
'''uploaded_files = []'''
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Model Initialization ---
# intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


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



def generate_query_key(query, corpus_indices):
    key_string = query + "_" + "_".join(map(str, sorted(corpus_indices)))
    return sha256(key_string.encode()).hexdigest()



def get_results(query, selected_filenames, uploaded_files, search_engine):
    cache_key = generate_query_key(query, selected_filenames)
    '''if cache_key in query_cache:
        return query_cache[cache_key]  # Cache hit'''
    cached = query_cache.get(cache_key)
    if cached:
        return cached


    # Preprocess query once
    query_pre = preprocess_text(query)
    query_terms = query_pre.split()
    query_tokens = set(query_terms)

    # Get indices of selected documents
    filename_to_index = {f['filename']: i for i, f in enumerate(uploaded_files)}
    corpus_indices = [filename_to_index[fname] for fname in selected_filenames if fname in filename_to_index]

    # Score documents using BM25
    ranked_docs = [
        (i, search_engine._compute_bm25_score(query_terms, i))
        for i in corpus_indices
    ]
    ranked_docs = sorted(ranked_docs, key=lambda x: x[1], reverse=True)

    # Collect top 3 matching documents
    results = []
    for doc_index, score in ranked_docs[:3]:
        full_text = uploaded_files[doc_index]['content']
        sentences = sent_tokenize(full_text)

        # Score each sentence by token overlap
        scored_sentences = [
            (sent, len(query_tokens & set(preprocess_text(sent).split())))
            for sent in sentences
        ]
        top_sentences = [s for s, _ in nlargest(3, scored_sentences, key=lambda x: x[1])]
        paraphrased = interpret_answer(query, top_sentences)

        results.append({
            'doc_index': doc_index,
            'score': score,
            'matching_sections': top_sentences,
            'paraphrased_response': paraphrased,
            'content': full_text,
            'filename': uploaded_files[doc_index]['filename']
        })

    #query_cache[cache_key] = results
    query_cache.put(cache_key, results)
    return results



# --- Routes ---
@app.route('/home', methods=['GET', 'POST'])
@login_required
def index():
    db = mysql.connector.connect(
        user=db_user,
        password=db_pass,
        host=db_host,
        database=db_name)
    cursor = db.cursor(dictionary=True)
    user_id = current_user.id

    if request.method == 'POST':
        if 'files' in request.files:
            files = request.files.getlist('files')

            # Remove old files for this user
            cursor.execute("DELETE FROM user_files WHERE user_id = %s", (user_id,))

            for file in files:
                if file.filename == '':
                    continue
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                text = read_text_file(file_path) if file.filename.endswith('.txt') else extract_text_from_pdf(file_path)

                cursor.execute("""
                    INSERT INTO user_files (user_id, filename, content)
                    VALUES (%s, %s, %s)
                """, (user_id, file.filename, text))

            db.commit()
            return redirect(url_for('index'))

        elif 'query' in request.form:
            query = request.form['query']
            selected_filenames = request.form.getlist('selected_files')

            cursor.execute("SELECT filename, content FROM user_files WHERE user_id = %s", (user_id,))
            files = cursor.fetchall()

            if not files:
                return render_template('index.html', error="No documents uploaded yet.", uploaded_files=[])

            # Ensure uploaded_files is a list of dictionaries with 'filename' and 'content'
            uploaded_files = [{'filename': f['filename'], 'content': f['content']} for f in files]

            if not any(f['filename'] in selected_filenames for f in uploaded_files):
                return render_template('index.html', error="No documents selected.", uploaded_files=uploaded_files)

            corpus = [preprocess_text(f['content']) for f in uploaded_files if f['filename'] in selected_filenames]
            search_engine = BM25(corpus)
            results = get_results(query, selected_filenames, uploaded_files,search_engine)  # Now correctly formatted

            for result in results:
                cursor.execute("""
                    INSERT INTO history (userId, query, docIndex, filename, score, matching_sections, paraphrased_response)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id,
                    query,
                    result['doc_index'],
                    result['filename'],
                    result['score'],
                    "; ".join(result['matching_sections']),
                    result['paraphrased_response']
                ))

            # Keep only last 10 history entries
            cursor.execute("""
                DELETE FROM history
                WHERE userId = %s AND hid NOT IN (
                    SELECT hid FROM (
                        SELECT hid FROM history
                        WHERE userId = %s
                        ORDER BY hid DESC
                        LIMIT 10
                    ) AS recent
                )
            """, (user_id, user_id))
            db.commit()

            cursor.execute("SELECT filename, content FROM user_files WHERE user_id = %s", (user_id,))
            uploaded_files = [{'name': row['filename'], 'content': row['content']} for row in cursor.fetchall()]
            history = getHistory()

            return render_template('index.html',
                                   results=results,
                                   query=query,
                                   uploaded_files=uploaded_files,
                                   history=history,
                                   final_answer=interpret_answer(query, [r['content'] for r in results]))

    # GET request fallback
    cursor.execute("SELECT filename, content FROM user_files WHERE user_id = %s", (user_id,))
    uploaded_files = [{'name': row['filename'], 'content': row['content']} for row in cursor.fetchall()]
    history = getHistory()
    cursor.close()
    db.close()
    return render_template('index.html', uploaded_files=uploaded_files, history=history)




'''@app.route('/view/<filename>')
def view_file(filename):
    return render_template('viewer.html', filename=filename)'''

'''@app.route('/view/<filename>')
def view_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)'''
@app.route('/view/<filename>')
def view_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return "File not found", 404
    
    # Check if file contains a link
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            
            if content.startswith('https'):
                return redirect(content)
    except:
        # If no redirection occurred, serve the file normally
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=False)

# return file
@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

#return history
@app.route('/history/')
@login_required
def getHistory():
    db=mysql.connector.connect(
            user=db_user,
            password=db_pass,
            host=db_host,
            database=db_name)
    cursor = db.cursor()

    
    cursor = db.cursor(dictionary=True) 

    user_id = current_user.id
    cursor.execute("SELECT * FROM history WHERE userId = %s ORDER BY hid DESC", (user_id,))  

    history = cursor.fetchall()

    cursor.close()
    db.close()

    return history
#return history item
@app.route('/history/<int:hid>')
@login_required
def viewHistory(hid):
    db = mysql.connector.connect(
        user=db_user,
        password=db_pass,
        host=db_host,
        database=db_name
    )
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM history WHERE hid = %s AND userId = %s", (hid, current_user.id))
    item = cursor.fetchone()
    cursor.close()
    db.close()
    history=getHistory()
    if not item:
        return "History item not found or unauthorized.", 404

    return render_template('index.html', item=item,history=history)

if __name__ == '__main__':
    app.run(debug=True)
