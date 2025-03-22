import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    # Stem words that are not stopwords
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)
