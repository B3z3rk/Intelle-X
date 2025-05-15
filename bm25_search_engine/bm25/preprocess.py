import re
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

ALLOWED_POS = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS'}

SYNONYM_MAP = {
    "study": ["research", "experiment"],
    "result": ["finding", "outcome"],
    "theory": ["model", "framework"]
}

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = word_tokenize(text)
    tagged = pos_tag(words)
    filtered = [lemmatizer.lemmatize(w, 'v') for w, tag in tagged if tag in ALLOWED_POS and w not in stop_words]
    return " ".join(filtered)

'''def expand_query(query):
    terms = preprocess_text(query).split()
    expanded = []
    for term in terms:
        expanded.append(term)
        expanded.extend(SYNONYM_MAP.get(term, []))
    return " ".join(expanded)'''



def get_synonyms(term):
    synonyms = set()
    for synset in wordnet.synsets(term):
        for lemma in synset.lemmas():
            name = lemma.name().replace("_", " ").lower()
            if name != term:
                synonyms.add(name)
    return list(synonyms)

def expand_query(query):
    terms = preprocess_text(query).split()  
    expanded = set(terms) 

    for term in terms:
        expanded.update(get_synonyms(term))

    return " ".join(sorted(expanded))  # Sort optional
