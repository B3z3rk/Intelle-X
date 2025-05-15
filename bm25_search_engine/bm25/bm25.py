from collections import defaultdict,Counter
import numpy as np
import math
from .preprocess import preprocess_text, expand_query


class BM25:
    def __init__(self, corpus):
        print(corpus)
        self.corpus = corpus
        self.doc_lengths = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / len(corpus) if corpus else 0
        self.k1 = self._estimate_k1()
        self.b = self._estimate_b()
        self.vocab = set()
        self.df = defaultdict(int)
        self.idf = {}
        self.tf = []
        self._preprocess_corpus()

    def _estimate_k1(self):
        std_len = np.std(self.doc_lengths)
        k1 = 1.2 + (std_len / self.avgdl if self.avgdl > 0 else 0)
        return round(min(k1, 2.0), 3)  # cap at 2

    def _estimate_b(self):
        var_len = np.var(self.doc_lengths)
        b = min(0.5 + (var_len / (self.avgdl**2 + 1e-6)), 1.0) #cap at 1
        return round(b, 3)

    def _preprocess_corpus(self):
        N = len(self.corpus)
        for doc in self.corpus:
            terms = doc.split()
            self.vocab.update(terms)
            term_freq = Counter(terms)
            self.tf.append(term_freq)
            for term in term_freq.keys(): 
                self.df[term] += 1

        # Precompute IDF
        for term in self.vocab:
            df = self.df[term]
            self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _compute_bm25_score(self, query, doc_index):
        score = 0
        doc_length = self.doc_lengths[doc_index]
        doc_tf = self.tf[doc_index]
        for term in query:
            if term not in self.idf:
                continue
            tf = doc_tf.get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * ((1 - self.b) + self.b * (doc_length / self.avgdl))
            score += self.idf[term] * (numerator / denominator)
            
        return score

    def rank_documents(self, query):
        query_terms = query.split()
        return sorted(
            ((i, self._compute_bm25_score(query_terms, i)) for i in range(len(self.corpus))),
            key=lambda x: x[1],
            reverse=True
        )

