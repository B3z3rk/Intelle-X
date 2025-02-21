from collections import defaultdict
import math

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc.split()) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / len(corpus) if corpus else 0 #Handles empty corpus (a collection of written texts)
        self.vocab = set()
        self.df = defaultdict(int)
        self.idf = {}
        self.tf = []
        self._preprocess_corpus()

    def _preprocess_corpus(self):
        for doc in self.corpus:
            terms = doc.split()
            self.vocab.update(terms)
            term_freq = defaultdict(int)
            for term in terms:
                term_freq[term] += 1
            self.tf.append(term_freq)
            for term in set(terms):
                self.df[term] += 1

        N = len(self.corpus)
        for term in self.vocab:
            self.idf[term] = math.log((N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)

    def _compute_bm25_score(self, query, doc_index):
        score = 0
        doc_length = self.doc_lengths[doc_index]
        for term in query:
            if term not in self.vocab:
                continue
            tf = self.tf[doc_index].get(term, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
            score += self.idf[term] * (numerator / denominator)
        return score

    def rank_documents(self, query):
        query_terms = query.split()
        scores = []
        for i in range(len(self.corpus)):
            score = self._compute_bm25_score(query_terms, i)
            scores.append((i, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)