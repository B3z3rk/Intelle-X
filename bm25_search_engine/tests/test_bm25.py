import pytest
from bm25.bm25 import BM25

def test_bm25():
    corpus = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are great"]
    bm25 = BM25(corpus)
    query = "cat and dog"
    ranked_docs = bm25.rank_documents(query)
    assert len(ranked_docs) == len(corpus)


def test_bm25_empty_corpus():
    corpus = []
    bm25 = BM25(corpus)
    query = "cat and dog"
    ranked_docs = bm25.rank_documents(query)
    assert len(ranked_docs) == 0

def test_bm25_empty_query():
    corpus = ["the cat sat on the mat", "the dog sat on the log"]
    bm25 = BM25(corpus)
    query = ""
    ranked_docs = bm25.rank_documents(query)
    assert len(ranked_docs) == len(corpus)
    # All documents should have a score of 0 for an empty query
    assert all(score == 0 for _, score in ranked_docs)

def test_bm25_no_matching_terms():
    corpus = ["the cat sat on the mat", "the dog sat on the log"]
    bm25 = BM25(corpus)
    query = "elephant"
    ranked_docs = bm25.rank_documents(query)
    assert len(ranked_docs) == len(corpus)
    # All documents should have a score of 0 for no matching terms
    assert all(score == 0 for _, score in ranked_docs)

def test_bm25_single_document():
    corpus = ["the cat sat on the mat"]
    bm25 = BM25(corpus)
    query = "cat"
    ranked_docs = bm25.rank_documents(query)
    assert len(ranked_docs) == 1
    assert ranked_docs[0][1] > 0  # The document should have a positive score

def test_bm25_multiple_queries():
    corpus = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are great"]
    bm25 = BM25(corpus)
    queries = ["cat", "dog", "cat and dog"]
    for query in queries:
        ranked_docs = bm25.rank_documents(query)
        assert len(ranked_docs) == len(corpus)
        # Ensure the top-ranked document contains at least one query term
        top_doc_index = ranked_docs[0][0]
        assert any(term in corpus[top_doc_index] for term in query.split())