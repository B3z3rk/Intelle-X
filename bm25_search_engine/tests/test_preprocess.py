from bm25.preprocess import preprocess_text

def test_preprocess_text():
    text = "The cat sat on the mat!"
    processed_text = preprocess_text(text)
    assert processed_text == "cat sat mat"

def test_preprocess_punctuation():
    text = "Hello, world! This is a test."
    processed_text = preprocess_text(text)
    assert processed_text == "hello world test"

def test_preprocess_stopwords():
    text = "This is a test of the preprocessing function"
    processed_text = preprocess_text(text)
    assert processed_text == "test preprocessing function"

def test_preprocess_mixed_case():
    text = "Hello World"
    processed_text = preprocess_text(text)
    assert processed_text == "hello world"

def test_preprocess_numbers():
    text = "This is test number 123"
    processed_text = preprocess_text(text)
    assert processed_text == "test number"

