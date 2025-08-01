import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 1. Download NLTK data (only needs to be done once) ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')


# --- 2. Load Dataset from File ---
def load_documents_from_file(filepath):
    """
    Loads documents from a text file.
    Each line should be either:
      - 'doc_id: document content'
      - or just 'document content' (IDs will be auto-generated)
    Returns a dictionary: {doc_id: content}
    """
    documents = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if ':' in line and line.split(':', 1)[0].startswith('doc'):
                doc_id, content = line.split(':', 1)
                doc_id = doc_id.strip()
                content = content.strip()
            else:
                doc_id = f"doc{idx+1}"
                content = line
            documents[doc_id] = content
    return documents

# Specify your dataset file path here
# DATASET_FILE = "IMDB Dataset.csv"
DATASET_FILE = "Articles.csv"

documents = load_documents_from_file(DATASET_FILE)

# --- 3. Text Preprocessing ---
def preprocess_text(text):
    """
    Cleans and standardizes text for processing.
    - Lowercases text
    - Removes punctuation
    - Tokenizes text (splits into words)
    - Removes stopwords (common words like 'the', 'a', 'is')
    - Stems words (reduces words to their root form, e.g., 'running' -> 'run')
    """
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return stemmed_tokens

# Preprocess all documents in the dataset
processed_docs = {doc_id: preprocess_text(doc_text) for doc_id, doc_text in documents.items()}
# We need the processed text as a single string for the TfidfVectorizer
processed_docs_str = {doc_id: ' '.join(tokens) for doc_id, tokens in processed_docs.items()}


# --- 4. Inverted Index ---
def create_inverted_index(processed_docs):
    """
    Creates an inverted index from the processed documents.
    The index maps each term to a list of document IDs where the term appears.
    Example: {'new': ['doc1', 'doc2'], 'sanction': ['doc1']}
    """
    inverted_index = {}
    for doc_id, tokens in processed_docs.items():
        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)
    return inverted_index

inverted_index = create_inverted_index(processed_docs)


# --- 5. Vector Model and Ranking (TF-IDF) ---
# We use Scikit-learn's TfidfVectorizer for this part.
# It handles tokenization, counting, and TF-IDF transformation in one step.
# For consistency, we'll fit it on our already processed documents.
doc_list = list(processed_docs_str.values())
doc_ids = list(processed_docs_str.keys())

# Initialize the TF-IDF Vectorizer
# It will convert our text documents into a matrix of TF-IDF features.
vectorizer = TfidfVectorizer()

# Learn vocabulary and idf from the documents.
tfidf_matrix = vectorizer.fit_transform(doc_list)

# Get the vocabulary (the terms the vectorizer learned)
feature_names = vectorizer.get_feature_names_out()


# --- 6. Search Function ---
def search(query, top_n=5 ):
    """
    Performs a search on the documents.
    1. Preprocesses the query.
    2. Converts the query to a TF-IDF vector.
    3. Computes the cosine similarity between the query vector and all document vectors.
    4. Ranks documents based on similarity and returns the top N results.
    """
    # Preprocess the user's query
    processed_query = ' '.join(preprocess_text(query))

    if not processed_query.strip():
        print("Your query was empty after processing. Please try a different query.")
        return

    # Transform the query into a TF-IDF vector using the learned vocabulary
    query_vector = vectorizer.transform([processed_query])

    # Compute cosine similarity between the query vector and all document vectors
    # This gives us a score of how relevant each document is to the query.
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get the indices of the top N most similar documents
    # We use np.argsort to get the indices that would sort the array,
    # then we take the last 'top_n' indices in reverse order.
    # We add a small check to handle cases where the query has no matching terms.
    if np.all(cosine_similarities == 0):
        print("No relevant documents found for your query.")
        return

    top_doc_indices = np.argsort(cosine_similarities)[-top_n:][::-1]

    # --- Display Results ---
    # print(f"\n--- Search Results for '{query}' ---")
    for i, idx in enumerate(top_doc_indices):
        # We only show results with a similarity score > 0
        if cosine_similarities[idx] > 0:
            doc_id = doc_ids[idx]
            original_doc = documents[doc_id]
            score = cosine_similarities[idx]
            print(f"{i+1}. Document: {doc_id} (Score: {score:.4f})")
            print(f"\n")
            print(f"   Content: {original_doc}\n")
            print(f"\n")

# --- 7. Main Execution Block (Command-Line Interface) ---
if __name__ == "__main__":
    print("--- Simple Search Engine ---")
    print(f"Loaded {len(documents)} documents from '{DATASET_FILE}'.")
    print("An inverted index has been created from the sample documents.")
    print("A TF-IDF Vectorizer has been trained on the document set.")
    print("Type your query below or type 'exit' to quit.")

    while True:
        user_query = input("\nEnter your search query: ")
        if user_query.lower() == 'exit':
            break
        search(user_query)
