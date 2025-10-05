import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data if not already present
def download_nltk_data():
    """Download required NLTK datasets."""
    required_data = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab'
    }
    
    for name, path in required_data.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK {name}...")
            nltk.download(name, quiet=True)

download_nltk_data()

# Global variables
documents = {}
processed_docs = {}
processed_docs_str = {}
inverted_index = {}
vectorizer = None
tfidf_matrix = None
doc_ids = []

def load_documents_from_file(filepath):
    """
    Load documents from a text file.
    Expected format:
        - 'doc_id: document content' 
        - or just 'document content' (auto-generated IDs)
    
    Returns:
        dict: {doc_id: content}
    """
    docs = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Check if line has explicit doc_id
                if ':' in line and line.split(':', 1)[0].strip().startswith('doc'):
                    doc_id, content = line.split(':', 1)
                    doc_id = doc_id.strip()
                    content = content.strip()
                else:
                    doc_id = f"doc{idx}"
                    content = line
                
                docs[doc_id] = content
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return {}
    except Exception as e:
        print(f"Error loading documents: {e}")
        return {}
    
    return docs

def preprocess_text(text):
    """
    Clean and standardize text for processing.
    
    Steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Tokenize into words
        4. Remove stopwords
        5. Stem words to root form
    
    Returns:
        list: Processed tokens
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stem words
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return stemmed_tokens

def create_inverted_index(proc_docs):
    """
    Create an inverted index mapping terms to document IDs.
    
    Args:
        proc_docs (dict): {doc_id: [tokens]}
    
    Returns:
        dict: {term: [doc_ids]}
    """
    index = {}
    for doc_id, tokens in proc_docs.items():
        for token in set(tokens):  # Use set to avoid duplicates
            if token not in index:
                index[token] = []
            index[token].append(doc_id)
    return index

def initialize_search_engine(filepath):
    """
    Initialize the search engine with documents from file.
    
    Args:
        filepath (str): Path to the documents file
    
    Returns:
        bool: True if successful, False otherwise
    """
    global documents, processed_docs, processed_docs_str, inverted_index
    global vectorizer, tfidf_matrix, doc_ids
    
    # Load documents
    documents = load_documents_from_file(filepath)
    if not documents:
        print("No documents loaded. Search engine not initialized.")
        return False
    
    # Preprocess documents
    processed_docs = {
        doc_id: preprocess_text(doc_text) 
        for doc_id, doc_text in documents.items()
    }
    
    # Convert to strings for TF-IDF
    processed_docs_str = {
        doc_id: ' '.join(tokens) 
        for doc_id, tokens in processed_docs.items()
    }
    
    # Create inverted index
    inverted_index = create_inverted_index(processed_docs)
    
    # Initialize TF-IDF vectorizer
    doc_list = list(processed_docs_str.values())
    doc_ids = list(processed_docs_str.keys())
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(doc_list)
    
    print(f"Search engine initialized with {len(documents)} documents.")
    return True

def search(query, top_n=5):
    """
    Search documents using TF-IDF and cosine similarity.
    
    Args:
        query (str): Search query
        top_n (int): Number of top results to return
    
    Returns:
        list: List of tuples (doc_id, score, content)
    """
    if not documents or vectorizer is None:
        print("Error: Search engine not initialized. Please load documents first.")
        return []
    
    # Preprocess query
    processed_query = ' '.join(preprocess_text(query))
    
    if not processed_query.strip():
        print("Query is empty after processing. Please try different terms.")
        return []
    
    # Transform query to TF-IDF vector
    query_vector = vectorizer.transform([processed_query])
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Check if any matches found
    if np.all(cosine_similarities == 0):
        print("No relevant documents found for your query.")
        return []
    
    # Get top N documents
    top_doc_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    
    results = []
    for idx in top_doc_indices:
        if cosine_similarities[idx] > 0:
            doc_id = doc_ids[idx]
            score = cosine_similarities[idx]
            content = documents[doc_id]
            results.append((doc_id, score, content))
    
    return results

def display_results(query, results):
    """Display search results in formatted output."""
    if not results:
        print(f"\nNo results found for '{query}'")
        return
    
    print(f"\n--- Search Results for '{query}' ---")
    for i, (doc_id, score, content) in enumerate(results, 1):
        print(f"\n{i}. Document: {doc_id} (Score: {score:.4f})")
        print(f"   Content: {content[:200]}{'...' if len(content) > 200 else ''}\n")

def main():
    """Main execution for command-line interface."""
    print("=" * 60)
    print("         DOCUMENT SEARCH ENGINE")
    print("=" * 60)
    
    # Specify dataset file
    dataset_file = "Articles.csv"
    
    # Initialize search engine
    if not initialize_search_engine(dataset_file):
        return
    
    print("\nType your query or 'exit' to quit.\n")
    
    while True:
        try:
            user_query = input("Search: ").strip()
            
            if user_query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not user_query:
                continue
            
            results = search(user_query, top_n=5)
            display_results(user_query, results)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()