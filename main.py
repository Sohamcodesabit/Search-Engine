import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from difflib import SequenceMatcher

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
word_vocabulary = set()  # Store all unique words for spell checking

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

def similarity_ratio(word1, word2):
    """Calculate similarity ratio between two words."""
    return SequenceMatcher(None, word1.lower(), word2.lower()).ratio()

def correct_spelling(word, vocabulary, threshold=0.8):
    """
    Correct spelling of a word using vocabulary.
    
    Args:
        word (str): Word to correct
        vocabulary (set): Set of correct words
        threshold (float): Similarity threshold (0-1)
    
    Returns:
        str: Corrected word or original if no match found
    """
    word_lower = word.lower()
    
    # If word exists in vocabulary, return as is
    if word_lower in vocabulary:
        return word_lower
    
    # Find best match
    best_match = word_lower
    best_score = 0
    
    for vocab_word in vocabulary:
        # Skip if length difference is too large
        if abs(len(word_lower) - len(vocab_word)) > 3:
            continue
            
        score = similarity_ratio(word_lower, vocab_word)
        
        if score > best_score and score >= threshold:
            best_score = score
            best_match = vocab_word
    
    return best_match

def preprocess_text(text, correct_spelling_flag=False):
    """
    Clean and standardize text for processing.
    
    Steps:
        1. Convert to lowercase
        2. Remove punctuation
        3. Tokenize into words
        4. Correct spelling (optional)
        5. Remove stopwords
        6. Stem words to root form
    
    Returns:
        list: Processed tokens
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Correct spelling if flag is set and vocabulary is available
    if correct_spelling_flag and word_vocabulary:
        tokens = [correct_spelling(token, word_vocabulary) for token in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stem words
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    
    return stemmed_tokens

def build_vocabulary(docs):
    """
    Build vocabulary from all documents.
    
    Args:
        docs (dict): {doc_id: content}
    
    Returns:
        set: Set of all unique words
    """
    vocab = set()
    for doc_text in docs.values():
        # Remove punctuation and tokenize
        text = doc_text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        vocab.update(tokens)
    return vocab

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

def highlight_query_terms(text, query_terms):
    """
    Highlight query terms in text by adding underline markers.
    
    Args:
        text (str): Original document text
        query_terms (list): List of query terms to highlight
    
    Returns:
        str: Text with HTML-style underline tags or plain underscores
    """
    if not query_terms:
        return text
    
    # Create a case-insensitive pattern for each query term
    highlighted_text = text
    
    for term in query_terms:
        if not term:
            continue
        
        # Create pattern that matches whole words (case-insensitive)
        pattern = re.compile(r'\b(' + re.escape(term) + r')\b', re.IGNORECASE)
        
        # Replace with underlined version
        highlighted_text = pattern.sub(r'__\1__', highlighted_text)
    
    return highlighted_text

def initialize_search_engine(filepath):
    """
    Initialize the search engine with documents from file.
    
    Args:
        filepath (str): Path to the documents file
    
    Returns:
        bool: True if successful, False otherwise
    """
    global documents, processed_docs, processed_docs_str, inverted_index
    global vectorizer, tfidf_matrix, doc_ids, word_vocabulary
    
    # Load documents
    documents = load_documents_from_file(filepath)
    if not documents:
        print("No documents loaded. Search engine not initialized.")
        return False
    
    # Build vocabulary for spell checking
    word_vocabulary = build_vocabulary(documents)
    
    # Preprocess documents (without spelling correction for building index)
    processed_docs = {
        doc_id: preprocess_text(doc_text, correct_spelling_flag=False) 
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

def search(query, top_n=5, return_highlights=True):
    """
    Search documents using TF-IDF and cosine similarity.
    
    Args:
        query (str): Search query
        top_n (int): Number of top results to return
        return_highlights (bool): Whether to return highlighted text
    
    Returns:
        list: List of tuples (doc_id, score, content, highlighted_content, query_terms)
    """
    if not documents or vectorizer is None:
        print("Error: Search engine not initialized. Please load documents first.")
        return []
    
    # Store original query terms (cleaned but not stemmed) for highlighting
    query_lower = query.lower()
    query_lower = query_lower.translate(str.maketrans('', '', string.punctuation))
    original_query_tokens = word_tokenize(query_lower)
    
    # Correct spelling in query terms
    corrected_query_tokens = [correct_spelling(token, word_vocabulary) for token in original_query_tokens]
    
    # Remove stopwords from corrected tokens
    stop_words = set(stopwords.words('english'))
    filtered_query_tokens = [word for word in corrected_query_tokens if word not in stop_words]
    
    # Preprocess query with spelling correction
    processed_query_tokens = preprocess_text(query, correct_spelling_flag=True)
    processed_query = ' '.join(processed_query_tokens)
    
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
            
            # Highlight query terms in the content
            if return_highlights:
                highlighted_content = highlight_query_terms(content, filtered_query_tokens)
            else:
                highlighted_content = content
            
            results.append((doc_id, score, content, highlighted_content, filtered_query_tokens))
    
    return results

def display_results(query, results):
    """Display search results in formatted output."""
    if not results:
        print(f"\nNo results found for '{query}'")
        return
    
    print(f"\n{'='*80}")
    print(f"Search Results for '{query}'")
    print(f"{'='*80}\n")
    
    for i, (doc_id, score, content, highlighted_content, query_terms) in enumerate(results, 1):
        print(f"{i}. Document: {doc_id} (Relevance Score: {score:.4f})")
        print(f"   {'-'*76}")
        
        # Display highlighted content (in terminal, underscores represent underlines)
        display_text = highlighted_content[:300] + ('...' if len(highlighted_content) > 300 else '')
        print(f"   {display_text}")
        print(f"   {'-'*76}\n")

def main():
    """Main execution for command-line interface."""
    print("=" * 80)
    print("         ENHANCED DOCUMENT SEARCH ENGINE")
    print("         Features: Spell Correction + Query Highlighting")
    print("=" * 80)
    
    # Specify dataset file
    dataset_file = "Articles.csv"
    
    # Initialize search engine
    if not initialize_search_engine(dataset_file):
        return
    
    print("\nType your query or 'exit' to quit.")
    print("Note: Query terms will be underlined in results (shown as __term__)\n")
    
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