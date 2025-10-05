# 🔍 Document Search Engine

A powerful and user-friendly document search engine built with Python, featuring both command-line and modern GUI interfaces. Uses TF-IDF vectorization and cosine similarity for intelligent document ranking.

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## ✨ Features

- **Intelligent Search**: TF-IDF (Term Frequency-Inverse Document Frequency) based ranking
- **Natural Language Processing**: Advanced text preprocessing with stemming and stopword removal
- **Two Interfaces**: 
  - Modern GUI with Tkinter
  - Command-line interface for quick searches
- **Fast Performance**: Efficient inverted index and vectorized operations
- **Cosine Similarity**: Accurate relevance scoring
- **User-Friendly**: Clean, modern interface with real-time search results

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Dependencies
```
nltk==3.8.1
scikit-learn==1.3.0
numpy==1.24.3
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/document-search-engine.git
cd document-search-engine
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create Requirements File
Create a `requirements.txt` file with:
```
nltk>=3.8.1
scikit-learn>=1.3.0
numpy>=1.24.3
```

### 4. Prepare Your Dataset
Create an `Articles.csv` file in the project directory with your documents. Format:
```
doc1: Your first document content here
doc2: Your second document content here
Or just plain text without doc IDs
```

Each line represents one document. You can either:
- Use explicit IDs: `doc_id: content`
- Or just write content (IDs will be auto-generated)

## 💻 Usage

### GUI Version (Recommended)

Launch the graphical interface:
```bash
python search_gui.py
```

**Features:**
- 🎨 Modern, intuitive interface
- 📊 Visual result cards with relevance scores
- 🔄 Real-time search as you type (press Enter)
- 📜 Scrollable results
- ⚡ Fast, responsive performance

### Command-Line Version

For quick searches in terminal:
```bash
python main.py
```

**Usage:**
```
Search: your query here
Search: machine learning algorithms
Search: exit
```

## 🏗️ Project Structure

```
document-search-engine/
│
├── main.py              # Core search engine logic
├── search_gui.py        # GUI interface
├── Articles.csv         # Your document dataset
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## 🔧 How It Works

### 1. **Text Preprocessing**
- Converts text to lowercase
- Removes punctuation and special characters
- Tokenizes into individual words
- Removes common stopwords (the, is, at, etc.)
- Stems words to their root form (running → run)

### 2. **Inverted Index**
Creates a mapping of terms to documents:
```
{
  'machine': ['doc1', 'doc5', 'doc12'],
  'learning': ['doc1', 'doc8', 'doc12'],
  ...
}
```

### 3. **TF-IDF Vectorization**
- **TF (Term Frequency)**: How often a term appears in a document
- **IDF (Inverse Document Frequency)**: How rare/common a term is across all documents
- Creates numerical vectors representing document content

### 4. **Cosine Similarity**
Measures the angle between query and document vectors:
- Score of 1.0 = Perfect match
- Score of 0.0 = No similarity

## 📊 Example Usage

### Sample Dataset (`Articles.csv`)
```
doc1: Machine learning is a subset of artificial intelligence
doc2: Deep learning uses neural networks with multiple layers
doc3: Natural language processing helps computers understand human language
doc4: Computer vision enables machines to interpret visual information
```

### Sample Searches
| Query | Expected Results |
|-------|-----------------|
| "machine learning" | doc1, doc2 (high scores) |
| "neural networks" | doc2 (high score) |
| "language" | doc3 (high score) |

## 🎨 GUI Screenshots

### Main Interface
- Clean, modern design with primary blue color scheme
- Large search bar for easy query input
- Real-time results display

### Search Results
- Document cards with relevance scores
- Content preview (300 characters max)
- Scrollable list for multiple results

## 🛠️ Customization

### Change Dataset File
Edit `main.py` or `search_gui.py`:
```python
dataset_file = "your_dataset.csv"
```

### Adjust Number of Results
Modify the `top_n` parameter:
```python
results = search(query, top_n=10)  # Show top 10 results
```

### Customize Colors (GUI)
Edit the color scheme in `search_gui.py`:
```python
self.colors = {
    'primary': '#6366f1',  # Change primary color
    'secondary': '#ec4899',
    ...
}
```

## 🐛 Troubleshooting

### NLTK Data Not Found
If you get NLTK errors:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

### File Not Found Error
Ensure `Articles.csv` exists in the same directory as the Python files.

### Import Errors
Install missing dependencies:
```bash
pip install nltk scikit-learn numpy
```

### GUI Not Displaying Properly
- Ensure you're using Python 3.7+
- Update tkinter: `sudo apt-get install python3-tk` (Linux)

## 📈 Performance Tips

1. **Large Datasets**: For datasets with 10,000+ documents, consider:
   - Using sparse matrices (already implemented)
   - Implementing pagination in GUI
   - Adding caching for frequent queries

2. **Faster Loading**: Pre-process documents and save the vectorizer:
   ```python
   import pickle
   pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
   ```

3. **Memory Optimization**: Process documents in batches for very large datasets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
git clone https://github.com/yourusername/document-search-engine.git
cd document-search-engine
pip install -r requirements.txt
```

### Areas for Contribution
- [ ] Add more preprocessing options
- [ ] Implement query suggestions
- [ ] Add export functionality (CSV, JSON)
- [ ] Support for PDF/DOCX file uploads
- [ ] Add search history
- [ ] Implement boolean search operators (AND, OR, NOT)

## 📝 License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 👨‍💻 Author

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/document-search-engine](https://github.com/yourusername/document-search-engine)

## 🙏 Acknowledgments

- NLTK team for natural language processing tools
- Scikit-learn for machine learning utilities
- Python community for excellent documentation

---

⭐ If you find this project useful, please consider giving it a star!

## 📚 Additional Resources

- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
