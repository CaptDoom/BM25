# AuraAI Quick Start Guide

## 🚀 Launch the Application

### Step 1: Ensure dependencies are installed
```bash
pip install -r config/requirements.txt
```

### Step 2: Launch the app
```bash
python run_app.py
```

This will automatically:
- Set up Python path
- Start Streamlit server
- Open browser at `http://localhost:8501`

---

## 📊 Using the Application

### 1. Select a Dataset
In the sidebar, choose from available BeIR datasets:
- **SciDocs**: Scientific document retrieval
- **FEVER**: Fact verification & evidence retrieval
- **FIQA**: Financial Q&A
- **Quora**: Duplicate question retrieval

### 2. Build Index (if needed)
If index doesn't exist, click "Build Index Live" button
- Small datasets (< 1M docs): ~1-5 minutes
- Large datasets: Use CLI instead (see below)

### 3. Search Documents
- Enter your query in the search box
- Adjust "Top results" slider (1-10)
- (Optional) Add metadata filter: `field == value`

### 4. View Results
- **Top match**: Highlighted with full details
- **Other results**: Ranked by relevance
- **Metrics**: Scores, IDs, metadata

---

## 📈 Command Line Interface

### Interactive CLI
```bash
python src/interactive.py
```
Features:
- REPL-style search
- No UI overhead
- Perfect for batch queries

### Evaluate & Index (Advanced)
```bash
# Build index for specific dataset
python src/evaluate.py --dataset BeIR/quora

# Run evaluation with qrels
python src/evaluate.py --dataset BeIR/quora --evaluate

# Custom parameters for SciDocs
python src/evaluate.py --dataset BeIR/scidocs --batch-size 10000
```

---

## 🔧 Configuration

### Key Settings
Edit `config/config.yaml` to customize:

```yaml
# Default dataset
dataset_name: "BeIR/scidocs"

# Query preprocessing
preprocessing:
  lowercase: true           # Convert to lowercase
  remove_punctuation: true  # Remove special chars
  remove_stopwords: true    # Remove common words
  
# Spell correction
spell_correction:
  min_similarity: 0.85      # Fuzzy matching threshold

# BM25 tuning
bm25:
  k1: 1.5    # Term frequency parameter
  b: 0.75    # Document length normalization
  
# Results tuning
reranker:
  top_k: 10  # Number of final results
```

---

## 🧪 Testing

### Run Full Test Suite
```bash
python -m pytest tests/test_comprehensive.py -v
```

### Run Specific Tests
```bash
# Data loading tests
python -m pytest tests/test_comprehensive.py::TestDataLoading -v

# BM25 performance tests
python -m pytest tests/test_comprehensive.py::TestBM25Pipeline -v

# Performance benchmarks
python -m pytest tests/test_comprehensive.py::TestPerformanceBenchmarks -v
```

### Run Original Tests
```bash
python -m unittest tests.test_pipeline
```

---

## 📊 Example Queries

### Scientific Documents
- "machine learning algorithms"
- "deep neural networks"
- "natural language processing"

### Financial Topics
- "investment strategies"
- "stock market analysis"
- "portfolio management"

### General Knowledge
- "climate change"
- "artificial intelligence"
- "renewable energy"

---

## 🎯 Performance Tips

### Speed Up Search
1. Use shorter queries (2-4 words)
2. Avoid very common words
3. Be specific with terminology

### Improve Relevance
1. Use exact keywords from documents
2. Add metadata filters when available
3. Adjust top-k (lower = better precision)

### Better Filtering
Filter by metadata when available:
```
year == 2020
rating >= 4.5
category != "spam"
```

---

## 📚 Project Structure

```
.
├── src/
│   ├── app.py                 # Streamlit web app
│   ├── pipeline.py            # Search pipeline
│   ├── preprocessing.py       # Query cleaning
│   ├── interactive.py         # CLI interface
│   ├── evaluate.py            # Indexing
│   ├── data_loader.py         # Dataset loading
│   ├── reranker.py            # Cross-encoder
│   └── retrievers/
│       └── bm25.py            # BM25 implementation
├── config/
│   └── config.yaml            # Configuration
├── tests/
│   ├── test_pipeline.py       # Original tests
│   └── test_comprehensive.py  # Full test suite
├── run_app.py                 # Application launcher
└── IMPLEMENTATION_STATUS.md   # Full report
```

---

## 🔍 Advanced Usage

### Using with Python Code
```python
from src.pipeline import RetrievalPipeline

# Initialize pipeline
pipeline = RetrievalPipeline("config/config.yaml", "src/index/BeIR_quora")
pipeline.load_indexes()

# Search documents
results = pipeline.search("python programming", top_k=10)

# Results structure
for result in results:
    print(f"ID: {result['id']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Title: {result['title']}")
    print(f"Text: {result['text'][:200]}...")
```

### Building Custom Index
```python
from src.retrievers.bm25 import ShardedBM25
from src.data_loader import DataLoader

# Load dataset
loader = DataLoader("BeIR/quora")
corpus = loader.load_corpus()

# Build index
index = ShardedBM25("custom_index", shard_size=500000)
index.index(corpus)
index.load()

# Search
results = index.search(["python", "programming"], top_k=10)
```

---

## 📞 Support

### Common Issues

**Q: App won't start**
```bash
# Use launcher script instead
python run_app.py

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m streamlit run src/app.py
```

**Q: Index building too slow**
- Use CLI for large datasets (> 1M docs)
- Increase batch size in code
- Consider indexing on stronger hardware

**Q: Search returns no results**
- Verify index exists: `ls src/index/`
- Build index if missing
- Check metadata filter syntax
- Try simpler query terms

**Q: Search is slow**
- Smaller top-k value (fewer results to rerank)
- Check system RAM usage
- Verify no other heavy processes

---

## 📖 Documentation

- Full implementation status: `IMPLEMENTATION_STATUS.md`
- BM25 details: See `src/retrievers/bm25.py`
- Configuration options: `config/config.yaml`
- Test coverage: `tests/test_comprehensive.py`

---

**Ready to search!** 🎉

Start with: `python run_app.py`
