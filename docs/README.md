# AuraAI - Scalable Multi-Dataset Hybrid Retrieval Pipeline

A modular, production-ready, and memory-efficient hybrid retrieval system featuring:
- **Query Preprocessing**: Clean-up, stopword removal, stemming/lemmatization.
- **Spelling Correction**: Corpus-aware dictionary matching using **RapidFuzz**.
- **Sparse BM25 Indexing**: Inverted index built from scratch with early termination scoring.
- **Dense ANN Indexing**: CPU-optimized `FAISS` using `SentenceTransformers` and memory-mapped file loading.
- **Rank Fusion**: Reciprocal Rank Fusion (RRF) combining sparse and dense rankings.
- **Reranker**: Cross-Encoder batch inference on final candidates.
- **Database Backend**: SQLite file-backed corpus lookup to keep RAM overhead low.

## Setup & Dependencies

Dependencies are listed in [config/requirements.txt](file:///c:/Users/Asus/Desktop/auraAI/dataset/dataset/config/requirements.txt). To install:

```bash
pip install -r config/requirements.txt
```

## Configuration

Settings are managed via [config/config.yaml](file:///c:/Users/Asus/Desktop/auraAI/dataset/dataset/config/config.yaml). Key settings:
- `dataset_name`: Name of the dataset (e.g. `BeIR/scidocs`, `BeIR/fever`, `BeIR/fiqa`, `BeIR/quora`, `BeIR/msmarco`).
- `bm25.k1`, `bm25.b`: Tunable parameters.
- `dense.model_name`: Sentence Transformer model.
- `reranker.model_name`: Cross-Encoder model.

## Command Line Operations

### 1. Build Index & Run Evaluation

To index and evaluate a dataset:

```bash
# Evaluate Scidocs (automatic indexing if not exists)
python src/evaluate.py --dataset BeIR/scidocs

# Evaluate Fever
python src/evaluate.py --dataset BeIR/fever

# Evaluate MS MARCO (highly recommended to use evaluate_only if index already built, otherwise CPU indexing takes hours)
python src/evaluate.py --dataset BeIR/msmarco --evaluate_only
```

### 2. Interactive Search CLI

To search the indexed database interactively:

```bash
python src/interactive.py --dataset BeIR/scidocs
```

## Running the Web Dashboard

To run the updated Streamlit dashboard:

```bash
python -m streamlit run src/app.py
```
