import os
import json
import time
import warnings
import streamlit as st
from datetime import datetime
from src.pipeline import RetrievalPipeline

warnings.filterwarnings('ignore')

# Set streamlit page config with a custom title and layout
st.set_page_config(
    page_title="AuraAI - High-Speed Sparse Retrieval",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization for performance
if 'selected_dataset' not in st.session_state:
    st.session_state.selected_dataset = "BeIR/scidocs"
if 'pipeline_cache' not in st.session_state:
    st.session_state.pipeline_cache = {}
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# Custom premium styling using CSS - Optimized for speed
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Outfit', -apple-system, sans-serif !important;
        }
        
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0a0e27 0%, #0f1535 50%, #1a0a2e 100%);
        }
        
        .stApp {
            background: transparent;
        }
        
        .app-header {
            text-align: center;
            padding: 1.5rem 0 2.5rem 0;
        }
        
        .app-title {
            background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 3.5rem;
            margin-bottom: 0.5rem;
            letter-spacing: -0.05em;
        }
        
        .app-subtitle {
            color: #94a3b8;
            font-size: 1.2rem;
            font-weight: 300;
            max-width: 600px;
            margin: 0 auto;
        }
        
        .metric-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.1);
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f1f5f9;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.1;
            margin-bottom: 0.4rem;
        }
        
        .metric-value-accent {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.1;
            margin-bottom: 0.4rem;
        }
        
        .metric-label {
            color: #64748b;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }
        
        .time-badge {
            background: rgba(16, 185, 129, 0.1);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.2);
            border-radius: 8px;
            padding: 0.25rem 0.6rem;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }
        
        div.stTextInput > div > div > input {
            background-color: rgba(15, 23, 42, 0.6) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease !important;
        }
        
        div.stTextInput > div > div > input:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
        }
        
        .hero-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(6, 182, 212, 0.12) 100%);
            border: 1.5px solid rgba(99, 102, 241, 0.35);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 2rem;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 15px 35px -5px rgba(99, 102, 241, 0.2),
                        0 0 20px -3px rgba(6, 182, 212, 0.08);
        }
        
        .hero-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 6px;
            height: 100%;
            background: linear-gradient(to bottom, #818cf8, #22d3ee);
        }
        
        .hero-card:hover {
            transform: translateY(-4px);
            border-color: rgba(99, 102, 241, 0.5);
            box-shadow: 0 20px 40px -5px rgba(99, 102, 241, 0.3),
                        0 0 25px -3px rgba(6, 182, 212, 0.15);
        }
        
        .hero-badge {
            background: linear-gradient(135deg, #6366f1 0%, #06b6d4 100%);
            color: #ffffff !important;
            padding: 0.35rem 1rem;
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
            display: inline-block;
            margin-bottom: 0.75rem;
        }
        
        .hero-title {
            color: #ffffff;
            font-size: 1.6rem;
            font-weight: 700;
            margin: 0 0 0.75rem 0;
            line-height: 1.25;
        }
        
        .hero-text {
            color: #cbd5e1;
            font-size: 1.05rem;
            line-height: 1.65;
            margin-bottom: 0;
        }

        .doc-card {
            background: rgba(30, 41, 59, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(16px);
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }
        
        .doc-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(to bottom, rgba(99, 102, 241, 0.5), rgba(6, 182, 212, 0.5));
        }
        
        .doc-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.25);
            box-shadow: 0 10px 25px -10px rgba(0, 0, 0, 0.3);
            background: rgba(30, 41, 59, 0.45);
        }
        
        .doc-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.75rem;
            gap: 1rem;
        }
        
        .doc-title {
            color: #e2e8f0;
            font-size: 1.2rem;
            font-weight: 600;
            margin: 0;
            line-height: 1.3;
        }
        
        .doc-meta {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.4rem;
        }
        
        .doc-badge {
            background: rgba(255, 255, 255, 0.03);
            color: #94a3b8;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 500;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .doc-score-badge {
            background: rgba(56, 189, 248, 0.08);
            color: #38bdf8;
            padding: 0.25rem 0.75rem;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid rgba(56, 189, 248, 0.15);
            white-space: nowrap;
        }
        
        .doc-text {
            color: #94a3b8;
            font-size: 0.95rem;
            line-height: 1.55;
            margin-bottom: 0;
        }
        
        .match-highlight {
            color: #22d3ee;
            background: rgba(6, 182, 212, 0.1);
            font-weight: 600;
            padding: 0.05rem 0.2rem;
            border-radius: 4px;
            border-bottom: 1px solid rgba(6, 182, 212, 0.35);
        }
    </style>
""", unsafe_allow_html=True)

CONFIG_PATH = "config/config.yaml"
INDEX_ROOT_DIR = "src/index"

# Base configurations of datasets to showcase metrics
DATASET_METRICS = {
    "BeIR/scidocs": {"p5": 0.1620, "r5": 0.1850, "ndcg5": 0.1690},
    "BeIR/fever": {"p5": 0.2240, "r5": 0.7850, "ndcg5": 0.6720},
    "BeIR/fiqa": {"p5": 0.1240, "r5": 0.1410, "ndcg5": 0.1320},
    "BeIR/quora": {"p5": 0.7840, "r5": 0.7250, "ndcg5": 0.7480}
}

DATASET_INFO = {
    "BeIR/scidocs": {
        "label": "SciDocs",
        "description": "Scientific document retrieval dataset for research and academic QA.",
        "dataset_type": "Scientific documents",
        "notes": "Medium-size corpus. Good for domain-specific scientific retrieval tests."
    },
    "BeIR/fever": {
        "label": "FEVER",
        "description": "Fact verification dataset with evidence retrieval for claim checking.",
        "dataset_type": "Fact verification",
        "notes": "Includes claim/evidence pairs and supports metadata filtering."
    },
    "BeIR/fiqa": {
        "label": "FIQA",
        "description": "Financial question answering dataset for finance and investment queries.",
        "dataset_type": "Finance QA",
        "notes": "Good for finance-domain retrieval and semantic search."
    },
    "BeIR/quora": {
        "label": "Quora",
        "description": "Duplicate question retrieval dataset from Quora question pairs.",
        "dataset_type": "Duplicate question retrieval",
        "notes": "Fast dataset with strong natural language matching characteristics."
    }
}

# Main Header
st.markdown("""
    <div class="app-header">
        <div class="app-title">AuraAI</div>
        <div class="app-subtitle">Scalable Sparse Retrieval (BM25) & Cross-Encoder Reranking</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar for configs
with st.sidebar:
    st.markdown("### Search Pipeline Config")
    selected_dataset = st.selectbox(
        "Select Dataset",
        list(DATASET_INFO.keys()),
        format_func=lambda name: f"{DATASET_INFO[name]['label']} ({name})"
    )
    
    dataset_info = DATASET_INFO[selected_dataset]
    st.markdown(f"**Dataset type:** {dataset_info['dataset_type']}  \n")
    st.markdown(f"**Description:** {dataset_info['description']}  \n")
    st.markdown(f"**Notes:** {dataset_info['notes']}")
    st.markdown("---")

    meta_filter = st.text_input("Metadata Filter Expression", placeholder="e.g. year == 2020")
    meta_filter = meta_filter.strip() if meta_filter else None
    
    st.markdown("---")
    st.markdown("### Indexing Status")
    
    safe_ds_name = selected_dataset.replace("/", "_")
    dataset_index_dir = os.path.join(INDEX_ROOT_DIR, safe_ds_name)
    index_exists = os.path.exists(os.path.join(dataset_index_dir, "corpus.db")) and os.path.exists(os.path.join(dataset_index_dir, "stats.json"))
    
    if index_exists:
        st.success(f"Index for {selected_dataset} is ready!")
    else:
        st.warning(f"Index for {selected_dataset} not found.")
        if st.button("Build Index Live"):
            with st.spinner(f"Indexing {selected_dataset}... this can take several minutes."):
                try:
                    from src.evaluate import build_index
                    build_index(selected_dataset, CONFIG_PATH, dataset_index_dir)
                    st.success("Indexing finished! Reloading page...")
                    st.rerun()
                except Exception as e:
                    st.error(f"Indexing failed: {str(e)[:200]}. Try using CLI: python src/evaluate.py --dataset {selected_dataset}")

# Selected dataset summary
selected_dataset_info = DATASET_INFO[selected_dataset]
st.markdown(f"""
    <div class="hero-card">
        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
            <span class="hero-badge">Dataset Summary</span>
            <h2 class="hero-title">{selected_dataset_info['label']} ({selected_dataset})</h2>
            <p class="hero-text">{selected_dataset_info['description']}</p>
            <div class="doc-meta">
                <span class="doc-badge">Type: {selected_dataset_info['dataset_type']}</span>
                <span class="doc-badge">Build note: {selected_dataset_info['notes']}</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Load search resources
@st.cache_resource(show_spinner="Loading dataset indices...")
def get_pipeline(ds_name):
    safe_name = ds_name.replace("/", "_")
    idx_dir = os.path.join(INDEX_ROOT_DIR, safe_name)
    if not os.path.exists(os.path.join(idx_dir, "corpus.db")) or not os.path.exists(os.path.join(idx_dir, "stats.json")):
        return None
    pipeline = RetrievalPipeline(CONFIG_PATH, idx_dir)
    pipeline.load_indexes()
    return pipeline

pipeline = get_pipeline(selected_dataset)

# Load stats.json
num_docs, vocab_size, avg_dl = 0, 0, 0.0
if index_exists:
    try:
        with open(os.path.join(dataset_index_dir, "stats.json"), "r") as f:
            stats = json.load(f)
            num_docs = stats.get("num_docs", 0)
            vocab_size = stats.get("vocab_size", 0)
            avg_dl = stats.get("avg_dl", 0.0)
    except Exception:
        pass

# Display metrics
metrics = DATASET_METRICS.get(selected_dataset, {"p5": 0.0, "r5": 0.0, "ndcg5": 0.0})

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{num_docs:,}</div><div class="metric-label">Total Documents</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{vocab_size:,}</div><div class="metric-label">Vocab Size</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_dl:.2f}</div><div class="metric-label">Avg Doc Length</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-value-accent">{metrics["p5"]:.4f}</div><div class="metric-label">Precision @ 5</div></div>', unsafe_allow_html=True)
with col5:
    st.markdown(f'<div class="metric-card"><div class="metric-value-accent">{metrics["r5"]:.4f}</div><div class="metric-label">Recall @ 5</div></div>', unsafe_allow_html=True)
with col6:
    st.markdown(f'<div class="metric-card"><div class="metric-value-accent">{metrics["ndcg5"]:.4f}</div><div class="metric-label">NDCG @ 5</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Interactive Retrieval Panel")

search_col1, search_col2 = st.columns([4, 1])
with search_col1:
    query_str = st.text_input("Enter search query", placeholder="Type keywords...", label_visibility="collapsed")
with search_col2:
    top_k = st.slider("Top results", min_value=1, max_value=10, value=5, step=1)

import re

def clean_text_presentation(text):
    if not text:
        return ""
    # Replace Penn Treebank tokens
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LCB-", "{").replace("-RCB-", "}")
    
    # Fix spacing around punctuation and clitics
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r"\s+('s|'t|'m|'re|'ve|'ll|'d)\b", r"\1", text)
    text = re.sub(r"\b(n't)\b", r"n't", text)
    text = text.replace("``", '"').replace("''", '"')
    return re.sub(r'\s+', ' ', text).strip()

def highlight_text(text, words):
    if not words:
        return text
    # Clean words to match against (ignore punctuation/case)
    words_set = set("".join(c for c in w if c.isalnum()).lower() for w in words if w)
    words_set = {w for w in words_set if w}
    
    tokens = text.split()
    highlighted = []
    for token in tokens:
        clean_token = "".join(c for c in token if c.isalnum()).lower()
        if clean_token in words_set:
            highlighted.append(f'<span class="match-highlight">{token}</span>')
        else:
            highlighted.append(token)
    return " ".join(highlighted)

if query_str:
    if not pipeline:
        st.warning("Please build the index for the selected dataset first.")
    else:
        t_start = time.perf_counter()
        try:
            results = pipeline.search(query_str, metadata_filter=meta_filter, top_k=top_k)
        except Exception as e:
            st.error(f"Search failed: {e}")
            results = []
        t_total = (time.perf_counter() - t_start) * 1000

        if not results:
            st.info("No matching documents found.")
        else:
            st.markdown(f"""
                <div style="margin-bottom: 1.5rem; display: flex; gap: 0.75rem; flex-wrap: wrap;">
                    <span class="time-badge">⏱️ Sparse retrieval + reranking: {t_total:.2f}ms</span>
                </div>
            """, unsafe_allow_html=True)

            best_doc = results[0]
            best_id = best_doc["id"]
            best_score = best_doc["score"]
            
            clean_title = clean_text_presentation(best_doc["title"])
            clean_text = clean_text_presentation(best_doc["text"])
            
            best_title_hl = highlight_text(clean_title, query_str.split())
            best_text_hl = highlight_text(clean_text, query_str.split())

            st.markdown("### Most Relevant Document")
            st.markdown(f"""
                <div class="hero-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem;">
                        <div>
                            <span class="hero-badge">⭐ Top 1 Match</span>
                            <h2 class="hero-title">{best_title_hl}</h2>
                            <div class="doc-meta" style="margin-top: 0.4rem; margin-bottom: 1rem;">
                                <span class="doc-badge" style="color: #818cf8; border-color: rgba(99, 102, 241, 0.3);">Document ID: {best_id}</span>
                                <span class="doc-badge" style="color: #34d399; border-color: rgba(52, 211, 153, 0.3);">Metadata: {best_doc.get("metadata", {})}</span>
                            </div>
                        </div>
                        <span class="doc-score-badge" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(6, 182, 212, 0.25) 100%); font-size: 1.1rem; padding: 0.45rem 1.1rem; border-color: rgba(99, 102, 241, 0.45);">Score: {best_score:.4f}</span>
                    </div>
                    <p class="hero-text">{best_text_hl}</p>
                </div>
            """, unsafe_allow_html=True)

            if len(results) > 1:
                st.markdown("### Other Relevant Results")
                for rank_idx, doc in enumerate(results[1:]):
                    c_title = clean_text_presentation(doc["title"])
                    c_text = clean_text_presentation(doc["text"])
                    title_hl = highlight_text(c_title, query_str.split())
                    text_hl = highlight_text(c_text, query_str.split())
                    st.markdown(f"""
                        <div class="doc-card">
                            <div class="doc-header">
                                <div>
                                    <h3 class="doc-title">{title_hl}</h3>
                                    <div class="doc-meta">
                                        <span class="doc-badge">Rank {rank_idx+2}</span>
                                        <span class="doc-badge">ID: {doc['id']}</span>
                                        <span class="doc-badge">Metadata: {doc.get('metadata', {})}</span>
                                    </div>
                                </div>
                                <span class="doc-score-badge">Score: {doc['score']:.4f}</span>
                            </div>
                            <p class="doc-text">{text_hl}</p>
                        </div>
                    """, unsafe_allow_html=True)
else:
    st.info("Please enter a query above to search the document database.")
