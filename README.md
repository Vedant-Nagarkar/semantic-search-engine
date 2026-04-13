# Semantic Search Engine

A full end-to-end semantic search engine built with sentence embeddings, vector databases, and a Groq-powered RAG answer layer.

## What It Does

- Takes a natural language query
- Converts it to a 384-dimensional vector using a sentence transformer model
- Searches for semantically similar documents using FAISS and ChromaDB
- Generates a grounded answer using Groq (llama-3.1-8b-instant) based on the retrieved context

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` via Sentence Transformers |
| Vector search | FAISS (L2 distance) |
| Vector database | ChromaDB (cosine distance, persistent) |
| LLM / RAG | Groq — `llama-3.1-8b-instant` |
| Interface | Interactive CLI (`main.py`) |

## Project Structure

```
semantic-search-engine/
├── data/                    # Put your .txt, .csv, or .pdf files here
├── index/
│   ├── faiss_store/         # FAISS index saved here after first run
│   └── chroma_store/        # ChromaDB collection saved here after first run
├── src/
│   ├── loader.py            # Loads documents from data/ folder
│   ├── embedder.py          # Wraps SentenceTransformer model
│   ├── faiss_store.py       # FAISS build, save, load, search
│   ├── chroma_store.py      # ChromaDB build, save, load, search
│   └── answerer.py          # Groq RAG answer layer
├── notebooks/
│   └── semantic_search.ipynb
├── main.py                  # Entry point — CLI search interface
└── requirements.txt
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Vedant-Nagarkar/semantic-search-engine.git
cd semantic-search-engine
```

### 2. Create a conda environment

```bash
conda create -n semantic-search python=3.11
conda activate semantic-search
```

> Do not use plain `venv` if Anaconda is installed — it causes DLL conflicts on Windows.

### 3. Install PyTorch via conda

```bash
conda install pytorch cpuonly -c pytorch
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Set your Groq API key

Get a free key at [console.groq.com](https://console.groq.com)

**Windows:**
```bash
set GROQ_API=your_key_here
```

**Mac/Linux:**
```bash
export GROQ_API=your_key_here
```

### 6. (Optional) Add your own documents

Drop `.txt`, `.csv`, or `.pdf` files into the `data/` folder. If the folder is empty, the engine falls back to 20 built-in sample documents.

## Running

```bash
python main.py
```

**First run** — embeds documents, builds FAISS and ChromaDB indexes, saves to disk. Takes 10–30 seconds.

**Every run after** — loads indexes from disk, no re-embedding. Takes 1–2 seconds.

### Example

```
Query: what is machine learning

FAISS Results (L2 distance — lower = more similar):
  [1] score=0.3689  Machine learning is a subset of artificial intelligence...
  [2] score=0.9872  Gradient descent is an optimization algorithm...
  [3] score=1.0331  Supervised learning uses labeled training data...

ChromaDB Results (cosine distance — lower = more similar):
  [1] score=0.1845  Machine learning is a subset of artificial intelligence...
  ...

Answer (Groq / llama-3.1-8b-instant):
Machine learning is a subset of artificial intelligence that enables systems to learn from data.
```

Type `quit` or `exit` to stop. `Ctrl+C` also works cleanly.

## How It Works

1. `loader.py` reads documents from `data/` (or uses sample docs)
2. `embedder.py` converts each document to a 384-dim normalized vector
3. `faiss_store.py` builds an `IndexFlatL2` index — exact search, no compression
4. `chroma_store.py` builds a persistent collection with cosine distance
5. At query time, the query is embedded once and searched against both indexes
6. ChromaDB results are passed to `answerer.py` as context for Groq RAG

**Why ChromaDB scores ≈ FAISS scores ÷ 2:**
When vectors are normalized, cosine distance = L2 distance ÷ 2. Both indexes are on the same scale — just different metrics.

## Requirements

```
torch==2.5.1
sentence-transformers==5.4.0
faiss-cpu==1.13.2
chromadb==1.5.7
groq==1.1.2
```

## Notes

- Groq is free and works without a credit card. Gemini was tested but has a quota of 0 in India on the free tier.
- FAISS uses L2 distance. ChromaDB uses cosine distance. Results are consistent because embeddings are normalized.
- The engine skips re-embedding on subsequent runs by persisting both indexes to disk.