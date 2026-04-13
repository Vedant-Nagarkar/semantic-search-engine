"""
Basic smoke tests for semantic-search-engine.
Each test confirms the function runs without crashing and returns the right type.
Run with: pytest tests/test_engine.py -v
"""

import os
import sys
import numpy as np
import pytest

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.loader import load_documents
from src.embedder import Embedder
from src.faiss_store import FAISSStore
from src.chroma_store import ChromaStore


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_docs():
    """5 hardcoded docs — no dependency on data/ folder."""
    return [
        "Machine learning enables systems to learn from data.",
        "Neural networks are inspired by the human brain.",
        "Python is a popular language for data science.",
        "FAISS is a library for efficient similarity search.",
        "ChromaDB is a vector database for AI applications.",
    ]


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture(scope="module")
def embeddings(embedder, sample_docs):
    return embedder.embed_documents(sample_docs)


# ── loader.py ─────────────────────────────────────────────────────────────────

class TestLoader:

    def test_returns_list(self):
        docs = load_documents("data/")
        assert isinstance(docs, list)

    def test_returns_nonempty(self):
        docs = load_documents("data/")
        assert len(docs) > 0

    def test_each_item_is_string(self):
        docs = load_documents("data/")
        for doc in docs:
            assert isinstance(doc, str)


# ── embedder.py ───────────────────────────────────────────────────────────────

class TestEmbedder:

    def test_embed_documents_returns_numpy(self, embedder, sample_docs):
        result = embedder.embed_documents(sample_docs)
        assert isinstance(result, np.ndarray)

    def test_embed_documents_shape(self, embedder, sample_docs):
        result = embedder.embed_documents(sample_docs)
        assert result.shape == (len(sample_docs), 384)

    def test_embed_query_returns_numpy(self, embedder):
        result = embedder.embed_query("what is machine learning")
        assert isinstance(result, np.ndarray)

    def test_embed_query_shape(self, embedder):
        result = embedder.embed_query("what is machine learning")
        assert result.shape == (384,)

    def test_get_dimension(self, embedder):
        assert embedder.get_dimension() == 384


# ── faiss_store.py ────────────────────────────────────────────────────────────

class TestFAISSStore:

    def test_add_and_search(self, embedder, sample_docs, embeddings, tmp_path):
        store = FAISSStore(dimension=384)
        store.add_documents(sample_docs, embeddings)
        query_vec = embedder.embed_query("machine learning")
        results = store.search(query_vec, top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_search_result_has_required_keys(self, embedder, sample_docs, embeddings):
        store = FAISSStore(dimension=384)
        store.add_documents(sample_docs, embeddings)
        query_vec = embedder.embed_query("machine learning")
        results = store.search(query_vec, top_k=1)
        assert "document" in results[0]
        assert "score" in results[0]

    def test_save_and_load(self, embedder, sample_docs, embeddings, tmp_path):
        store = FAISSStore(dimension=384)
        store.add_documents(sample_docs, embeddings)
        index_path = str(tmp_path / "test.faiss")
        store.save(index_path)
        new_store = FAISSStore(dimension=384)
        new_store.load(index_path)
        query_vec = embedder.embed_query("machine learning")
        results = new_store.search(query_vec, top_k=2)
        assert len(results) == 2


# ── chroma_store.py ───────────────────────────────────────────────────────────

class TestChromaStore:

    def test_build_and_search(self, embedder, sample_docs, embeddings, tmp_path):
        store = ChromaStore(persist_directory=str(tmp_path / "chroma"))
        store.build(sample_docs, embeddings)
        query_vec = embedder.embed_query("machine learning")
        results = store.search(query_vec, top_k=3)
        assert isinstance(results, list)
        assert len(results) == 3

    def test_search_result_has_required_keys(self, embedder, sample_docs, embeddings, tmp_path):
        store = ChromaStore(persist_directory=str(tmp_path / "chroma2"))
        store.build(sample_docs, embeddings)
        query_vec = embedder.embed_query("machine learning")
        results = store.search(query_vec, top_k=1)
        assert "document" in results[0]
        assert "score" in results[0]

    def test_load_after_build(self, embedder, sample_docs, embeddings, tmp_path):
        chroma_path = str(tmp_path / "chroma3")
        store = ChromaStore(persist_directory=chroma_path)
        store.build(sample_docs, embeddings)
        new_store = ChromaStore(persist_directory=chroma_path)
        new_store.load()
        query_vec = embedder.embed_query("vector database")
        results = new_store.search(query_vec, top_k=2)
        assert len(results) == 2