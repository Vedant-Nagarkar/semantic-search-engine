import numpy as np
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# SECTION 1: The Embedder Class
# Wraps SentenceTransformer in a clean interface.
# The rest of the codebase never imports
# SentenceTransformer directly — only this class.
# ─────────────────────────────────────────────

class Embedder:
    """
    A wrapper around SentenceTransformer.

    Why wrap it in a class instead of calling it directly?
    - The model is heavy (~90MB). Loading it once and reusing
      it is much faster than loading it per function call.
    - If you ever want to swap models (e.g. all-mpnet-base-v2),
      you change one line here — nothing else in the codebase changes.
    - Centralizes all embedding logic in one place.
    """

    # Model name as a class constant.
    # Defined here so it's easy to find and change.
    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        """
        Loads the Sentence Transformer model into memory.

        What happens here:
        - First run: downloads ~90MB model from HuggingFace, caches it locally
        - Subsequent runs: loads from local cache instantly (~1-2 seconds)
        - Cache location: ~/.cache/huggingface/hub/
        """
        print(f"[embedder] Loading model '{self.MODEL_NAME}'...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        print(f"[embedder] Model loaded. Embedding dimension: {self.get_dimension()}")

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        """
        Converts a list of document strings into a 2D numpy array of vectors.

        Parameters:
            documents: list of plain text strings (your knowledge base)

        Returns:
            numpy array of shape (num_docs, 384)
            - num_docs rows, one per document
            - 384 columns, one per dimension of the vector

        Example:
            ["Python is great", "ML is fun"]
            → array of shape (2, 384)
            → two 384-dimensional vectors

        Why convert_to_numpy=True?
        FAISS requires numpy arrays. ChromaDB accepts lists.
        Keeping it as numpy and converting to list when needed
        (via .tolist()) is cleaner than the reverse.

        Why normalize_embeddings=True?
        Normalization scales every vector to length 1.
        This makes cosine similarity = dot product, which is faster.
        It also makes FAISS L2 distance mathematically equivalent
        to cosine distance — so both FAISS and ChromaDB are
        comparing vectors on the same scale.
        """
        print(f"[embedder] Embedding {len(documents)} documents...")
        embeddings = self.model.encode(
            documents,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True   # critical — explained above
        )
        print(f"[embedder] Done. Shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Converts a single query string into a 1D vector.

        Parameters:
            query: the user's search query as a plain string

        Returns:
            numpy array of shape (384,) — a single vector

        Why a separate function for queries?
        At search time you only have one string, not a list.
        This function handles that cleanly without forcing
        the caller to wrap it in a list and unwrap it again.

        Why the same normalize_embeddings=True?
        The query vector must be in the same vector space as
        the document vectors. If documents are normalized but
        the query isn't, distances are meaningless.
        """
        query_vector = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return query_vector

    def get_dimension(self) -> int:
        """
        Returns the embedding dimension of the loaded model.

        Why is this useful?
        FAISS requires you to specify the dimension when
        building the index: faiss.IndexFlatL2(dim)
        Rather than hardcoding 384 in faiss_store.py,
        we call embedder.get_dimension() — so if you swap
        models, the dimension updates automatically.
        """
        return self.model.get_sentence_embedding_dimension()