import os
import numpy as np
import faiss


# ─────────────────────────────────────────────
# SECTION 1: The FAISSStore Class
# ─────────────────────────────────────────────

class FAISSStore:
    """
    Wraps FAISS index operations: build, search, save, load.

    What is FAISS?
    FAISS (Facebook AI Similarity Search) is a library for
    fast nearest-neighbor search in high-dimensional spaces.
    It takes a query vector and finds the most similar vectors
    from a stored collection — measured by distance.

    What FAISS is NOT:
    - It is not a database. It stores vectors only, no text.
    - It has no metadata, no filtering, no persistence by default.
    - You must manually keep track of which vector maps to which document.

    This class handles all of that bookkeeping.
    """

    def __init__(self, dimension: int):
        """
        Initializes an empty FAISS index.

        Parameters:
            dimension: the size of each vector (384 for all-MiniLM-L6-v2)

        What is IndexFlatL2?
        - "Flat" means it stores all vectors as-is, no compression.
        - "L2" means it uses L2 (Euclidean) distance to measure similarity.
        - Lower L2 score = more similar vectors.
        - It is the simplest and most accurate FAISS index type.
        - Tradeoff: exact results, but slower on very large datasets (1M+ docs).
          For most real projects under 100k docs, it is fast enough.

        Why not use IndexIVFFlat or HNSW?
        Those are approximate indexes — faster but can miss results.
        For a learning project and small-medium datasets, exact search
        (IndexFlatL2) is the right choice.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)

        # FAISS stores only vectors — no text.
        # We maintain a parallel list of documents ourselves.
        # Index position in this list = vector index in FAISS.
        # So FAISS result index 5 → self.documents[5]
        self.documents = []

        print(f"[faiss] Initialized empty index. Dimension: {dimension}")

    def add_documents(self, documents: list[str], embeddings: np.ndarray):
        """
        Adds documents and their embeddings to the index.

        Parameters:
            documents:  list of original text strings
            embeddings: numpy array of shape (num_docs, dimension)

        Why must embeddings be float32?
        FAISS is written in C++ and only accepts 32-bit floats.
        Sentence Transformers returns float32 by default, but
        we cast explicitly to be safe.

        Why store documents separately?
        FAISS only stores and returns vector indices (integers).
        When search returns index 7, we look up self.documents[7]
        to get the actual text back. Without this list, you'd
        only get numbers as results, not readable text.
        """
        # Cast to float32 — FAISS requirement
        embeddings = embeddings.astype(np.float32)

        # Add vectors to FAISS index
        self.index.add(embeddings)

        # Store original text in parallel
        self.documents.extend(documents)

        print(f"[faiss] Added {len(documents)} documents. Total: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Searches the index for the most similar documents.

        Parameters:
            query_vector: numpy array of shape (384,) — the embedded query
            top_k:        number of results to return (default: 3)

        Returns:
            list of dicts, each containing:
            - rank:     result position (1 = most similar)
            - score:    L2 distance (lower = more similar)
            - document: the original text string

        How FAISS search works internally:
        1. Takes your query vector
        2. Computes L2 distance to every stored vector
        3. Returns the top_k smallest distances + their indices

        Why reshape to (1, -1)?
        faiss.index.search() expects a 2D array: (num_queries, dimension).
        embed_query() returns shape (384,) — a 1D array.
        Reshaping to (1, 384) tells FAISS: "1 query, 384 dimensions".
        Without this reshape, FAISS throws a dimension error.
        """
        # Ensure correct shape and dtype
        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        # Search — returns distances and indices arrays, both shape (1, top_k)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            # idx == -1 means FAISS couldn't find enough results
            # (happens if top_k > number of indexed documents)
            if idx == -1:
                continue

            results.append({
                "rank"    : rank + 1,
                "score"   : round(float(dist), 4),
                "document": self.documents[idx]
            })

        return results

    def save(self, path: str):
        """
        Saves the FAISS index to disk.

        Parameters:
            path: file path to save the index (e.g. "index/faiss_store/index.faiss")

        What gets saved:
        - The FAISS binary index file (vectors only)
        - A separate .txt file with one document per line

        Why save documents separately?
        faiss.write_index() only saves the vectors, not the text.
        We save documents to a plain text file alongside the index.
        On load, we read both files and reconstruct the full state.

        Why save at all?
        Without saving, you re-embed all documents every time
        you run the program. Embedding is slow (seconds to minutes
        depending on dataset size). Saving lets you build once,
        load instantly on every subsequent run.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save FAISS binary index
        faiss.write_index(self.index, path)

        # Save documents to a parallel text file
        docs_path = path.replace(".faiss", "_docs.txt")
        with open(docs_path, "w", encoding="utf-8") as f:
            for doc in self.documents:
                # Replace newlines within a doc with a space
                # so each line in the file = exactly one document
                f.write(doc.replace("\n", " ") + "\n")

        print(f"[faiss] Index saved to '{path}'")
        print(f"[faiss] Documents saved to '{docs_path}'")

    def load(self, path: str):
        """
        Loads a previously saved FAISS index from disk.

        Parameters:
            path: file path of the saved index (same path used in save())

        Raises:
            FileNotFoundError: if index file doesn't exist at given path

        After loading:
        - self.index contains all the saved vectors
        - self.documents contains all the saved text strings
        - The store is ready to search immediately
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[faiss] No index found at '{path}'. "
                f"Run build first to create it."
            )

        # Load FAISS binary index
        self.index = faiss.read_index(path)

        # Load documents from parallel text file
        docs_path = path.replace(".faiss", "_docs.txt")
        with open(docs_path, "r", encoding="utf-8") as f:
            self.documents = [line.strip() for line in f if line.strip()]

        print(f"[faiss] Loaded index from '{path}'")
        print(f"[faiss] Documents loaded: {len(self.documents)}")
        print(f"[faiss] Vectors in index: {self.index.ntotal}")