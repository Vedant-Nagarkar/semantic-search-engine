import os
import numpy as np
import chromadb


# ─────────────────────────────────────────────
# SECTION 1: The ChromaStore Class
# ─────────────────────────────────────────────

class ChromaStore:
    """
    Wraps ChromaDB collection operations: build, search, save, load.

    What is ChromaDB?
    ChromaDB is a full vector database — unlike FAISS which is
    just a search library. It stores vectors AND the original text
    AND metadata together in one place.

    Key differences from FAISS:
    ┌─────────────────┬──────────────────┬──────────────────┐
    │ Feature         │ FAISS            │ ChromaDB         │
    ├─────────────────┼──────────────────┼──────────────────┤
    │ Stores text     │ No (manual)      │ Yes (built-in)   │
    │ Stores metadata │ No               │ Yes              │
    │ Distance metric │ L2 (Euclidean)   │ Cosine           │
    │ Score meaning   │ Lower = better   │ Lower = better   │
    │ Persistence     │ Manual save/load │ Built-in         │
    │ Filtering       │ No               │ Yes (by metadata)│
    └─────────────────┴──────────────────┴──────────────────┘

    Why use both in this project?
    To compare them side by side. In a real project you would
    pick one. ChromaDB is better for production (metadata,
    filtering, persistence). FAISS is better for raw speed
    at very large scale (millions of vectors).
    """

    # Default collection name
    COLLECTION_NAME = "semantic_search"

    def __init__(self, persist_directory: str = "index/chroma_store"):
        """
        Initializes ChromaDB with a persistent client.

        Parameters:
            persist_directory: folder where ChromaDB stores its data
                               (default: "index/chroma_store")

        What is a persistent client?
        - chromadb.PersistentClient() saves data to disk automatically.
        - Every add/update is written to persist_directory.
        - On next run, load() reads from the same directory.
        - No manual save() needed — unlike FAISS.

        What is chromadb.Client()?
        - That's the in-memory client used in the notebook.
        - Data is lost when the program exits.
        - We use PersistentClient here for production use.
        """
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory

        # Suppress ChromaDB's noisy telemetry error logs
        import logging
        logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None  # created in build() or loaded in load()

        print(f"[chroma] Client initialized. Storage: '{persist_directory}'")

    def build(self, documents: list[str], embeddings: np.ndarray):
        """
        Creates a new ChromaDB collection and adds documents.

        Parameters:
            documents:  list of original text strings
            embeddings: numpy array of shape (num_docs, 384)

        What is a collection?
        A collection is ChromaDB's equivalent of a table in SQL
        or an index in FAISS. It holds a set of related documents
        with their vectors and metadata.

        Why cosine distance?
        Cosine distance measures the angle between vectors,
        not their magnitude. This is standard for sentence embeddings
        because the direction of a vector encodes meaning,
        not its length.

        Since we normalize embeddings in embedder.py:
        cosine distance = L2 distance / 2
        That's why ChromaDB scores are always roughly half
        of FAISS scores for the same query — same ranking,
        different scale.

        Why delete existing collection first?
        If you run the program twice, ChromaDB would throw an error
        trying to create a collection that already exists.
        Deleting first makes build() idempotent — safe to run
        multiple times without crashing.

        What are IDs?
        ChromaDB requires a unique string ID for every document.
        We use "doc_0", "doc_1", etc. — simple and predictable.
        """
        # Delete existing collection if it exists (makes rebuild safe)
        try:
            self.client.delete_collection(name=self.COLLECTION_NAME)
            print(f"[chroma] Existing collection deleted. Rebuilding...")
        except Exception:
            pass  # collection didn't exist — that's fine

        # Create fresh collection with cosine distance
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Add documents, embeddings, and IDs together
        self.collection.add(
            ids        = [f"doc_{i}" for i in range(len(documents))],
            embeddings = embeddings.tolist(),  # ChromaDB needs a Python list
            documents  = documents
        )

        print(f"[chroma] Collection built with {len(documents)} documents.")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> list[dict]:
        """
        Searches the collection for the most similar documents.

        Parameters:
            query_vector: numpy array of shape (384,) — the embedded query
            top_k:        number of results to return (default: 3)

        Returns:
            list of dicts, each containing:
            - rank:     result position (1 = most similar)
            - score:    cosine distance (lower = more similar)
            - document: the original text string

        Why wrap query_vector in a list?
        collection.query() expects query_embeddings to be a list
        of vectors (2D), not a single vector (1D).
        We pass [query_vector.tolist()] — a list containing one vector.
        The [0] when reading results unwraps that outer list.

        ChromaDB returns results as nested lists because it supports
        querying multiple vectors at once. Since we query one at a time,
        results["documents"] = [["doc1", "doc2", "doc3"]]
                                 ↑ outer list = one query
        results["documents"][0] = ["doc1", "doc2", "doc3"]
                                    ↑ inner list = results for that query
        """
        if self.collection is None:
            raise RuntimeError(
                "[chroma] No collection loaded. Run build() or load() first."
            )

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k
        )

        # Unwrap nested lists and format results
        output = []
        for rank, (doc, dist) in enumerate(
            zip(results["documents"][0], results["distances"][0])
        ):
            output.append({
                "rank"    : rank + 1,
                "score"   : round(float(dist), 4),
                "document": doc
            })

        return output

    def load(self):
        """
        Loads an existing ChromaDB collection from the persist directory.

        Unlike FAISS where we call faiss.read_index() manually,
        ChromaDB's PersistentClient handles persistence automatically.
        We just need to get the collection by name.

        Raises:
            RuntimeError: if the collection doesn't exist on disk yet.
                          This means build() hasn't been run yet.
        """
        try:
            self.collection = self.client.get_collection(
                name=self.COLLECTION_NAME
            )
            count = self.collection.count()
            print(f"[chroma] Collection loaded. Documents: {count}")

        except Exception:
            raise RuntimeError(
                f"[chroma] Collection '{self.COLLECTION_NAME}' not found. "
                f"Run build() first to create it."
            )