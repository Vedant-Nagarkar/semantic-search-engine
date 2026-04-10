import os
import sys

from src.loader       import load_documents
from src.embedder     import Embedder
from src.faiss_store  import FAISSStore
from src.chroma_store import ChromaStore
from src.answerer     import Answerer


# ─────────────────────────────────────────────
# CONSTANTS
# Paths for saving/loading indexes.
# Defined here so they're easy to change in one place.
# ─────────────────────────────────────────────

FAISS_INDEX_PATH  = "index/faiss_store/index.faiss"
CHROMA_INDEX_PATH = "index/chroma_store"
DATA_FOLDER       = "data"
TOP_K             = 3  # number of results to retrieve


# ─────────────────────────────────────────────
# SECTION 1: Build or Load Indexes
# On first run  → embed documents + build indexes
# On later runs → load indexes from disk (fast)
# ─────────────────────────────────────────────

def initialize_engine():
    """
    Initializes all components of the search engine.

    Returns:
        tuple of (embedder, faiss_store, chroma_store, answerer)

    First run logic:
        - No index exists on disk yet
        - Load documents → embed → build FAISS + ChromaDB → save
        - Takes ~10-30 seconds depending on dataset size

    Subsequent run logic:
        - Index files already exist on disk
        - Load directly — no re-embedding needed
        - Takes ~1-2 seconds
    """

    # ── Step 1: Load documents ───────────────────────────────────
    documents = load_documents(DATA_FOLDER)

    # ── Step 2: Initialize embedder ──────────────────────────────
    embedder = Embedder()

    # ── Step 3: FAISS — build or load ────────────────────────────
    faiss_store = FAISSStore(dimension=embedder.get_dimension())

    if os.path.exists(FAISS_INDEX_PATH):
        # Index already built — load from disk
        print("\n[main] FAISS index found on disk. Loading...")
        faiss_store.load(FAISS_INDEX_PATH)
    else:
        # First run — embed and build
        print("\n[main] No FAISS index found. Building from scratch...")
        embeddings = embedder.embed_documents(documents)
        faiss_store.add_documents(documents, embeddings)
        faiss_store.save(FAISS_INDEX_PATH)

    # ── Step 4: ChromaDB — build or load ─────────────────────────
    chroma_store = ChromaStore(persist_directory=CHROMA_INDEX_PATH)
    try:
        chroma_store.load()
        print("\n[main] ChromaDB collection loaded from disk.")
    except RuntimeError:
        print("\n[main] No ChromaDB collection found. Building from scratch...")
        try:
            embeddings
        except NameError:
            embeddings = embedder.embed_documents(documents)
        chroma_store.build(documents, embeddings)

    # ── Step 5: Initialize Groq answerer ─────────────────────────
    print("\n[main] Initializing Groq answerer...")
    try:
        answerer = Answerer()  # reads GROQ_API from environment
    except ValueError as e:
        print(f"\n⚠️  {e}")
        print("Search will work but answers won't be generated.")
        answerer = None

    return embedder, faiss_store, chroma_store, answerer


# ─────────────────────────────────────────────
# SECTION 2: Search and Display
# Runs one query through both FAISS and ChromaDB,
# then generates a Groq answer from the results.
# ─────────────────────────────────────────────

def run_query(query: str, embedder, faiss_store, chroma_store, answerer):
    """
    Runs a single query through the full pipeline.

    Parameters:
        query:        the user's search query string
        embedder:     initialized Embedder instance
        faiss_store:  initialized FAISSStore instance
        chroma_store: initialized ChromaStore instance
        answerer:     initialized Answerer instance (or None)
    """

    # Embed the query once — reuse for both FAISS and ChromaDB
    # This is important: embedding is the slow step.
    # We embed once and pass the same vector to both stores.
    query_vector = embedder.embed_query(query)

    # ── FAISS search ─────────────────────────────────────────────
    faiss_results  = faiss_store.search(query_vector, top_k=TOP_K)

    # ── ChromaDB search ──────────────────────────────────────────
    chroma_results = chroma_store.search(query_vector, top_k=TOP_K)

    # ── Display results ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Query: {query}")
    print(f"{'='*65}")

    print(f"\n  FAISS Results (L2 distance — lower = more similar):")
    for r in faiss_results:
        print(f"    [{r['rank']}] score={r['score']}  {r['document'][:200]}")

    print(f"\n  ChromaDB Results (cosine distance — lower = more similar):")
    for r in chroma_results:
        print(f"    [{r['rank']}] score={r['score']}  {r['document'][:200]}")

    # ── Groq answer ──────────────────────────────────────────────
    if answerer:
        # Use ChromaDB top results as context for the answer
        context_docs = [r["document"] for r in chroma_results]
        result = answerer.answer_with_sources(query, context_docs)

        print(f"\n  Answer (Groq / llama-3.1-8b-instant):")
        print(f"  {result['answer']}")

        print(f"\n  Sources used:")
        for i, src in enumerate(result["sources"], 1):
            print(f"    [{i}] {src[:200]}")
    else:
        print(f"\n  Answer: [Groq not configured — set GROQ_API env variable]")

    print(f"\n{'='*65}\n")


# ─────────────────────────────────────────────
# SECTION 3: Interactive CLI Loop
# Keeps asking for queries until the user quits.
# ─────────────────────────────────────────────

def run_cli(embedder, faiss_store, chroma_store, answerer):
    """
    Starts the interactive query loop.

    Commands:
        - Type any query and press Enter to search
        - Type 'quit' or 'exit' or press Ctrl+C to stop
        - Type 'help' to see example queries
    """
    print("\n" + "="*65)
    print("  Semantic Search Engine — Ready")
    print("="*65)
    print("  Type a query to search. Type 'quit' to exit.")
    print("  Type 'help' for example queries.")
    print("="*65 + "\n")

    example_queries = [
        "How do neural networks learn?",
        "What is a vector database?",
        "How does Python handle concurrency?",
        "What causes climate change?",
        "How does space exploration work?",
    ]

    while True:
        try:
            # Get user input
            query = input("  Query: ").strip()

            # Handle empty input
            if not query:
                print("  Please enter a query.\n")
                continue

            # Handle commands
            if query.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye.\n")
                sys.exit(0)

            if query.lower() == "help":
                print("\n  Example queries:")
                for q in example_queries:
                    print(f"    - {q}")
                print()
                continue

            # Run the query
            run_query(query, embedder, faiss_store, chroma_store, answerer)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n  Goodbye.\n")
            sys.exit(0)


# ─────────────────────────────────────────────
# SECTION 4: Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Initialize all components
    print("\n" + "="*65)
    print("  Initializing Semantic Search Engine...")
    print("="*65)

    embedder, faiss_store, chroma_store, answerer = initialize_engine()

    # Start the CLI
    run_cli(embedder, faiss_store, chroma_store, answerer)