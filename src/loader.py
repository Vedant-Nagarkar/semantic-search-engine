import os
import csv


# ─────────────────────────────────────────────
# SECTION 1: File Loaders
# Each function handles one file type.
# They all return a list of strings (one string per document/chunk).
# ─────────────────────────────────────────────

def load_txt(filepath: str) -> list[str]:
    """
    Reads a plain .txt file.
    Each non-empty line becomes one document.

    Why split by line?
    In a typical knowledge base txt file, each line is a
    self-contained fact or sentence. Keeping chunks small
    improves embedding quality — shorter = more focused vector.
    """
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # skip blank lines
                documents.append(line)
    return documents


def load_csv(filepath: str) -> list[str]:
    """
    Reads a .csv file.
    Joins all columns in each row into one string, separated by " | ".
    Each row becomes one document.

    Why join columns?
    The embedder works on plain text. If a row has columns like
    [Title, Description, Category], joining them gives the model
    full context: "Python | A general-purpose language | Programming"
    This produces a richer, more meaningful vector than embedding
    just one column.
    """
    documents = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                continue  # skip header row
            joined = " | ".join(cell.strip() for cell in row if cell.strip())
            if joined:
                documents.append(joined)
    return documents


def load_pdf(filepath: str) -> list[str]:
    """
    Reads a .pdf file page by page using PyMuPDF (fitz).
    Each page becomes one document.

    Why PyMuPDF?
    Fast, handles complex layouts, extracts clean text.
    PyPDF2 is older and breaks on many modern PDFs.

    Imported inside the function so the rest of the code
    doesn't crash if pymupdf isn't installed and no PDF is used.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    documents = []
    pdf = fitz.open(filepath)
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text().strip()
        if text:
            documents.append(text)
    pdf.close()
    return documents


# ─────────────────────────────────────────────
# SECTION 2: Folder Scanner
# Walks the data/ folder and dispatches each file
# to the right loader based on its extension.
# ─────────────────────────────────────────────

def load_from_folder(folder_path: str) -> list[str]:
    """
    Scans a folder for supported files (.txt, .csv, .pdf).
    Calls the right loader for each file.
    Returns a combined flat list of all document strings.

    "Flat list" means: if file1.txt gives 5 lines and file2.csv
    gives 8 rows, you get back one list of 13 strings — not nested.
    The embedder doesn't care which file a doc came from.
    """
    all_documents = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if not os.path.isfile(filepath):
            continue  # skip subdirectories

        ext = filename.lower().split(".")[-1]

        if ext == "txt":
            docs = load_txt(filepath)
            print(f"  [loader] Loaded {len(docs)} docs from {filename}")
            all_documents.extend(docs)

        elif ext == "csv":
            docs = load_csv(filepath)
            print(f"  [loader] Loaded {len(docs)} docs from {filename}")
            all_documents.extend(docs)

        elif ext == "pdf":
            docs = load_pdf(filepath)
            print(f"  [loader] Loaded {len(docs)} docs from {filename}")
            all_documents.extend(docs)

        else:
            print(f"  [loader] Skipping unsupported file: {filename}")

    return all_documents


# ─────────────────────────────────────────────
# SECTION 3: Fallback Sample Documents
# 20 hardcoded documents across 4 topics.
# Used when the data/ folder is empty or missing.
# ─────────────────────────────────────────────

SAMPLE_DOCUMENTS = [
    # Machine Learning (5 docs)
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Supervised learning uses labeled training data to learn a mapping from inputs to outputs.",
    "Neural networks are inspired by the human brain and consist of layers of interconnected nodes.",
    "Overfitting occurs when a model learns the training data too well and fails to generalize.",
    "Gradient descent is an optimization algorithm used to minimize the loss function in ML models.",

    # Python Programming (5 docs)
    "Python is a high-level, interpreted programming language known for its simplicity and readability.",
    "List comprehensions in Python provide a concise way to create lists using a single line of code.",
    "Decorators in Python are functions that modify the behavior of other functions.",
    "Python's GIL (Global Interpreter Lock) prevents multiple threads from executing Python bytecode simultaneously.",
    "Virtual environments in Python isolate project dependencies to avoid version conflicts.",

    # Space Exploration (5 docs)
    "The James Webb Space Telescope can observe galaxies formed shortly after the Big Bang.",
    "Mars has two small moons named Phobos and Deimos, both irregularly shaped.",
    "SpaceX's Falcon 9 is the first orbital-class rocket capable of reflight.",
    "The International Space Station orbits Earth at approximately 400 kilometers altitude.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",

    # Climate Science (5 docs)
    "The greenhouse effect is the warming of Earth's surface due to trapped heat from the atmosphere.",
    "Arctic sea ice has been declining at a rate of about 13% per decade since 1979.",
    "Carbon dioxide levels in the atmosphere have exceeded 420 ppm for the first time in human history.",
    "Renewable energy sources like solar and wind produce electricity without direct carbon emissions.",
    "Ocean acidification occurs when CO2 dissolves in seawater, lowering its pH level.",
]


# ─────────────────────────────────────────────
# SECTION 4: Main Public Function
# The only function the rest of the codebase calls.
# ─────────────────────────────────────────────

def load_documents(data_folder: str = "data") -> list[str]:
    """
    Primary entry point for document loading.

    Logic:
    1. Check if data/ folder exists
    2. If yes → scan for supported files
    3. If files found → return those documents
    4. If folder empty or missing → warn + return sample documents

    Parameters:
        data_folder: path to the data directory (default: "data")

    Returns:
        list of document strings ready for embedding
    """
    print(f"\n[loader] Scanning '{data_folder}/' for documents...")

    if os.path.exists(data_folder) and os.path.isdir(data_folder):
        documents = load_from_folder(data_folder)

        if documents:
            print(f"[loader] Total documents loaded from files: {len(documents)}")
            return documents
        else:
            print(f"[loader] No supported files found in '{data_folder}/'.")
    else:
        print(f"[loader] Folder '{data_folder}/' not found.")

    # Fallback
    print(f"[loader] Using {len(SAMPLE_DOCUMENTS)} built-in sample documents.\n")
    return SAMPLE_DOCUMENTS