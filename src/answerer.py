import os
from groq import Groq


# ─────────────────────────────────────────────
# SECTION 1: The Answerer Class
# ─────────────────────────────────────────────

class Answerer:
    """
    The RAG answer layer powered by Groq.

    What is RAG?
    RAG = Retrieval Augmented Generation.
    Instead of asking an LLM a question from memory alone,
    you first retrieve relevant documents, then pass them
    to the LLM as context, then ask it to answer using
    only that context.

    Why RAG instead of just asking the LLM directly?
    - Without context: LLM answers from training data → can hallucinate
    - With context:    LLM answers from your documents → grounded + accurate

    The flow:
        query
          ↓
        embedder  →  search (FAISS or ChromaDB)
          ↓
        top-k documents  →  answerer
          ↓
        Groq LLM  →  final answer

    Why Groq instead of Gemini?
    Gemini free tier has zero quota in India.
    Groq's free tier is genuinely free, works in India,
    and runs llama-3.1-8b-instant which is fast and accurate
    enough for RAG tasks.
    """

    # Model to use — defined as a constant so it's easy to change
    MODEL_NAME = "llama-3.1-8b-instant"

    def __init__(self, api_key: str = None):
        """
        Initializes the Groq client.

        Parameters:
            api_key: Groq API key as a string.
                     If None, reads from GROQ_API environment variable.

        Why support both?
        - In development: set GROQ_API as an environment variable,
          never hardcode keys in source files.
        - In testing: pass the key directly as a parameter.
        - Either way, the key never appears in committed code.

        How to set the environment variable:
            Windows:  set GROQ_API=your_key_here
            Mac/Linux: export GROQ_API=your_key_here

        Or in your .env file (if using python-dotenv):
            GROQ_API=your_key_here
        """
        # Use provided key or fall back to environment variable
        resolved_key = api_key or os.environ.get("GROQ_API")

        if not resolved_key:
            raise ValueError(
                "[answerer] GROQ_API key not found.\n"
                "Set it as an environment variable: set GROQ_API=your_key\n"
                "Or pass it directly: Answerer(api_key='your_key')"
            )

        self.client = Groq(api_key=resolved_key)
        print(f"[answerer] Groq client initialized. Model: {self.MODEL_NAME}")

    def answer(self, query: str, context_docs: list[str]) -> str:
        """
        Generates a grounded answer using retrieved documents as context.

        Parameters:
            query:        the user's original question as a plain string
            context_docs: list of retrieved document strings (top-k results)

        Returns:
            A natural language answer string grounded in the context.
            Falls back gracefully if Groq is unavailable.

        How the prompt is structured:
        We use a simple but effective RAG prompt pattern:
        1. Role instruction — tells the model what it is
        2. Context — the retrieved documents as bullet points
        3. Constraint — "answer using ONLY the context"
           This is critical. Without it, the model blends its
           training knowledge with your context, making it
           impossible to know which source the answer came from.
        4. Question — the user's query
        5. Answer: — the empty label that primes the model to respond

        Why temperature=0.2?
        Temperature controls randomness:
        - 0.0 = fully deterministic, same answer every time
        - 1.0 = creative, varied, sometimes inaccurate
        - 0.2 = mostly factual with slight natural variation
        For RAG (fact retrieval), low temperature is correct.
        You want accuracy, not creativity.

        Why max_tokens=512?
        Enough for a thorough answer (3-5 sentences) without
        wasting quota on unnecessarily long responses.
        """
        # Build context string from retrieved docs
        context = "\n".join([f"- {doc}" for doc in context_docs])

        prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say: "I don't have enough information to answer that."
Do not use any outside knowledge beyond what is in the context.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.2
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            # Never crash the pipeline — always return something useful
            fallback = context_docs[0] if context_docs else "No results found."
            return (
                f"[Groq unavailable: {type(e).__name__}]\n"
                f"Top match: {fallback}"
            )

    def answer_with_sources(self, query: str, context_docs: list[str]) -> dict:
        """
        Same as answer() but returns both the answer AND the sources used.

        Returns:
            dict with:
            - answer:  the generated answer string
            - sources: the list of context documents that were used

        Why is this useful?
        In a real application you want to show the user:
        - What the answer is
        - Which documents it came from (for verification)

        This is standard practice in production RAG systems.
        main.py uses this method instead of answer() so the CLI
        can display both the answer and its sources.
        """
        answer_text = self.answer(query, context_docs)

        return {
            "answer" : answer_text,
            "sources": context_docs
        }