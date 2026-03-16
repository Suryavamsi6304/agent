"""
Retriever: formats retrieved chunks into context for the LLM.
"""

from typing import List, Dict, Tuple
from .vectorstore import get_store


def retrieve_context(query: str, n_results: int = 5) -> Tuple[str, List[Dict]]:
    """
    Retrieve relevant chunks and format them as a context string.
    Returns (context_string, raw_chunks).
    """
    store = get_store()
    chunks = store.query(query, n_results=n_results)

    if not chunks:
        return "", []

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, Page {chunk['page']} | Relevance: {chunk['score']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return context, chunks
