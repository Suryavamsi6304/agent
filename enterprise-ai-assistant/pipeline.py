"""
Main pipeline: orchestrates ingestion, retrieval, and generation.
"""

from typing import Iterator, List, Dict, Tuple
from rag.ingestion import load_document
from rag.vectorstore import get_store
from rag.retriever import retrieve_context
from llm.claude_client import stream_response, generate_insights


def ingest_file(file_bytes: bytes, file_name: str) -> Tuple[int, int]:
    """
    Process and add a file to the knowledge base.
    Returns (chunks_added, total_chunks_processed).
    """
    chunks = load_document(file_bytes=file_bytes, file_name=file_name)
    if not chunks:
        return 0, 0
    store = get_store()
    added = store.add_documents(chunks)
    return added, len(chunks)


def ask(
    query: str,
    chat_history: List[Dict] = None,
    n_context: int = 5,
) -> Tuple[Iterator[str], List[Dict]]:
    """
    Full RAG pipeline: retrieve context + stream answer.
    Returns (stream_iterator, retrieved_chunks).
    """
    context, chunks = retrieve_context(query, n_results=n_context)
    stream = stream_response(query, context, chat_history)
    return stream, chunks


def generate_report(topic: str, n_context: int = 8) -> Tuple[Iterator[str], List[Dict]]:
    """
    Generate a structured report on a topic using the knowledge base.
    """
    context, chunks = retrieve_context(topic, n_results=n_context)
    stream = generate_insights(context, topic)
    return stream, chunks


def get_kb_stats() -> Dict:
    """Return knowledge base statistics."""
    return get_store().get_stats()


def clear_kb():
    """Clear the entire knowledge base."""
    get_store().clear()
