"""
Document ingestion: load, parse, and chunk files into text segments.
Supports PDF, DOCX, CSV, XLSX, and TXT formats.
"""

import os
import io
import pandas as pd
from typing import List, Dict


def load_document(file_path: str = None, file_bytes: bytes = None, file_name: str = "") -> List[Dict]:
    """
    Load a document from file path or bytes and return a list of text chunks with metadata.
    Each chunk: {"text": str, "source": str, "page": int}
    """
    ext = os.path.splitext(file_name or file_path or "")[-1].lower()

    if file_bytes:
        source = file_name
        data = file_bytes
    else:
        source = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            data = f.read()

    if ext == ".pdf":
        return _parse_pdf(data, source)
    elif ext in (".docx", ".doc"):
        return _parse_docx(data, source)
    elif ext == ".csv":
        return _parse_csv(data, source)
    elif ext in (".xlsx", ".xls"):
        return _parse_excel(data, source)
    elif ext in (".txt", ".md"):
        return _parse_text(data, source)
    else:
        # Try treating as plain text
        return _parse_text(data, source)


def _parse_pdf(data: bytes, source: str) -> List[Dict]:
    import fitz  # PyMuPDF
    chunks = []
    doc = fitz.open(stream=data, filetype="pdf")
    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            for chunk in _split_text(text, chunk_size=800, overlap=100):
                chunks.append({"text": chunk, "source": source, "page": page_num + 1})
    doc.close()
    return chunks


def _parse_docx(data: bytes, source: str) -> List[Dict]:
    from docx import Document
    doc = Document(io.BytesIO(data))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    chunks = []
    for i, chunk in enumerate(_split_text(full_text, chunk_size=800, overlap=100)):
        chunks.append({"text": chunk, "source": source, "page": i + 1})
    return chunks


def _parse_csv(data: bytes, source: str) -> List[Dict]:
    df = pd.read_csv(io.BytesIO(data))
    return _dataframe_to_chunks(df, source)


def _parse_excel(data: bytes, source: str) -> List[Dict]:
    df = pd.read_excel(io.BytesIO(data))
    return _dataframe_to_chunks(df, source)


def _dataframe_to_chunks(df: pd.DataFrame, source: str) -> List[Dict]:
    chunks = []
    # Summary chunk
    summary = f"Dataset: {source}\nColumns: {', '.join(df.columns)}\nRows: {len(df)}\n\nSample data:\n{df.head(20).to_string()}"
    chunks.append({"text": summary, "source": source, "page": 1})

    # Statistical summary if numeric columns exist
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        stats = f"Statistical Summary:\n{df[numeric_cols].describe().to_string()}"
        chunks.append({"text": stats, "source": source, "page": 2})

    # Row batches for large datasets
    batch_size = 50
    for i in range(0, min(len(df), 500), batch_size):
        batch_text = df.iloc[i:i + batch_size].to_string()
        chunks.append({"text": f"Rows {i}-{i+batch_size}:\n{batch_text}", "source": source, "page": i // batch_size + 3})

    return chunks


def _parse_text(data: bytes, source: str) -> List[Dict]:
    text = data.decode("utf-8", errors="replace")
    chunks = []
    for i, chunk in enumerate(_split_text(text, chunk_size=800, overlap=100)):
        chunks.append({"text": chunk, "source": source, "page": i + 1})
    return chunks


def _split_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks by character count."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at a sentence boundary
        if end < len(text):
            break_at = text.rfind(". ", start, end)
            if break_at == -1:
                break_at = text.rfind("\n", start, end)
            if break_at != -1:
                end = break_at + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks
