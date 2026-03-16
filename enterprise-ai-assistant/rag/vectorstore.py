"""
Vector store operations using ChromaDB.
Manages document embeddings for RAG retrieval.
"""

import os
import hashlib
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "enterprise_knowledge"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, lightweight, good quality


class VectorStore:
    def __init__(self):
        self._client = None
        self._collection = None
        self._embedder = None

    def _init(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        if self._embedder is None:
            self._embedder = SentenceTransformer(EMBED_MODEL_NAME)

    def add_documents(self, chunks: List[Dict]) -> int:
        """
        Add document chunks to the vector store.
        Returns number of new chunks added (skips duplicates).
        """
        self._init()
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

        ids = []
        docs = []
        embeds = []
        metas = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = hashlib.md5(chunk["text"].encode()).hexdigest()
            ids.append(doc_id)
            docs.append(chunk["text"])
            embeds.append(emb)
            metas.append({"source": chunk["source"], "page": chunk["page"]})

        # Add in batches, upsert to avoid duplicate errors
        batch_size = 100
        added = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_docs = docs[i:i + batch_size]
            batch_embeds = embeds[i:i + batch_size]
            batch_metas = metas[i:i + batch_size]

            # Check which already exist
            existing = self._collection.get(ids=batch_ids)["ids"]
            new_mask = [bid not in existing for bid in batch_ids]

            new_ids = [bid for bid, m in zip(batch_ids, new_mask) if m]
            new_docs = [d for d, m in zip(batch_docs, new_mask) if m]
            new_embeds = [e for e, m in zip(batch_embeds, new_mask) if m]
            new_metas = [md for md, m in zip(batch_metas, new_mask) if m]

            if new_ids:
                self._collection.add(
                    ids=new_ids,
                    documents=new_docs,
                    embeddings=new_embeds,
                    metadatas=new_metas
                )
                added += len(new_ids)

        return added

    def query(self, query_text: str, n_results: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query.
        Returns list of {"text": str, "source": str, "page": int, "score": float}
        """
        self._init()
        count = self._collection.count()
        if count == 0:
            return []

        n_results = min(n_results, count)
        query_embedding = self._embedder.encode([query_text], show_progress_bar=False).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 1),
                "score": round(1 - dist, 3)  # Convert distance to similarity
            })

        return retrieved

    def get_stats(self) -> Dict:
        """Return stats about the knowledge base."""
        self._init()
        count = self._collection.count()
        sources = set()
        if count > 0:
            all_metas = self._collection.get(include=["metadatas"])["metadatas"]
            sources = {m.get("source", "unknown") for m in all_metas}
        return {"total_chunks": count, "unique_sources": list(sources)}

    def clear(self):
        """Clear all documents from the knowledge base."""
        self._init()
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )


# Singleton instance
_store = VectorStore()


def get_store() -> VectorStore:
    return _store
