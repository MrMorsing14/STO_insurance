"""
ChromaDB-wrapper for policychunks.

VIGTIGT om e5-modeller: intfloat/multilingual-e5-* er trænet med prefixes.
Dokumenter SKAL embeddes med "passage: " og søgninger med "query: " —
ellers falder retrieval-kvaliteten markant.
"""
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import CHROMA_DIR, EMBEDDING_MODEL

COLLECTION_NAME = "policy_chunks"


class PolicyVectorStore:
    def __init__(self, persist_dir=CHROMA_DIR):
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)

    def add_chunks(self, chunks: list[dict]) -> int:
        """chunks: [{"id": str, "text": str, "metadata": dict}, ...]"""
        if not chunks:
            return 0
        texts = [f"passage: {c['text']}" for c in chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()
        self._collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            embeddings=embeddings,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        kortniveau: str,
        dækningstype: Optional[str] = None,
        n_results: int = 5,
        inkluder_generelt: bool = True,
    ) -> list[dict]:
        """
        Søg efter relevante chunks, filtreret på kortniveau og evt. dækningstype.
        'generelt'-chunks (Sektion A) medtages som default, da generelle
        undtagelser kan være afgørende for ethvert delkrav.
        """
        conditions = [{"kortniveau": kortniveau}]
        if dækningstype:
            if inkluder_generelt:
                conditions.append({"dækningstype": {"$in": [dækningstype, "generelt"]}})
            else:
                conditions.append({"dækningstype": dækningstype})

        where = {"$and": conditions} if len(conditions) > 1 else conditions[0]

        query_embedding = self._embedder.encode(
            [f"query: {query}"], show_progress_bar=False
        ).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
        )

        chunks = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks

    def chunk_count(self) -> int:
        return self._collection.count()

    def reset(self):
        """Slet og genopret collection (brug ved re-indeksering)."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
