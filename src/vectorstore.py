"""
vectorstore.py — Gestion de la base vectorielle ChromaDB
Utilise sentence-transformers (gratuit, local) pour les embeddings
"""

import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Modèle d'embedding léger et performant (22M params, gratuit)
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "docurag_collection"
CHROMA_PATH = "chroma_db"


class VectorStore:
    """Wrapper ChromaDB + SentenceTransformer pour le stockage vectoriel."""

    def __init__(self, persist_directory: str = CHROMA_PATH):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        print(f"  Chargement du modèle d'embedding : {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"  Collection ChromaDB : '{COLLECTION_NAME}' ({self.collection.count()} chunks existants)")

    def add_documents(self, chunks: List[Dict]) -> None:
        """Encode et stocke les chunks dans ChromaDB."""
        if not chunks:
            print("  Aucun chunk à indexer.")
            return

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

        print(f"  Encodage de {len(texts)} chunks...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

        # Ajout par batches pour éviter les limites mémoire
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            self.collection.add(
                documents=texts[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                ids=ids[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )

        print(f"  {len(texts)} chunks indexés. Total collection : {self.collection.count()}")

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Recherche sémantique : retourne les chunks les plus proches."""
        query_embedding = self.embedder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for i in range(len(results["documents"][0])):
            hits.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": 1 - results["distances"][0][i]  # cosine similarity
            })

        return hits

    def reset(self) -> None:
        """Vide la collection (utile pour réindexer depuis zéro)."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("  Collection réinitialisée.")

    def count(self) -> int:
        return self.collection.count()


if __name__ == "__main__":
    from ingestion import load_documents_from_folder
    vs = VectorStore()
    chunks = load_documents_from_folder("documents")
    vs.add_documents(chunks)
    results = vs.search("intelligence artificielle", n_results=3)
    for r in results:
        print(f"\n[{r['source']}] Score: {r['score']:.3f}\n{r['text'][:150]}...")
