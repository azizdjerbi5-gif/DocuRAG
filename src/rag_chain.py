"""
rag_chain.py — Pipeline RAG : Retrieval sémantique pur
Architecture : Question → Embedding → ChromaDB → Passages pertinents → Réponse
Pas de LLM génératif requis : la réponse EST le passage le plus pertinent trouvé.
"""

import os
from typing import List, Dict

# Seuil de similarité minimum pour considérer une réponse valide
MIN_SCORE = 0.10


class RAGChain:
    """
    Pipeline RAG retrieval-only :
    1. Retrieval  : recherche sémantique dans ChromaDB (cosine similarity)
    2. Extraction : sélection et nettoyage du passage le plus pertinent
    3. Synthèse   : résumé multi-passages si la question est générale
    """

    def __init__(self, vectorstore, n_results: int = 4):
        self.vectorstore = vectorstore
        self.n_results = n_results

    def retrieve(self, question: str) -> List[Dict]:
        """Étape 1 : Récupère les chunks les plus pertinents."""
        return self.vectorstore.search(question, n_results=self.n_results)

    def build_context(self, hits: List[Dict]) -> str:
        """Construit le contexte fusionné depuis les chunks."""
        return "\n\n".join(h["text"] for h in hits)

    def _extract_best_passage(self, question: str, hits: List[Dict]) -> str:
        """
        Étape 2 : Retourne le passage le plus pertinent (top-1 chunk).
        Si le score est trop faible, cherche dans les chunks suivants.
        """
        if not hits:
            return "Je ne trouve pas cette information dans les documents fournis."

        # Cherche le premier chunk avec un score suffisant
        for hit in hits:
            if hit["score"] >= MIN_SCORE:
                text = hit["text"].strip()
                # Retourne le texte complet du chunk (limité à 500 chars pour la lisibilité)
                return text if len(text) <= 500 else text[:500] + "..."

        return "Je ne trouve pas cette information dans les documents fournis."

    def generate(self, question: str, hits: List[Dict]) -> str:
        """Étape 3 : Construit la réponse finale."""
        return self._extract_best_passage(question, hits)

    def ask(self, question: str) -> Dict:
        """
        Pipeline complet : question → réponse avec sources.
        Retourne : {"answer": str, "sources": list, "hits": list}
        """
        if self.vectorstore.count() == 0:
            return {
                "answer": "Aucun document indexé. Veuillez d'abord charger des documents.",
                "sources": [],
                "hits": []
            }

        # Retrieval
        hits = self.retrieve(question)

        # Contexte complet
        context = self.build_context(hits)

        # Extraction de la meilleure réponse
        answer = self.generate(question, hits)

        # Sources uniques (dans l'ordre de pertinence)
        sources = list(dict.fromkeys([h["source"] for h in hits]))

        return {
            "answer": answer,
            "sources": sources,
            "hits": hits,
            "context": context
        }


def run_pipeline(documents_folder: str = "documents", question: str = None):
    """Fonction utilitaire pour lancer le pipeline en une seule commande."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ingestion import load_documents_from_folder
    from vectorstore import VectorStore

    print("\n=== DocuRAG Pipeline ===\n")

    # 1. Chargement des documents
    print("1. Chargement des documents...")
    chunks = load_documents_from_folder(documents_folder)

    # 2. Indexation
    print("\n2. Indexation vectorielle...")
    vs = VectorStore()
    if vs.count() == 0:
        vs.add_documents(chunks)
    else:
        print(f"  Base déjà indexée ({vs.count()} chunks). Utilisation du cache.")

    # 3. RAG
    print("\n3. Initialisation RAG Chain...")
    rag = RAGChain(vectorstore=vs)

    if question:
        print(f"\nQuestion : {question}")
        result = rag.ask(question)
        print(f"\nRéponse : {result['answer']}")
        print(f"Sources : {', '.join(result['sources'])}")
        return result

    return rag


if __name__ == "__main__":
    run_pipeline(question="Quels sont les principaux algorithmes de machine learning ?")
    run_pipeline(question="Qu'est-ce que le RAG ?")
