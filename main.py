"""
main.py — Point d'entrée CLI pour DocuRAG
Usage: python main.py [--folder documents] [--question "..."] [--reset]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    parser = argparse.ArgumentParser(description="DocuRAG — Q&A sur vos documents")
    parser.add_argument("--folder", default="documents", help="Dossier contenant les documents")
    parser.add_argument("--question", "-q", type=str, help="Question à poser")
    parser.add_argument("--reset", action="store_true", help="Réinitialise la base vectorielle")
    parser.add_argument("--interactive", "-i", action="store_true", help="Mode interactif")
    args = parser.parse_args()

    from ingestion import load_documents_from_folder
    from vectorstore import VectorStore
    from rag_chain import RAGChain

    print("\n" + "="*50)
    print("         DocuRAG — RAG Pipeline")
    print("="*50)

    # Initialisation vectorstore
    vs = VectorStore()

    if args.reset:
        print("\nRéinitialisation de la base vectorielle...")
        vs.reset()

    # Indexation si besoin
    if vs.count() == 0:
        print(f"\nIndexation des documents depuis '{args.folder}'...")
        chunks = load_documents_from_folder(args.folder)
        vs.add_documents(chunks)
    else:
        print(f"\nBase vectorielle existante : {vs.count()} chunks.")

    # RAG chain
    rag = RAGChain(vectorstore=vs)

    # Mode question unique
    if args.question:
        print(f"\nQuestion : {args.question}")
        print("-" * 40)
        result = rag.ask(args.question)
        print(f"\nRéponse :\n{result['answer']}")
        print(f"\nSources : {', '.join(result['sources'])}")
        return

    # Mode interactif
    print("\n💬 Mode interactif (tapez 'quit' pour quitter)\n")
    while True:
        try:
            question = input("Question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir !")
            break

        if not question:
            continue

        result = rag.ask(question)
        print(f"\n🤖 Réponse : {result['answer']}")
        print(f"📄 Sources : {', '.join(result['sources'])}\n")


if __name__ == "__main__":
    main()
