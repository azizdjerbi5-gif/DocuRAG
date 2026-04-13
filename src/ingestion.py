"""
ingestion.py — Chargement et découpage des documents en chunks
Supporte : .txt, .pdf, .md
"""

import os
import re
from pathlib import Path
from typing import List, Dict


def load_document(file_path: str) -> str:
    """Charge un document texte ou PDF et retourne son contenu brut."""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        try:
            import pdfplumber
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
                    text += "\n"
            return text
        except ImportError:
            raise ImportError("pdfplumber requis pour les PDFs : pip install pdfplumber")

    else:
        raise ValueError(f"Format non supporté : {ext}. Utiliser .txt, .md ou .pdf")


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> List[str]:
    """
    Découpe un texte en chunks avec overlap.
    chunk_size : nombre de mots par chunk (120 = granularité fine pour CVs/docs courts)
    overlap    : nombre de mots partagés entre chunks consécutifs
    """
    # Nettoyage basique
    text = re.sub(r'\n{3,}', '\n\n', text)
    words = text.split()

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def load_documents_from_folder(folder_path: str) -> List[Dict]:
    """
    Charge tous les documents d'un dossier et retourne une liste de dicts:
    [{"text": ..., "source": ..., "chunk_id": ...}, ...]
    """
    folder = Path(folder_path)
    all_chunks = []

    supported = [".txt", ".md", ".pdf"]
    files = [f for f in folder.iterdir() if f.suffix.lower() in supported]

    if not files:
        raise FileNotFoundError(f"Aucun fichier supporté trouvé dans {folder_path}")

    for file_path in files:
        print(f"  Chargement : {file_path.name}")
        try:
            raw_text = load_document(str(file_path))
            chunks = chunk_text(raw_text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": file_path.name,
                    "chunk_id": f"{file_path.stem}_chunk_{i}"
                })
        except Exception as e:
            print(f"  Erreur avec {file_path.name} : {e}")

    print(f"  Total chunks générés : {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    docs = load_documents_from_folder("documents")
    print(f"\nExemple de chunk :\n{docs[0]['text'][:200]}...")
