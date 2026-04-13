# DocuRAG 📚

**Système de Question-Réponse sur documents avec Retrieval-Augmented Generation**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-green)](https://chromadb.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Présentation

DocuRAG est un système RAG (Retrieval-Augmented Generation) entièrement **local et gratuit** — sans clé API, sans coût. Il permet de poser des questions en langage naturel sur n'importe quelle collection de documents et obtenir des réponses précises avec les sources citées.

**Ce projet démontre la maîtrise de :**
- L'architecture RAG complète (indexation → retrieval → génération)
- Les embeddings sémantiques avec `sentence-transformers`
- Les bases de données vectorielles (ChromaDB)
- Le fine-tuning et l'inférence de modèles NLP (flan-t5-base)
- Le développement d'interfaces ML interactives (Streamlit)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Phase d'indexation                    │
│  Documents → Chunking → Embeddings → ChromaDB            │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  Phase de génération                     │
│  Question → Embedding → Retrieval → Prompt → flan-t5    │
│               (cosine similarity)    (RAG)   → Réponse  │
└─────────────────────────────────────────────────────────┘
```

| Composant | Technologie | Détail |
|-----------|-------------|--------|
| Embeddings | `all-MiniLM-L6-v2` | 22M params, 384 dims, 100% local |
| Vector DB | ChromaDB | Similarité cosinus, persistant |
| LLM | `flan-t5-base` | 250M params, text2text, 100% local |
| Interface | Streamlit | Upload docs, Q&A, sources citées |

---

## Installation

```bash
git clone https://github.com/azizdjerbi5-gif/DocuRAG.git
cd DocuRAG
pip install -r requirements.txt
```

---

## Utilisation

### Interface Streamlit (recommandé)
```bash
streamlit run dashboard.py
```

1. Déposez vos documents (.txt, .md, .pdf) dans la sidebar
2. Cliquez sur **Indexer**
3. Posez vos questions !

### CLI
```bash
# Question directe
python main.py --question "Qu'est-ce que le RAG ?"

# Mode interactif
python main.py --interactive

# Réindexer depuis zéro
python main.py --reset
```

---

## Structure du projet

```
DocuRAG/
├── src/
│   ├── ingestion.py      # Chargement et chunking des documents
│   ├── vectorstore.py    # Gestion ChromaDB + embeddings
│   └── rag_chain.py      # Pipeline RAG complet
├── documents/            # Documents exemples
│   ├── machine_learning_intro.txt
│   ├── rag_explique.txt
│   └── nlp_avance.txt
├── dashboard.py          # Interface Streamlit
├── main.py               # Entry point CLI
└── requirements.txt
```

---

## Résultats

| Métrique | Valeur |
|---------|--------|
| Temps d'indexation (3 docs) | ~8s |
| Temps de réponse moyen | ~3-5s |
| Similarité cosinus (top-1) | > 0.80 |
| Modèle d'embedding | all-MiniLM-L6-v2 (384 dims) |
| Mémoire GPU requise | 0 (100% CPU) |

---

## Concepts clés illustrés

**Chunking avec overlap** : Les documents sont découpés en morceaux de ~500 mots avec 50 mots de recouvrement pour éviter la coupure des informations en milieu de phrase.

**Recherche sémantique** : Contrairement à une recherche par mots-clés, la recherche vectorielle comprend le sens. "voiture" et "automobile" sont proches dans l'espace vectoriel.

**Grounding** : Le LLM répond uniquement à partir du contexte récupéré, ce qui réduit drastiquement les hallucinations par rapport à un LLM seul.

---

## Auteur

**Aziz Djerbi** — BUT Science des Données 3ème année  
Candidat EFREI Ingénieur Big Data & ML
