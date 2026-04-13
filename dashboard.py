"""
dashboard.py — Interface Streamlit pour DocuRAG
RAG system : posez vos questions sur vos propres documents
"""

import streamlit as st
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingestion import load_documents_from_folder
from vectorstore import VectorStore
from rag_chain import RAGChain

# ──────────────────────────────────────────────
# CONFIG PAGE
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DocuRAG — Q&A sur vos documents",
    page_icon="📚",
    layout="wide"
)

# ──────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1a1a2e; }
    .sub-title   { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .answer-box  {
        background: linear-gradient(135deg, #667eea15, #764ba215);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 1rem 0;
    }
    .source-badge {
        background: #e8f4fd;
        color: #1a73e8;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
    }
    .chunk-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        margin-top: 4px;
    }
    .metric-chip {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# ÉTAT DE SESSION
# ──────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "vs" not in st.session_state:
    st.session_state.vs = None
if "history" not in st.session_state:
    st.session_state.history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

# ──────────────────────────────────────────────
# SIDEBAR — Gestion des documents
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Documents")
    st.markdown("---")

    # Upload de fichiers
    uploaded_files = st.file_uploader(
        "Déposer vos fichiers",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
        help="Formats supportés : .txt, .md, .pdf"
    )

    if uploaded_files:
        upload_dir = Path("documents/uploaded")
        upload_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files:
            dest = upload_dir / f.name
            dest.write_bytes(f.read())
        st.success(f"{len(uploaded_files)} fichier(s) prêt(s)")

    st.markdown("---")

    # Choix du dossier source
    doc_folder = st.selectbox(
        "Dossier de documents",
        ["documents", "documents/uploaded"],
        help="Dossier contenant les documents à indexer"
    )

    # Bouton d'indexation
    col1, col2 = st.columns(2)
    with col1:
        index_btn = st.button("🔄 Indexer", use_container_width=True, type="primary")
    with col2:
        reset_btn = st.button("🗑️ Reset", use_container_width=True)

    if reset_btn and st.session_state.vs:
        st.session_state.vs.reset()
        st.session_state.indexed = False
        st.session_state.history = []
        st.session_state.rag = None
        st.rerun()

    if index_btn:
        with st.spinner("Chargement et indexation des documents..."):
            try:
                chunks = load_documents_from_folder(doc_folder)
                vs = VectorStore()
                vs.reset()
                vs.add_documents(chunks)
                rag = RAGChain(vectorstore=vs, n_results=4)
                st.session_state.vs = vs
                st.session_state.rag = rag
                st.session_state.indexed = True
                st.success(f"✅ {vs.count()} chunks indexés")
            except Exception as e:
                st.error(f"Erreur : {e}")

    # Statistiques base vectorielle
    if st.session_state.vs:
        st.markdown("---")
        st.markdown("### 📊 Stats")
        count = st.session_state.vs.count()
        st.metric("Chunks indexés", count)
        st.metric("Questions posées", len(st.session_state.history))

    st.markdown("---")
    st.markdown("""
    **Architecture**
    🔹 Embeddings : `all-MiniLM-L6-v2`
    🔹 VectorDB : `ChromaDB`
    🔹 LLM : `flan-t5-base`
    🔹 Interface : `Streamlit`
    """)

# ──────────────────────────────────────────────
# MAIN — Interface Q&A
# ──────────────────────────────────────────────
st.markdown('<p class="main-title">📚 DocuRAG</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Posez vos questions sur vos propres documents grâce au Retrieval-Augmented Generation</p>', unsafe_allow_html=True)

# Schéma pipeline
with st.expander("🔍 Comment ça marche ?", expanded=False):
    st.markdown("""
    ```
    Votre question
         ↓
    [Embedding] → Vectorisation de la question
         ↓
    [ChromaDB] → Recherche des chunks les + similaires
         ↓
    [Context] → Construction du prompt enrichi
         ↓
    [flan-t5] → Génération de la réponse
         ↓
    Réponse + Sources
    ```
    **RAG = Retrieval-Augmented Generation**
    Le modèle ne répond pas de mémoire — il cherche dans VOS documents avant de répondre.
    """)

# Zone de question
if not st.session_state.indexed:
    st.info("👈 Commencez par indexer vos documents dans la barre latérale.")
else:
    st.markdown(f"**Base vectorielle active** · {st.session_state.vs.count()} chunks")

    # Questions suggérées
    st.markdown("##### 💡 Questions suggérées")
    sample_questions = [
        "De quoi parlent ces documents ?",
        "Quels sont les points clés abordés ?",
        "Résume les informations principales.",
        "Quelles sont les conclusions ?"
    ]
    cols = st.columns(4)
    chosen_sample = None
    for i, q in enumerate(sample_questions):
        if cols[i].button(q, use_container_width=True):
            chosen_sample = q

    # Input question
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Votre question",
            value=chosen_sample or "",
            placeholder="Ex : Quels sont les principaux algorithmes de ML ?"
        )
        submitted = st.form_submit_button("Envoyer →", use_container_width=False, type="primary")

    if submitted and question.strip():
        with st.spinner("Recherche dans vos documents..."):
            t0 = time.time()
            result = st.session_state.rag.ask(question)
            elapsed = time.time() - t0

        # Ajout à l'historique
        st.session_state.history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "hits": result["hits"],
            "time": elapsed
        })

    # Affichage des résultats (du plus récent au plus ancien)
    if st.session_state.history:
        for entry in reversed(st.session_state.history):
            st.markdown("---")

            # Question
            st.markdown(f"**❓ {entry['question']}**")

            # Réponse
            st.markdown(
                f'<div class="answer-box">🤖 {entry["answer"]}</div>',
                unsafe_allow_html=True
            )

            # Sources
            if entry["sources"]:
                sources_html = " ".join([
                    f'<span class="source-badge">📄 {s}</span>'
                    for s in entry["sources"]
                ])
                st.markdown(f"**Sources :** {sources_html}", unsafe_allow_html=True)

            st.caption(f"⏱️ {entry['time']:.1f}s")

            # Chunks récupérés (détail)
            with st.expander("📎 Voir les passages récupérés", expanded=False):
                for i, hit in enumerate(entry["hits"], 1):
                    score_pct = int(hit["score"] * 100)
                    st.markdown(f"""
                    <div class="chunk-box">
                        <strong>Chunk #{i}</strong> · {hit['source']} · Similarité : {score_pct}%
                        <div class="score-bar" style="width:{score_pct}%"></div>
                        <br>{hit['text'][:300]}{'...' if len(hit['text']) > 300 else ''}
                    </div>
                    """, unsafe_allow_html=True)

        # Bouton vider l'historique
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.history = []
            st.rerun()
