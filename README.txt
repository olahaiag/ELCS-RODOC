# ELCS RODOC

## Assistant IA documentaire pour ELCS Research

ELCS RODOC est un assistant intelligent permettant d'interroger
la documentation technique d'un laboratoire à l'aide d'une
architecture RAG (Retrieval-Augmented Generation).

L'objectif est de permettre aux utilisateurs de poser des questions
en langage naturel et d'obtenir des réponses basées sur les
documents techniques disponibles dans la base documentaire.

---

# 1. Architecture du projet

Le système repose sur une architecture RAG :

Documents
    ↓
Chargement des documents
    ↓
Découpage en chunks
    ↓
Génération des embeddings
    ↓
Base vectorielle FAISS
    ↓
Recherche des passages pertinents
    ↓
LLM Llama 3.2
    ↓
Réponse basée sur le contexte récupéré


Les principaux composants sont :

- DocumentLoader : chargement des documents
- DocumentSplitter : découpage des documents
- EmbeddingsGenerator : génération des embeddings
- VectorStore : gestion de FAISS
- Retriever : recherche des passages pertinents
- PromptBuilder : construction du prompt
- Chatbot : communication avec le LLM
- RAGChain : orchestration du système RAG
- Streamlit : interface utilisateur

---

# 2. Structure du projet

```text
Assistant_Elcs/
│
├── ui/
│   └── streamlit_app.py
│
├── rag/
│   ├── __init__.py
│   ├── loader.py
│   ├── splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retreiver.py
│   ├── chatbot.py
│   ├── prompt_builder.py
│   ├── rag_chain.py
│   └── sources.py
│
├── assets/
│   ├── elcs_rodoc.png
│   └── image.png
│
├── documents/
│
├── database/
│
├── requirements.txt
├── README.md
└── .gitignore