import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from rag.embeddings import EmbeddingsGenerator
from rag.vector_store import VectorStore
from rag.retreiver import Retriever
from rag.chatbot import chatbot
from rag.prompt_builder import PromptBuilder
from rag.rag_chain import RAGChain


st.set_page_config(
    page_title="Assistant IA ELCS Research",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def load_rag():

    # Embeddings
    embedding = EmbeddingsGenerator()

    # Vector store
    vector_store = VectorStore(
        embedding.get_embeddings_model()
    )

    # Charger FAISS existant
    db = vector_store.load_vector_store(
        "database"
    )

    # Retriever
    retriever = Retriever(db)

    # LLM
    chatbot_model = chatbot()

    # Prompt
    prompt_builder = PromptBuilder()

    # RAG
    rag = RAGChain(
        retriever,
        chatbot_model,
        prompt_builder
    )

    return rag


# Charger le RAG
rag = load_rag()


# Interface
st.title("🤖 Assistant IA ELCS Research")

st.write(
    "Posez une question concernant la documentation technique."
)


question = st.text_input(
    "Votre question :"
)


if question:

    with st.spinner("Recherche dans la documentation..."):

        response, sources = rag.answer_question(
            question,
            k=5
        )

    st.subheader("Réponse")

    st.write(response)

    st.subheader("📚 Sources")

    if sources:

        for i, doc in enumerate(sources, start=1):

            filename = doc.metadata.get(
                "filename",
                "Inconnu"
            )

            page = doc.metadata.get(
                "page",
                "?"
            )

            st.write(
                f"**{i}.** {filename} — page {page}"
            )

    else:

        st.write("Aucune source trouvée.")
        