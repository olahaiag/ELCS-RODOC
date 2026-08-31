import sys
from pathlib import Path
import base64
import html

# ============================================================
# CHEMIN RACINE DU PROJET
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from rag.loader import DocumentLoader
from rag.splitter import DocumentSplitter
from rag.embeddings import EmbeddingsGenerator
from rag.vector_store import VectorStore

from rag.retreiver import Retriever
from rag.chatbot import Chatbot
from rag.prompt_builder import PromptBuilder
from rag.rag_chain import RAGChain


# ============================================================
# DOSSIERS
# ============================================================

DOCUMENTS_DIR = ROOT_DIR / "documents"
DATABASE_DIR = ROOT_DIR / "database"
ASSETS_DIR = ROOT_DIR / "assets"

LOGO_PATH = ASSETS_DIR / "elcs_rodoc.png"
SOURCE_ICON_PATH = ASSETS_DIR / "image.png"


# Création automatique des dossiers
DOCUMENTS_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="ELCS RODOC",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FONCTION IMAGE → BASE64
# ============================================================

def image_to_base64(path):

    if not path.exists():
        return None

    try:

        return base64.b64encode(
            path.read_bytes()
        ).decode("utf-8")

    except Exception:

        return None


# ============================================================
# CHARGEMENT DES IMAGES
# ============================================================

logo_base64 = image_to_base64(
    LOGO_PATH
)

source_icon_base64 = image_to_base64(
    SOURCE_ICON_PATH
)


# ============================================================
# CSS
# ============================================================

st.html("""
<style>

/* ============================================================
   GLOBAL
   ============================================================ */

.stApp {
    background: #f6f8fb;
}

.main .block-container {
    max-width: 1200px;
    padding-top: 25px;
    padding-bottom: 60px;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e9ef;
}

section[data-testid="stSidebar"] > div {
    padding-top: 25px;
}


/* ============================================================
   SIDEBAR BRAND
   ============================================================ */

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 25px;
}

.sidebar-logo-box {
    width: 48px;
    height: 48px;
    border-radius: 10px;
    background: #f1f4f8;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    flex-shrink: 0;
}

.sidebar-logo {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.sidebar-brand-title {
    font-size: 19px;
    font-weight: 750;
    color: #172033;
}

.sidebar-brand-subtitle {
    font-size: 11px;
    color: #7a8290;
    margin-top: 2px;
}


/* ============================================================
   DISCUSSIONS
   ============================================================ */

.conversation-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
    color: #8a92a0;
    margin-top: 25px;
    margin-bottom: 10px;
}


/* ============================================================
   MAIN HEADER
   ============================================================ */

.main-header {
    background: #ffffff;
    border: 1px solid #e3e7ee;
    border-radius: 18px;
    padding: 17px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 30px;
    box-shadow: 0 4px 18px rgba(0,0,0,.035);
}

.main-header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.main-logo-box {
    width: 55px;
    height: 55px;
    border-radius: 12px;
    background: #f1f4f8;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.main-logo {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.main-title {
    font-size: 22px;
    font-weight: 750;
    color: #172033;
}

.main-subtitle {
    font-size: 12px;
    color: #7b8491;
    margin-top: 4px;
}


/* ============================================================
   STATUS
   ============================================================ */

.status {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 12px;
    border-radius: 20px;
    background: #f0f8f3;
    border: 1px solid #d8ebdf;
    color: #27734d;
    font-size: 12px;
    font-weight: 650;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #3ca66b;
}


/* ============================================================
   WELCOME
   ============================================================ */

.welcome {
    text-align: center;
    margin-top: 60px;
    margin-bottom: 40px;
}

.welcome-title {
    font-size: 31px;
    font-weight: 750;
    color: #172033;
}

.welcome-description {
    max-width: 680px;
    margin: 12px auto;
    color: #737c8a;
    font-size: 14px;
    line-height: 1.7;
}


/* ============================================================
   CHAT
   ============================================================ */

.chat-container {
    max-width: 900px;
    margin: auto;
}


/* ============================================================
   USER MESSAGE
   ============================================================ */

.user-message {
    display: flex;
    justify-content: flex-end;
    margin: 18px 0;
}

.user-bubble {
    max-width: 75%;
    background: #1f4e79;
    color: #ffffff;
    padding: 12px 16px;
    border-radius: 16px 16px 4px 16px;
    font-size: 14px;
    line-height: 1.55;
}


/* ============================================================
   ASSISTANT MESSAGE
   ============================================================ */

.assistant-message {
    display: flex;
    align-items: flex-start;
    margin: 20px 0;
}

.assistant-icon-box {
    width: 36px;
    height: 36px;
    min-width: 36px;
    border-radius: 9px;
    background: #ffffff;
    border: 1px solid #e3e7ee;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    margin-right: 10px;
}

.assistant-icon {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.assistant-content {
    max-width: 85%;
    background: #ffffff;
    border: 1px solid #e3e7ee;
    color: #2c3442;
    padding: 15px 18px;
    border-radius: 4px 16px 16px 16px;
    font-size: 14px;
    line-height: 1.65;
    box-shadow: 0 3px 12px rgba(0,0,0,.025);
}


/* ============================================================
   SOURCES
   ============================================================ */

.sources-box {
    margin-top: 15px;
    margin-left: 46px;
    max-width: 800px;
}

.sources-title {
    font-size: 12px;
    font-weight: 700;
    color: #596273;
    margin-bottom: 9px;
}

.source-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #f8fafc;
    border: 1px solid #e4e8ee;
    border-radius: 10px;
    padding: 9px 11px;
    margin-bottom: 6px;
}

.source-icon-box {
    width: 30px;
    height: 30px;
    min-width: 30px;
    border-radius: 7px;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}

.source-icon {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.source-name {
    font-size: 12px;
    font-weight: 650;
    color: #374151;
    word-break: break-word;
}

.source-page {
    font-size: 11px;
    color: #89919e;
    margin-top: 2px;
}


/* ============================================================
   DOCUMENT MANAGEMENT
   ============================================================ */

.document-title {
    font-size: 18px;
    font-weight: 750;
    color: #172033;
}

.document-description {
    font-size: 12px;
    line-height: 1.5;
    color: #737c8a;
    margin-bottom: 15px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #e4e8ef;
    color: #9aa1ad;
    font-size: 11px;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 700px) {

    .main-header {
        padding: 14px;
    }

    .main-title {
        font-size: 18px;
    }

    .main-subtitle {
        font-size: 10px;
    }

    .status {
        display: none;
    }

    .welcome-title {
        font-size: 25px;
    }

    .user-bubble {
        max-width: 88%;
    }

    .assistant-content {
        max-width: 88%;
    }

}

</style>
""")


# ============================================================
# SESSION STATE
# ============================================================

if "conversations" not in st.session_state:

    st.session_state.conversations = {}


if "current_conversation" not in st.session_state:

    st.session_state.current_conversation = None


# ============================================================
# NOUVELLE DISCUSSION
# ============================================================

def create_new_conversation():

    conversation_id = (
        f"discussion_{len(st.session_state.conversations) + 1}"
    )

    st.session_state.conversations[
        conversation_id
    ] = {

        "title": "Nouvelle discussion",

        "messages": []

    }

    st.session_state.current_conversation = (
        conversation_id
    )


# ============================================================
# INITIALISATION DISCUSSION
# ============================================================

if not st.session_state.conversations:

    create_new_conversation()


if (
    st.session_state.current_conversation
    not in st.session_state.conversations
):

    create_new_conversation()


# ============================================================
# CHARGEMENT DU RAG
# ============================================================

@st.cache_resource
def load_rag():

    print("\nChargement du système RAG...")

    # --------------------------------------------------------
    # 1. EMBEDDINGS
    # --------------------------------------------------------

    embedding = EmbeddingsGenerator()

    embedding_model = (
        embedding.get_embeddings_model()
    )


    # --------------------------------------------------------
    # 2. VECTOR STORE
    # --------------------------------------------------------

    vector_store = VectorStore(
        embedding_model
    )


    # --------------------------------------------------------
    # Vérifier FAISS
    # --------------------------------------------------------

    index_file = DATABASE_DIR / "index.faiss"

    if not index_file.exists():

        print(
            "Base FAISS inexistante."
        )

        return None


    # --------------------------------------------------------
    # Charger FAISS
    # --------------------------------------------------------

    db = vector_store.load_vector_store(
        str(DATABASE_DIR)
    )


    # --------------------------------------------------------
    # 3. RETRIEVER
    # --------------------------------------------------------

    retriever = Retriever(

        db,

        fetch_k=20,

        distance_threshold=1.44,

        max_per_source=2

    )


    # --------------------------------------------------------
    # 4. LLM
    # --------------------------------------------------------

    chatbot_model = Chatbot()


    # --------------------------------------------------------
    # 5. PROMPT
    # --------------------------------------------------------

    prompt_builder = PromptBuilder()


    # --------------------------------------------------------
    # 6. RAG CHAIN
    # --------------------------------------------------------

    rag = RAGChain(

        retriever,

        chatbot_model,

        prompt_builder

    )


    print(
        "RAG chargé avec succès."
    )

    return rag


# ============================================================
# INDEXATION DES DOCUMENTS
# ============================================================

def index_documents():

    print("\n" + "=" * 60)
    print("INDEXATION DES DOCUMENTS")
    print("=" * 60)


    # --------------------------------------------------------
    # 1. LOADER
    # --------------------------------------------------------

    loader = DocumentLoader(
        str(DOCUMENTS_DIR)
    )

    documents = loader.load_documents()


    if not documents:

        raise ValueError(
            "Aucun document PDF trouvé."
        )


    print(
        f"Documents chargés : {len(documents)}"
    )


    # --------------------------------------------------------
    # 2. SPLITTER
    # --------------------------------------------------------

    splitter = DocumentSplitter(

        chunk_size=600,

        chunk_overlap=80

    )

    chunks = splitter.split_documents(
        documents
    )


    if not chunks:

        raise ValueError(
            "Aucun chunk généré."
        )


    print(
        f"Chunks générés : {len(chunks)}"
    )


    # --------------------------------------------------------
    # 3. EMBEDDINGS
    # --------------------------------------------------------

    embedding = EmbeddingsGenerator(

        model_name=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        )

    )

    embedding_model = (
        embedding.get_embeddings_model()
    )


    # --------------------------------------------------------
    # 4. FAISS
    # --------------------------------------------------------

    vector_store = VectorStore(
        embedding_model
    )

    db = vector_store.create_vector_store(
        chunks
    )


    # --------------------------------------------------------
    # 5. SAUVEGARDE
    # --------------------------------------------------------

    vector_store.save_vector_store(

        db,

        str(DATABASE_DIR)

    )


    print(
        "Base FAISS créée avec succès."
    )


    return (
        len(documents),
        len(chunks)
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:


    # ========================================================
    # LOGO
    # ========================================================

    if logo_base64:

        st.html(f"""

        <div class="sidebar-brand">

            <div class="sidebar-logo-box">

                <img
                    src="data:image/png;base64,{logo_base64}"
                    class="sidebar-logo"
                    alt="ELCS RODOC"
                >

            </div>

            <div>

                <div class="sidebar-brand-title">
                    ELCS RODOC
                </div>

                <div class="sidebar-brand-subtitle">
                    Assistant documentaire
                </div>

            </div>

        </div>

        """)

    else:

        st.markdown(
            "### ELCS RODOC"
        )


    # ========================================================
    # NOUVELLE DISCUSSION
    # ========================================================

    if st.button(
        "＋ Nouvelle discussion",
        use_container_width=True
    ):

        create_new_conversation()

        st.rerun()


    # ========================================================
    # HISTORIQUE
    # ========================================================

    st.html("""

    <div class="conversation-title">
        Discussions
    </div>

    """)


    for conversation_id, conversation_data in (
        st.session_state.conversations.items()
    ):

        title = conversation_data["title"]


        if title == "Nouvelle discussion":

            label = "💬 Nouvelle discussion"

        else:

            label = (
                "● "
                + title[:30]
            )


        if st.button(

            label,

            key=f"conversation_{conversation_id}",

            use_container_width=True

        ):

            st.session_state.current_conversation = (
                conversation_id
            )

            st.rerun()


    st.divider()


    # ========================================================
    # DOCUMENTS
    # ========================================================

    st.markdown(
        "### 📚 Documents"
    )


    st.caption(
        "Ajoutez les documents techniques du laboratoire."
    )


    # --------------------------------------------------------
    # UPLOAD PDF
    # --------------------------------------------------------

    uploaded_files = st.file_uploader(

        "Ajouter des documents PDF",

        type=["pdf"],

        accept_multiple_files=True

    )


    # --------------------------------------------------------
    # ENREGISTRER
    # --------------------------------------------------------

    if uploaded_files:

        if st.button(

            "📥 Ajouter les documents",

            use_container_width=True

        ):

            saved_count = 0


            for uploaded_file in uploaded_files:

                destination = (
                    DOCUMENTS_DIR
                    / uploaded_file.name
                )


                with open(
                    destination,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )


                saved_count += 1


            st.success(

                f"{saved_count} "
                f"document(s) ajouté(s)."

            )


            st.rerun()


    # ========================================================
    # DOCUMENTS EXISTANTS
    # ========================================================

    documents_files = sorted(

        DOCUMENTS_DIR.glob("*.pdf")

    )


    st.caption(

        f"{len(documents_files)} "
        f"document(s) disponible(s)"

    )


    if documents_files:

        for file in documents_files:

            st.write(
                f"📄 {file.name}"
            )


    # ========================================================
    # INDEXATION
    # ========================================================

    if st.button(

        "⚡ Indexer les documents",

        use_container_width=True

    ):

        if not documents_files:

            st.warning(
                "Ajoutez d'abord au moins un PDF."
            )

        else:

            with st.spinner(

                "Création de la base "
                "de connaissances..."

            ):

                try:

                    (
                        nb_documents,
                        nb_chunks
                    ) = index_documents()


                    # ------------------------------------------------
                    # IMPORTANT
                    # ------------------------------------------------

                    # Le RAG actuel utilise l'ancienne base
                    # éventuellement présente dans le cache.
                    #
                    # On vide donc le cache.

                    load_rag.clear()


                    st.success(

                        f"Indexation terminée : "
                        f"{nb_documents} document(s), "
                        f"{nb_chunks} chunks."

                    )


                    st.rerun()


                except Exception as e:

                    st.error(

                        "Une erreur est survenue "
                        "pendant l'indexation."

                    )

                    st.exception(e)


# ============================================================
# CHARGEMENT DU RAG
# ============================================================

with st.spinner(
    "Initialisation de l'assistant..."
):

    rag = load_rag()


# ============================================================
# HEADER PRINCIPAL
# ============================================================

if logo_base64:

    st.html(f"""

    <div class="main-header">

        <div class="main-header-left">

            <div class="main-logo-box">

                <img
                    src="data:image/png;base64,{logo_base64}"
                    class="main-logo"
                    alt="ELCS RODOC"
                >

            </div>

            <div>

                <div class="main-title">
                    ELCS RODOC
                </div>

                <div class="main-subtitle">
                    Assistant intelligent de documentation technique
                </div>

            </div>

        </div>


        <div class="status">

            <div class="status-dot"></div>

            Système actif

        </div>

    </div>

    """)

else:

    st.html("""

    <div class="main-header">

        <div>

            <div class="main-title">
                ELCS RODOC
            </div>

            <div class="main-subtitle">
                Assistant intelligent de documentation technique
            </div>

        </div>

    </div>

    """)


# ============================================================
# CONVERSATION COURANTE
# ============================================================

conversation = (
    st.session_state.conversations[
        st.session_state.current_conversation
    ]
)


# ============================================================
# PAGE D'ACCUEIL
# ============================================================

if not conversation["messages"]:

    st.html("""

    <div class="welcome">

        <div class="welcome-title">
            Comment puis-je vous aider ?
        </div>

        <div class="welcome-description">

            Posez une question sur la documentation
            technique du laboratoire.
            Les réponses sont générées à partir
            des documents indexés dans la base
            documentaire ELCS.

        </div>

    </div>

    """)


# ============================================================
# HISTORIQUE DE LA DISCUSSION
# ============================================================

for message in conversation["messages"]:

    role = message["role"]

    content = message["content"]


    # ========================================================
    # MESSAGE UTILISATEUR
    # ========================================================

    if role == "user":

        safe_content = html.escape(
            content
        )


        st.html(f"""

        <div class="user-message">

            <div class="user-bubble">

                {safe_content}

            </div>

        </div>

        """)


    # ========================================================
    # MESSAGE ASSISTANT
    # ========================================================

    else:

        if logo_base64:

            assistant_icon = f"""

            <img
                src="data:image/png;base64,{logo_base64}"
                class="assistant-icon"
                alt="AI"
            >

            """

        else:

            assistant_icon = "AI"


        st.html(f"""

        <div class="assistant-message">

            <div class="assistant-icon-box">

                {assistant_icon}

            </div>

            <div class="assistant-content">

        """)


        # ----------------------------------------------------
        # Réponse Markdown
        # ----------------------------------------------------

        st.markdown(content)


        st.html("""

            </div>

        </div>

        """)


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = message.get(
            "sources",
            []
        )


        if sources:

            st.html("""

            <div class="sources-box">

                <div class="sources-title">
                    📚 Sources
                </div>

            """)


            for source in sources:

                filename = html.escape(
                    str(
                        source.get(
                            "filename",
                            "Document inconnu"
                        )
                    )
                )


                page = html.escape(
                    str(
                        source.get(
                            "page",
                            "?"
                        )
                    )
                )


                # ------------------------------------------------
                # ICONE
                # ------------------------------------------------

                if source_icon_base64:

                    icon = f"""

                    <div class="source-icon-box">

                        <img
                            src="data:image/png;base64,{source_icon_base64}"
                            class="source-icon"
                            alt="Source"
                        >

                    </div>

                    """

                else:

                    icon = """

                    <div class="source-icon-box">

                        📄

                    </div>

                    """


                # ------------------------------------------------
                # CARTE
                # ------------------------------------------------

                st.html(f"""

                <div class="source-card">

                    {icon}

                    <div>

                        <div class="source-name">

                            {filename}

                        </div>

                        <div class="source-page">

                            Page {page}

                        </div>

                    </div>

                </div>

                """)


            st.html("""

            </div>

            """)


# ============================================================
# ZONE DE CHAT
# ============================================================

question = st.chat_input(

    "Posez votre question sur la documentation..."

)


# ============================================================
# TRAITEMENT DE LA QUESTION
# ============================================================

if question:

    question = question.strip()


    if not question:

        st.warning(
            "Veuillez saisir une question."
        )

        st.stop()


    # ========================================================
    # VERIFICATION BASE FAISS
    # ========================================================

    if rag is None:

        st.warning(

            "La base documentaire n'est pas encore créée. "
            "Ajoutez vos documents PDF puis cliquez sur "
            "'Indexer les documents' dans le menu de gauche."

        )

        st.stop()


    # ========================================================
    # TITRE AUTOMATIQUE
    # ========================================================

    if (
        conversation["title"]
        == "Nouvelle discussion"
    ):

        title = question[:35]

        if len(question) > 35:

            title += "..."


        conversation["title"] = title


    # ========================================================
    # AJOUTER QUESTION
    # ========================================================

    conversation["messages"].append({

        "role": "user",

        "content": question

    })


    # ========================================================
    # AFFICHAGE QUESTION
    # ========================================================

    with st.chat_message("user"):

        st.write(question)


    # ========================================================
    # RAG
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(

            "Recherche dans la documentation..."

        ):

            try:

                (
                    response,
                    documents,
                    used_documents
                ) = rag.answer_question(

                    question,

                    k=5

                )


            except Exception as e:

                st.error(

                    "Une erreur est survenue "
                    "pendant la génération de la réponse."

                )

                st.exception(e)

                st.stop()


        # ----------------------------------------------------
        # REPONSE
        # ----------------------------------------------------

        st.markdown(response)


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        sources = []


        for doc in used_documents:

            sources.append({

                "filename": doc.metadata.get(

                    "filename",

                    "Document inconnu"

                ),

                "page": doc.metadata.get(

                    "page",

                    "?"

                )

            })


        if sources:

            st.markdown(
                "**📚 Sources**"
            )


            for source in sources:

                st.caption(

                    f"📄 "
                    f"{source['filename']} "
                    f"— page "
                    f"{source['page']}"

                )


    # ========================================================
    # SAUVEGARDER LA REPONSE
    # ========================================================

    conversation["messages"].append({

        "role": "assistant",

        "content": response,

        "sources": sources

    })


    # ========================================================
    # RAFRAICHIR
    # ========================================================

    st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.html("""

<div class="footer">

    ELCS Research · ELCS RODOC ·
    Assistant IA documentaire

</div>

""")