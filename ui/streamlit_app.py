import sys
from pathlib import Path
import base64

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

from rag.embeddings import EmbeddingsGenerator
from rag.vector_store import VectorStore
from rag.retreiver import Retriever
from rag.chatbot import Chatbot
from rag.prompt_builder import PromptBuilder
from rag.rag_chain import RAGChain


# ============================================================
# CHEMINS DES ASSETS
# ============================================================

LOGO_PATH = ROOT_DIR / "assets" / "elcs_rodoc.png"

# Ton fichier montré dans assets
SOURCE_ICON_PATH = ROOT_DIR / "assets" / "image.png"


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="ELCS RODOC",
    page_icon=str(LOGO_PATH),
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(path):
    """
    Convertit une image en Base64
    pour pouvoir l'utiliser dans le HTML.
    """

    if not path.exists():
        return None

    return base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")


# ============================================================
# CHARGEMENT DES IMAGES
# ============================================================

logo_base64 = image_to_base64(LOGO_PATH)

source_icon_base64 = image_to_base64(
    SOURCE_ICON_PATH
)


# ============================================================
# CSS
# ============================================================

st.html("""
<style>

    /* ========================================================
       PAGE
       ======================================================== */

    .stApp {
        background: #f6f8fb;
    }

    .main .block-container {
        max-width: 900px;
        padding-top: 35px;
        padding-bottom: 50px;
    }


    /* ========================================================
       HEADER
       ======================================================== */

    .elcs-header {

        width: 100%;
        box-sizing: border-box;

        display: flex;
        align-items: center;
        justify-content: space-between;

        padding: 18px 22px;

        background: #ffffff;

        border: 1px solid #e4e8ef;

        border-radius: 18px;

        box-shadow:
            0 5px 20px rgba(0, 0, 0, 0.045);

        margin-bottom: 38px;
    }


    .header-left {

        display: flex;
        align-items: center;

        gap: 15px;
    }


    /* ========================================================
       LOGO
       ======================================================== */

    .header-logo-box {

        width: 58px;
        height: 58px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 12px;

        background: #edf2f7;

        overflow: hidden;

        flex-shrink: 0;
    }


    .header-logo {

        width: 100%;
        height: 100%;

        object-fit: contain;

        border-radius: 12px;
    }


    /* Fallback si le logo n'est pas trouvé */

    .header-logo-placeholder {

        width: 58px;
        height: 58px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 12px;

        background: #edf2f7;

        font-weight: 700;

        color: #1f4e79;
    }


    /* ========================================================
       TITRE HEADER
       ======================================================== */

    .header-title {

        font-size: 23px;

        font-weight: 700;

        color: #172033;

        line-height: 1.2;
    }


    .header-subtitle {

        margin-top: 5px;

        font-size: 13px;

        color: #737b89;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .system-status {

        display: flex;

        align-items: center;

        gap: 7px;

        padding: 7px 12px;

        border-radius: 20px;

        background: #f0f8f3;

        border: 1px solid #d7ebdf;

        color: #27734d;

        font-size: 12px;

        font-weight: 600;

        white-space: nowrap;
    }


    .status-dot {

        width: 8px;
        height: 8px;

        border-radius: 50%;

        background: #3ca66b;
    }


    /* ========================================================
       INTRO
       ======================================================== */

    .intro {

        text-align: center;

        margin-bottom: 35px;
    }


    .intro-title {

        font-size: 30px;

        font-weight: 750;

        color: #172033;

        margin-bottom: 9px;
    }


    .intro-description {

        max-width: 650px;

        margin: auto;

        font-size: 15px;

        line-height: 1.6;

        color: #707887;
    }


    /* ========================================================
       QUESTION LABEL
       ======================================================== */

    .question-label {

        font-size: 14px;

        font-weight: 650;

        color: #343c4b;

        margin-bottom: 8px;
    }


    /* ========================================================
       INPUT
       ======================================================== */

    div[data-testid="stTextInput"] input {

        height: 48px;

        border-radius: 12px;

        border: 1px solid #dfe4ec;

        background: #ffffff;

        padding-left: 15px;

        font-size: 14px;
    }


    div[data-testid="stTextInput"] input:focus {

        border-color: #1f4e79;

        box-shadow:
            0 0 0 1px #1f4e79;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    div[data-testid="stButton"] button {

        width: 100%;

        height: 46px;

        margin-top: 8px;

        border: none;

        border-radius: 11px;

        background: #1f4e79;

        color: #ffffff;

        font-size: 14px;

        font-weight: 650;

        transition: all 0.2s ease;
    }


    div[data-testid="stButton"] button:hover {

        background: #173b5c;

        color: #ffffff;

        transform: translateY(-1px);

        box-shadow:
            0 5px 15px rgba(
                31,
                78,
                121,
                0.20
            );
    }


    /* ========================================================
       SECTION TITLE
       ======================================================== */

    .section-title {

        display: flex;

        align-items: center;

        gap: 10px;

        margin-top: 35px;

        margin-bottom: 14px;

        font-size: 19px;

        font-weight: 700;

        color: #172033;
    }


    .section-marker {

        width: 5px;

        height: 22px;

        background: #1f4e79;

        border-radius: 5px;
    }


    /* ========================================================
       RESPONSE CARD
       ======================================================== */

    .response-card {

        background: #ffffff;

        border: 1px solid #e3e7ee;

        border-left: 4px solid #1f4e79;

        border-radius: 14px;

        padding: 18px 20px;

        margin-bottom: 8px;

        box-shadow:
            0 4px 15px rgba(
                0,
                0,
                0,
                0.035
            );
    }


    .response-label {

        font-size: 11px;

        font-weight: 700;

        text-transform: uppercase;

        letter-spacing: 0.6px;

        color: #1f4e79;

        margin-bottom: 8px;
    }


    /* ========================================================
       SOURCES
       ======================================================== */

    .sources-description {

        font-size: 13px;

        color: #737b89;

        margin-bottom: 14px;
    }


    .source-card {

        width: 100%;

        box-sizing: border-box;

        display: flex;

        align-items: center;

        gap: 13px;

        background: #ffffff;

        border: 1px solid #e3e7ee;

        border-radius: 12px;

        padding: 13px 15px;

        margin-bottom: 9px;

        box-shadow:
            0 2px 10px rgba(
                0,
                0,
                0,
                0.025
            );
    }


    .source-icon {

        width: 35px;

        height: 35px;

        object-fit: contain;

        flex-shrink: 0;
    }


    .source-icon-placeholder {

        width: 35px;

        height: 35px;

        display: flex;

        align-items: center;

        justify-content: center;

        border-radius: 8px;

        background: #edf2f7;

        color: #1f4e79;

        font-size: 11px;

        font-weight: 700;
    }


    .source-content {

        min-width: 0;
    }


    .source-name {

        font-size: 13px;

        font-weight: 650;

        color: #303847;

        word-break: break-word;

        line-height: 1.4;
    }


    .source-page {

        margin-top: 3px;

        font-size: 12px;

        color: #808896;
    }


    /* ========================================================
       EMPTY STATE
       ======================================================== */

    .empty-state {

        width: 100%;

        box-sizing: border-box;

        text-align: center;

        background: #ffffff;

        border: 1px dashed #d5dbe5;

        border-radius: 14px;

        padding: 25px;

        color: #7b8492;

        font-size: 13px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {

        text-align: center;

        margin-top: 45px;

        padding-top: 18px;

        border-top: 1px solid #e4e8ef;

        color: #9aa1ad;

        font-size: 11px;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 600px) {

        .main .block-container {

            padding-left: 18px;
            padding-right: 18px;
        }


        .elcs-header {

            padding: 15px;
        }


        .header-title {

            font-size: 19px;
        }


        .header-subtitle {

            font-size: 11px;
        }


        .system-status {

            display: none;
        }


        .intro-title {

            font-size: 25px;
        }


        .intro-description {

            font-size: 14px;
        }

    }

</style>
""")


# ============================================================
# HEADER
# ============================================================

if logo_base64:

    # ========================================================
    # TON LOGO PNG
    # ========================================================

    logo_html = f"""
        <div class="header-logo-box">

            <img
                src="data:image/png;base64,{logo_base64}"
                class="header-logo"
                alt="ELCS RODOC"
            >

        </div>
    """

else:

    # ========================================================
    # FALLBACK
    # ========================================================

    logo_html = """
        <div class="header-logo-placeholder">
            AI
        </div>
    """


# ============================================================
# AFFICHAGE HEADER
# ============================================================

st.html(f"""

<div class="elcs-header">

    <div class="header-left">

        {logo_html}

        <div>

            <div class="header-title">
                ELCS RODOC
            </div>

            <div class="header-subtitle">
                Assistant intelligent de documentation
            </div>

        </div>

    </div>


    <div class="system-status">

        <div class="status-dot"></div>

        Bonjour chercheur ! L'assistant est prêt à répondre à vos questions.

    </div>

</div>

""")


# ============================================================
# INTRODUCTION
# ============================================================

st.html("""

<div class="intro">

    <div class="intro-title">
        Assistant IA ELCS Research
    </div>

    <div class="intro-description">

        Posez une question sur la documentation technique
        et obtenez une réponse basée sur les documents
        disponibles dans la base documentaire ELCS.

    </div>

</div>

""")


# ============================================================
# CHARGEMENT DU RAG
# ============================================================

@st.cache_resource
def load_rag():

    # --------------------------------------------------------
    # 1. EMBEDDINGS
    # --------------------------------------------------------

    embedding = EmbeddingsGenerator()

    embedding_model = (
        embedding.get_embeddings_model()
    )


    # --------------------------------------------------------
    # 2. VECTOR STORE / FAISS
    # --------------------------------------------------------

    vector_store = VectorStore(
        embedding_model
    )

    db = vector_store.load_vector_store(
        "database"
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
    # 5. PROMPT BUILDER
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


    return rag


# ============================================================
# INITIALISATION
# ============================================================

with st.spinner(
    "Initialisation de l'assistant..."
):

    rag = load_rag()


# ============================================================
# QUESTION
# ============================================================

st.html("""

<div class="question-label">
    Votre question
</div>

""")


question = st.text_input(

    "question",

    placeholder=(
        "Exemple : Que fait python -m "
        "ensurepip --default-pip ?"
    ),

    label_visibility="collapsed"
)


# ============================================================
# BOUTON
# ============================================================

ask = st.button(
    "Rechercher dans la documentation"
)


# ============================================================
# TRAITEMENT
# ============================================================

if ask:

    # --------------------------------------------------------
    # QUESTION VIDE
    # --------------------------------------------------------

    if not question.strip():

        st.warning(
            "Veuillez saisir une question."
        )

        st.stop()


    # --------------------------------------------------------
    # RECHERCHE RAG
    # --------------------------------------------------------

    with st.spinner(
        "Recherche dans la documentation..."
    ):

        try:

            response, documents, used_documents = (

                rag.answer_question(

                    question,

                    k=5
                )
            )

        except Exception as e:

            st.error(
                "Une erreur est survenue pendant "
                "le traitement de la question."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # RÉPONSE
    # ========================================================

    st.html("""

    <div class="section-title">

        <div class="section-marker"></div>

        Réponse

    </div>

    """)


    # --------------------------------------------------------
    # IDENTIFICATION DE L'ASSISTANT
    # --------------------------------------------------------

    st.html("""

    <div class="response-card">

        <div class="response-label">

            Assistant ELCS

        </div>

    </div>

    """)


    # --------------------------------------------------------
    # RÉPONSE DU RAG / LLM
    # --------------------------------------------------------

    st.markdown(response)


    # ========================================================
    # SOURCES
    # ========================================================

    st.html("""

    <div class="section-title">

        <div class="section-marker"></div>

        Sources

    </div>

    """)


    if used_documents:

        st.html(f"""

        <div class="sources-description">

            {len(used_documents)}
            source(s) utilisée(s) pour cette réponse.

        </div>

        """)


        # ----------------------------------------------------
        # AFFICHAGE DES SOURCES
        # ----------------------------------------------------

        for doc in used_documents:

            filename = doc.metadata.get(

                "filename",

                "Document inconnu"
            )

            page = doc.metadata.get(

                "page",

                "?"
            )


            # ------------------------------------------------
            # ICÔNE SOURCE
            # ------------------------------------------------

            if source_icon_base64:

                icon_html = f"""

                    <img

                        src="data:image/png;base64,
                        {source_icon_base64}"

                        class="source-icon"

                        alt="Source"

                    >

                """

            else:

                icon_html = """

                    <div class="source-icon-placeholder">

                        PDF

                    </div>

                """


            # ------------------------------------------------
            # CARTE SOURCE
            # ------------------------------------------------

            st.html(f"""

            <div class="source-card">

                {icon_html}

                <div class="source-content">

                    <div class="source-name">

                        {filename}

                    </div>

                    <div class="source-page">

                        Page {page}

                    </div>

                </div>

            </div>

            """)


    else:

        st.html("""

        <div class="empty-state">

            Aucune source explicitement identifiée
            pour cette réponse.

        </div>

        """)


# ============================================================
# ÉTAT INITIAL
# ============================================================

if not ask:

    st.html("""

    <div
        class="empty-state"
        style="margin-top:25px;"
    >

        Entrez une question ci-dessus pour interroger
        la documentation technique ELCS.

    </div>

    """)


# ============================================================
# FOOTER
# ============================================================

st.html("""

<div class="footer">

    ELCS Research · ELCS RODOC · Assistant IA documentaire

</div>

""")