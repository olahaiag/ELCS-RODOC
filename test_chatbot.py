from rag.embeddings import EmbeddingsGenerator
from rag.vector_store import VectorStore
from rag.retreiver import Retriever

from rag.chatbot import Chatbot
from rag.prompt_builder import PromptBuilder
from rag.rag_chain import RAGChain

from rag.sources import extract_sources


# ============================================================
# INITIALISATION
# ============================================================

print("\n" + "=" * 70)
print("ASSISTANT IA ELCS RESEARCH")
print("=" * 70)


# ============================================================
# 1. EMBEDDINGS
# ============================================================

print("\nChargement du modèle d'embeddings...")

embedding = EmbeddingsGenerator()

embedding_model = (
    embedding.get_embeddings_model()
)


# ============================================================
# 2. CHARGEMENT DE FAISS
# ============================================================

print("Chargement de la base FAISS...")

vector_store = VectorStore(
    embedding_model
)

db = vector_store.load_vector_store(
    "database"
)
print("\n===== INFORMATIONS FAISS =====")

print("Type de l'index :", type(db.index))
print("Type exact :", db.index.__class__)


if hasattr(db.index, "metric_type"):
    print("Metric type :", db.index.metric_type)

if hasattr(db.index, "d"):
    print("Dimension des vecteurs :", db.index.d)

# ============================================================
# 3. RETRIEVER
# ============================================================

retriever = Retriever(
    db,
    fetch_k=20,
    distance_threshold=1.44,
    max_per_source=2
)


# ============================================================
# 4. LLM
# ============================================================

print("Chargement du LLM...")

chatbot_model = Chatbot()


# ============================================================
# 5. PROMPT BUILDER
# ============================================================

prompt_builder = PromptBuilder()


# ============================================================
# 6. RAG
# ============================================================

rag = RAGChain(
    retriever,
    chatbot_model,
    prompt_builder
)


print("\n" + "=" * 70)
print("SYSTÈME PRÊT")
print("=" * 70)

print(
    "Posez votre question."
)

print(
    "Tapez 'q' pour quitter."
)


# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

while True:

    print("\n" + "=" * 70)

    question = input(
        "Votre question : "
    ).strip()


    # --------------------------------------------------------
    # Quitter
    # --------------------------------------------------------

    if question.lower() == "q":

        print(
            "\nAu revoir !"
        )

        break


    # --------------------------------------------------------
    # Question vide
    # --------------------------------------------------------

    if not question:

        print(
            "Veuillez entrer une question."
        )

        continue


    # ========================================================
    # TEST DU RETRIEVER
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST DU RETRIEVER")
    print("=" * 70)


    results_with_scores = (
        retriever.retrieve_with_score(
            question,
            k=5
        )
    )


    for i, (
        doc,
        distance
    ) in enumerate(
        results_with_scores,
        start=1
    ):

        print(
            f"\n--- RESULTAT {i} ---"
        )

        print(
            "Distance FAISS :",
            round(
                float(distance),
                4
            )
        )

        print(
            "Fichier :",
            doc.metadata.get(
                "filename",
                "Inconnu"
            )
        )

        print(
            "Page :",
            doc.metadata.get(
                "page",
                "?"
            )
        )

        print(
            "\nContenu :"
        )

        print(
            doc.page_content[:700]
        )


    # ========================================================
    # RAG
    # ========================================================

    print("\n" + "=" * 70)
    print("GÉNÉRATION DE LA RÉPONSE")
    print("=" * 70)


    response, documents, used_documents = (
    rag.answer_question(
        question,
        k=5
    )
    )

    # ========================================================
    # RÉPONSE
    # ========================================================

    print("\n" + "=" * 70)
    print("RÉPONSE")
    print("=" * 70)

    print(
        response
    )


   # ========================================================
# SOURCES
# ========================================================

sources = extract_sources(
    used_documents
)

print("\n" + "=" * 70)
print("SOURCES")
print("=" * 70)

if sources:

    for i, source in enumerate(
        sources,
        start=1
    ):

        print(
            f"{i}. "
            f"{source['filename']} "
            f"(page {source['page']})"
        )

else:

    print(
        "Aucune source trouvée."
    )