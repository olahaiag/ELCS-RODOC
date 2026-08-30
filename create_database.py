from rag.loader import DocumentLoader
from rag.splitter import DocumentSplitter
from rag.embeddings import EmbeddingsGenerator
from rag.vector_store import VectorStore


print("\n" + "=" * 70)
print("CRÉATION DE LA BASE DE CONNAISSANCES")
print("=" * 70)


# ============================================================
# 1. CHARGEMENT DES DOCUMENTS
# ============================================================

print("\n[1/4] Chargement des documents...")

loader = DocumentLoader(
    "documents"
)

documents = loader.load_documents()

print(
    f"\nDocuments chargés : "
    f"{len(documents)}"
)


# ============================================================
# 2. DÉCOUPAGE
# ============================================================

print("\n[2/4] Découpage des documents...")

splitter = DocumentSplitter(
    chunk_size=600,
    chunk_overlap=80
)

chunks = splitter.split_documents(
    documents
)


# ============================================================
# 3. EMBEDDINGS
# ============================================================

print("\n[3/4] Génération des embeddings...")

embedding = EmbeddingsGenerator(    model_name="sentence-transformers/all-MiniLM-L6-v2"  # ou "sentence-transformers/all-mpnet-base-v2"
)

embedding_model = (
    embedding.get_embeddings_model()
)


# ============================================================
# 4. FAISS
# ============================================================

print("\n[4/4] Création de FAISS...")

vector_store = VectorStore(
    embedding_model
)

db = vector_store.create_vector_store(
    chunks
)

vector_store.save_vector_store(
    db,
    "database"
)


print("\n" + "=" * 70)
print("BASE FAISS CRÉÉE AVEC SUCCÈS")
print("=" * 70)

print(
    f"Documents : {len(documents)}"
)

print(
    f"Chunks : {len(chunks)}"
)

print(
    "Emplacement : database/"
)