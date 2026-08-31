from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader
)

from rag.document_splitter import DocumentSplitter
from rag.vector_store import VectorStore


class DocumentManager:
    """
    Gère les documents techniques du laboratoire.

    Fonctionnement :

        PDF
         ↓
        PyPDFLoader
         ↓
        DocumentSplitter
         ↓
        Embeddings
         ↓
        FAISS
    """

    def __init__(
        self,
        embedding_model,
        documents_path="documents",
        database_path="database"
    ):

        self.documents_path = Path(
            documents_path
        )

        self.database_path = Path(
            database_path
        )

        self.embedding_model = (
            embedding_model
        )

        self.splitter = DocumentSplitter(
            chunk_size=600,
            chunk_overlap=80
        )

        self.vector_store_manager = (
            VectorStore(
                embedding_model
            )
        )

        # Créer le dossier documents
        self.documents_path.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # AJOUTER UN PDF
    # ========================================================

    def add_pdf(
        self,
        pdf_path,
        vector_store
    ):

        pdf_path = Path(pdf_path)

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"Document introuvable : "
                f"{pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":

            raise ValueError(
                "Seuls les fichiers PDF "
                "sont acceptés."
            )

        print(
            f"Chargement du document : "
            f"{pdf_path.name}"
        )

        # ----------------------------------------------------
        # 1. CHARGER LE PDF
        # ----------------------------------------------------

        loader = PyPDFLoader(
            str(pdf_path)
        )

        documents = loader.load()

        if not documents:

            raise ValueError(
                "Aucun contenu trouvé dans "
                "le document PDF."
            )

        # ----------------------------------------------------
        # 2. AJOUTER LES MÉTADONNÉES
        # ----------------------------------------------------

        for document in documents:

            document.metadata[
                "filename"
            ] = pdf_path.name

        # ----------------------------------------------------
        # 3. CHUNKING
        # ----------------------------------------------------

        chunks = (
            self.splitter.split_documents(
                documents
            )
        )

        if not chunks:

            raise ValueError(
                "Aucun chunk généré."
            )

        # ----------------------------------------------------
        # 4. AJOUTER À FAISS
        # ----------------------------------------------------

        vector_store = (
            self.vector_store_manager
            .add_documents(
                vector_store,
                chunks
            )
        )

        # ----------------------------------------------------
        # 5. SAUVEGARDER FAISS
        # ----------------------------------------------------

        self.vector_store_manager.save_vector_store(
            vector_store,
            str(self.database_path)
        )

        return {
            "filename": pdf_path.name,
            "pages": len(documents),
            "chunks": len(chunks)
        }

    # ========================================================
    # LISTE DES DOCUMENTS
    # ========================================================

    def list_documents(self):

        documents = sorted(
            self.documents_path.glob("*.pdf")
        )

        return documents