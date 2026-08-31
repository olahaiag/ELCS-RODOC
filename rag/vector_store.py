from langchain_community.vectorstores import FAISS


class VectorStore:
    """
    Gère la création, l'ajout, la sauvegarde et
    le chargement de la base vectorielle FAISS.
    """

    def __init__(self, embeddings_model):

        self.embedding_model = embeddings_model

    # ========================================================
    # CRÉER UNE NOUVELLE BASE
    # ========================================================

    def create_vector_store(self, chunks):

        if not chunks:
            raise ValueError(
                "Aucun chunk fourni pour créer "
                "le vector store."
            )

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_model
        )

        print(
            "Index FAISS utilisé : ",
            type(vector_store.index)
        )

        return vector_store

    # ========================================================
    # AJOUTER DES CHUNKS À UNE BASE EXISTANTE
    # ========================================================

    def add_documents(
        self,
        vector_store,
        chunks
    ):

        if not chunks:
            raise ValueError(
                "Aucun chunk fourni pour "
                "ajouter à la base."
            )

        vector_store.add_documents(
            documents=chunks
        )

        print(
            f"{len(chunks)} chunks ajoutés "
            "à la base FAISS."
        )

        return vector_store

    # ========================================================
    # SAUVEGARDER
    # ========================================================

    def save_vector_store(
        self,
        vector_store,
        path="database"
    ):

        vector_store.save_local(path)

        print(
            f"Base FAISS sauvegardée dans : "
            f"{path}"
        )

    # ========================================================
    # CHARGER
    # ========================================================

    def load_vector_store(
        self,
        path="database"
    ):

        vector_store = FAISS.load_local(
            path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        return vector_store