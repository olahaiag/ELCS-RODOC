from langchain_community.vectorstores import FAISS


class VectorStore:
    """
    Gère la création, la sauvegarde et
    le chargement de la base vectorielle FAISS.
    """

    def __init__(self, embeddings_model):

        self.embedding_model = embeddings_model

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
        print("Index FAISS utilise : ", type(vector_store.index))
        return vector_store

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