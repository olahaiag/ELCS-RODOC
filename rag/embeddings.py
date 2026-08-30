from langchain_huggingface import (
    HuggingFaceEmbeddings
)


class EmbeddingsGenerator:


    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.model_name = model_name
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def get_embeddings_model(self):

        return self.embeddings