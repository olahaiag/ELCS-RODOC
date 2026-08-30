from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


class DocumentSplitter:
    """
    Découpe les documents en petits morceaux
    appelés chunks.
    """

    def __init__(
        self,
        chunk_size=600,
        chunk_overlap=80
    ):

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    " ",
                    ""
                ]
            )
        )

    def split_documents(self, documents):

        if not documents:
            raise ValueError(
                "Aucun document à découper."
            )

        chunks = self.text_splitter.split_documents(
            documents
        )

        print(
            f"Nombre de chunks générés : "
            f"{len(chunks)}"
        )
        if chunks:
            lengths = [len(chunk.page_content) for chunk in chunks]

            print(f"Taille minimale : {min(lengths)}")
            print(f"Taille maximale : {max(lengths)}")
            print(f"Taille moyenne : {sum(lengths) / len(lengths):.0f}")


        return chunks