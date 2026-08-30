from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


class DocumentLoader:


    def __init__(self, folder_path: str):

        self.folder = Path(folder_path)

    def load_documents(self):

        list_documents = []

        if not self.folder.exists():
            raise FileNotFoundError(
                f"Le dossier '{self.folder}' n'existe pas."
            )

        for file in self.folder.rglob("*"):

            if not file.is_file():
                continue

            try:

                extension = file.suffix.lower()

                if extension == ".pdf":

                    loader = PyPDFLoader(
                        str(file)
                    )

                elif extension == ".txt":

                    loader = TextLoader(
                        str(file),
                        encoding="utf-8"
                    )

                elif extension == ".docx":

                    loader = Docx2txtLoader(
                        str(file)
                    )

                else:
                    continue



                docs = loader.load()

                for doc in docs:

                    doc.metadata["filename"] = file.name
                    doc.metadata["extension"] = extension



                    if extension == ".pdf":
                        if "page" in doc.metadata:
                            doc.metadata["page"] = (
                                doc.metadata["page"] + 1
                            )

                list_documents.extend(docs)

                print(
                    f"✓ {file.name} "
                    f"({len(docs)} pages/documents)"
                )

            except Exception as e:

                print(
                    f"✗ Erreur lors du chargement "
                    f"de {file.name}"
                )

                print(e)

        return list_documents