from rag.loader import Document_loader
from rag.splitter import Document_splitter 
from pathlib import Path
from rag.embeddings import Embeddings_generator
from rag.vector_store import Vector_store
from rag.retreiver import Retriever 
loader = Document_loader("documents")
documents = loader.load_documents()
print(f"Nombre de documents charges :{len(documents)}")
#for doc in documents : 
   # print("=" * 50)
   # print("Contenu : ")
   # print(doc.page_content[:300])
    #print()
   # print("Metadonnees : ")
   # print(doc.metadata)
    #source = Path(doc.metadata["source"])
    #doc.metadata["nomfichier"] = source.name
    #doc.metadata["extension"] = source.suffix
splitter = Document_splitter(chunk_size=800,chunk_overlap=200)
chunks = splitter.split_documents(documents)
embedding_model=Embeddings_generator()
vector_store=Vector_store(embedding_model.get_embeddings_model())
db=vector_store.create_vector_store(chunks)
vector_store.save_vector(db)
db= vector_store.load_vector()
retriever= Retriever(db)
print("Base FAISS cree avec succes")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 50)
    print(f"chunk { i +1}")
    print(chunk.page_content)
    print(chunk.metadata)

questions = [

    "Comment installer le logiciel ?",

    "Comment créer un utilisateur ?",

    "Comment sauvegarder les données ?",

    "Où sont stockées les données ?"

]
for question in questions:

    print("=" * 80)

    print("Question :")

    print(question)

    print()

    results = retriever.retrieve(question, k=3)

    for i, doc in enumerate(results):

        print(f"Résultat {i+1}")

        print(doc.page_content[:250])

        print(doc.metadata)

        print()