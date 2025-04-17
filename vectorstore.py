from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def embeddings():
    print("Generating Embeddings...")
    return HuggingFaceEmbeddings()

def vectorstore(data, embeddings):
    print("Creating VectorStore...")
    vectorstore = FAISS.from_documents(
        documents=data,
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}  # You can tune this
    )
    return retriever
