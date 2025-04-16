from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from Datapreprocessing import PreprocessingData
from pdfparsing import ExtractDatafrompdf
import warnings
warnings.filterwarnings("ignore")

def embeddings():
    print("Embeddings  are getting genrated,appriciation for you wait")
    embeddings = HuggingFaceEmbeddings()
    return embeddings
    
def vectorstore(data,embeddings):
    print("Vector store is created")
    vectorstore = FAISS.from_documents(
        documents=data,
        embedding = embeddings
    )
    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k":1})
    return retriever

if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Desktop\MultiModel-Rag\Multimodel-Rag-Application01\Deepseek.pdf"
    pdf_data = ExtractDatafrompdf(pdf_path)
    print("Successfully data is extracted")
    chunked_data = PreprocessingData(data=pdf_data)
    print("Data preprocessing in done")
    print(f"the length of the chunks are:{len(chunked_data)}")
    retriver=vectorstore(embeddings=embeddings(),data=chunked_data)
    print("VectoStore is created Successfully.")
    print(retriver.invoke("what is Summary of Evaluation Results?"))