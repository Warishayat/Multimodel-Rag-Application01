from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from pdfparsing import ExtractDatafrompdf
from Datapreprocessing import PreprocessingData
from vectorstore import embeddings, vectorstore
from langchain.chains import RetrievalQA

# Load environment
load_dotenv()
Groq_api_key = os.environ.get("GROQ_API_KEY")

# LLM setup
Model = ChatGroq(
    api_key=Groq_api_key,
    model="qwen-qwq-32b",  
    temperature=0.2,
)

def GenrateResponse(query, retrive):
    chain = RetrievalQA.from_chain_type(
        llm=Model,
        chain_type="stuff",
        retriever=retrive,
    )
    return chain.invoke({"query": query})

if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Desktop\MultiModel-Rag\Multimodel-Rag-Application01\Deepseek.pdf"

    print("Extracting PDF...")
    documents = ExtractDatafrompdf(pdf_path)

    print("Chunking Data...")
    chunked_data = PreprocessingData(documents)
    print(f"Total Chunks: {len(chunked_data)}")

    print("Vectorizing...")
    retriever = vectorstore(data=chunked_data, embeddings=embeddings())

    # Example query
    query = "what are the benchamrk of deepseek r1?"
    print("Answering Query...")
    result = GenrateResponse(query=query, retrive=retriever)

    print("Response:")
    print(result["result"])
