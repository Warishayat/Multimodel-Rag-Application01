import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from Datapreprocessing import PreprocessingData
from pdfparsing import ExtractDatafrompdf

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="📄 Chat with PDF", layout="wide")

# Sidebar for PDF Upload
st.sidebar.title("📂 Upload your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF", type="pdf")

# LLM and Embeddings - cached
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings()

@st.cache_resource
def get_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model="gemma2-9b-it", temperature=0.2)

# Build Retrieval Chain
def get_chain(retriever):
    llm = get_llm()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# PDF processing pipeline
def process_pdf_and_create_chain(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    documents = ExtractDatafrompdf(tmp_path)
    chunks = PreprocessingData(documents)
    embedder = get_embeddings()
    retriever = FAISS.from_documents(chunks, embedder).as_retriever(search_type="similarity", search_kwargs={"k": 1})
    return get_chain(retriever)

# Main UI
st.title("📄 Ask Questions About Your PDF")

if uploaded_file:
    if "chain" not in st.session_state:
        st.success("PDF uploaded successfully! Processing...")
        with st.spinner("Extracting and chunking PDF..."):
            st.session_state.chain = process_pdf_and_create_chain(uploaded_file)
        st.success("Ready to chat with your PDF!")
    else:
        st.sidebar.info("Using cached PDF session.")

    user_query = st.text_input("Ask a question about your PDF:")
    if user_query:
        with st.spinner("Generating answer..."):
            result = st.session_state.chain.invoke({"query": user_query})
            st.markdown("### 📌 Answer:")
            st.write(result["result"])
else:
    st.info("📤 Upload a PDF from the sidebar to begin.")
