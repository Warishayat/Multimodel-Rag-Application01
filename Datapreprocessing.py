from langchain.text_splitter import RecursiveCharacterTextSplitter

def PreprocessingData(documents, chunk_size=1500, chunk_overlap=40):
    """Chunk documents into smaller parts for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs
