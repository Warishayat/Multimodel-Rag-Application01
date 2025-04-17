import pymupdf4llm
from langchain.docstore.document import Document

def ExtractDatafrompdf(pdf_path):
    """Extract PDF data using pymupdf4llm and return LangChain Documents."""
    md_pages = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=True,
        image_path="images",
        image_format="png",
        page_chunks=True
    )

    
    print("First page structure:", md_pages[0])

    documents = []
    for page in md_pages:
        text = page["text"]  
        page_num = md_pages.index(page) + 1  
        documents.append(Document(page_content=text, metadata={"page_number": page_num}))
    
    return documents
