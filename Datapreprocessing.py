from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdfparsing import ExtractDatafrompdf
from langchain.docstore.document import Document
import warnings

warnings.filterwarnings("ignore")

def PreprocessingData(data,chunk_size=1000,chunk_overlap=40):
    """
    Preprocess the text by removing unnecessary characters and extra spaces.
    """
    try:
        documents = [Document(page_content=page["text"], metadata={"page_number": page["page_number"]}) for page in data["pages"]]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunked_text = text_splitter.split_documents(documents)

        return chunked_text
    except Exception as e:
        print(f"you may have some error {e}")


if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Desktop\MultiModel-Rag\Multimodel-Rag-Application01\Deepseek.pdf"
    pdf_data = ExtractDatafrompdf(pdf_path)

    if pdf_data is not None:
        chunked_data = PreprocessingData(data=pdf_data)
        print(len(chunked_data))
    else: 
        print("Pdf parsing may face some issue kindly check the file of pdfparsing.py")
