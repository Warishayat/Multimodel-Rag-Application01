import fitz  
import pdfplumber
import easyocr
from PIL import Image
import io
import numpy as np
import warnings


warnings.filterwarnings("ignore")
def ExtractDatafrompdf(pdf_path: str):
    try:
        doc = fitz.open(pdf_path)
        # Collect data in a structured dictionary
        result = {
            "metadata": doc.metadata,
            "pages": [],
            "tables": [],
            "ocr_images": []
        }
        # Loop through each page
        for page_num, page in enumerate(doc, start=1):
            page_data = {
                "page_number": page_num,
                "text": page.get_text()
            }

            result["pages"].append(page_data)

            # OCR for images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # OCR
                image = Image.open(io.BytesIO(image_bytes))
                # Convert PIL image to numpy array
                image_np = np.array(image)
                # Initialize EasyOCR Reader
                reader = easyocr.Reader(['en'])  
                # Perform OCR
                ocr_result = reader.readtext(image_np)
                # Extract text from OCR result
                ocr_text = "\n".join([text[1] for text in ocr_result])
                # Append OCR result to the data
                result["ocr_images"].append({
                    "page_number": page_num,
                    "image_index": img_index,
                    "ocr_text": ocr_text
                })

        # Extract tables
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table_num, table in enumerate(tables, start=1):
                    result["tables"].append({
                        "page_number": i,
                        "table_number": table_num,
                        "rows": table
                    })

        return result

    except FileNotFoundError:
        return {"error": f"The file {pdf_path} was not found."}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    pdf_path = r"C:\Users\HP\Desktop\MultiModel-Rag\Multimodel-Rag-Application01\Deepseek.pdf"
    data = ExtractDatafrompdf(pdf_path)

    if "error" in data:
        print("Error:", data["error"])
    else:
        print("Metadata:", data["metadata"])
        print("Page 1 Text:", data["pages"][0]["text"])
        print("First OCR image Text:", data["ocr_images"][0]["ocr_text"] if data["ocr_images"] else "No OCR data.")
        print("First Table:", data["tables"][0]["rows"] if data["tables"] else "No tables.")
