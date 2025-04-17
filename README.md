## 📄 Multi-Modal RAG PDF Chatbot

A Streamlit application that allows you to **upload a PDF**, ask questions about its content, and get accurate responses using a **Multi-Modal Retrieval-Augmented Generation (RAG)** pipeline powered by **Groq's Gemma-2 9B model**.
you can try it out this application here: https://huggingface.co/spaces/Waris01/Multi-Model-Pdf-Chat

---

### 🚀 Features

- 📁 Upload any PDF
- 🔍 Intelligent chunking and embedding
- 🧠 Ask natural language questions about your PDF
- ⚡ Powered by FAISS + HuggingFace + Groq LLM
- 🧠 Caches session so PDF isn't reprocessed on every query

---

### 🛠️ Installation (with `venv`)

1. **Clone the repo:**

```bash
git clone https://github.com/Warishayat/Multimodel-Rag-Application01.git
cd Multimodal-Rag-Application01
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
# Activate:
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up your `.env` file:**

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

### 📦 Project Structure

```
📁 Multimodal-Rag-Application01
├── main.py                 # Streamlit frontend
├── pdfparsing.py          # PDF parser using pymupdf4llm
├── Datapreprocessing.py   # Chunking & text cleaning
├── vectorstore.py         # Embedding & FAISS logic
├── .env                   # API keys
├── requirements.txt       # Python dependencies
└── README.md              # You're here!
```

---

### ▶️ Run the App

```bash
streamlit run main.py
```

Then open `http://localhost:8501` in your browser.

---

### 🧪 Example Queries

After uploading a PDF, try asking:
- "What is the summary of section 3?"
- "List all benchmarks mentioned."
- "How is this model different from others?"

---

### 💡 Tips

- PDF is processed only once per session using `st.session_state`.
- Uses `RecursiveCharacterTextSplitter` for effective chunking.
- Embedding with `HuggingFaceEmbeddings`.

---

### 📋 Requirements

Make sure your `requirements.txt` includes at least:

```txt
streamlit
python-dotenv
langchain
langchain-community
langchain-groq
faiss-cpu
pymupdf4llm
```

---

### 📬 Credits

Built with ❤️ by Waris Hayat Abbasi.

---
