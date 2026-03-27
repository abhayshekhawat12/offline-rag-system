# 🚀 Private Local Offline RAG System

A completely offline, local Retrieval-Augmented Generation (RAG) dashboard that supports **PDFs**, **TXT** files, and even **Images (PNG, JPG)**. Built entirely on your own local environment—no data leaves your machine. 

## 🛠 Features
- **Offline First**: Uses local Llama-3 & LLaVA (Ollama) and ChromaDB.
- **Multimodal**: Seamlessly read text files or describe images.
- **Auto Data Ingestion**: Background indexing of documents & images as soon as you upload.
- **Auto Q&A Generator**: 1-Click generation of a Study Guide from your uploaded context.
- **Clean Dashboard**: Dark UI, chat interface, profile sidebars and simple document management.

## ⚙️ Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/)** Installed on your system

### Ollama Setup
Start your local models by downloading them into your Ollama engine. Open your terminal and run:
```bash
ollama pull llama3
ollama pull llava
```

## 📦 Installation & Setup

1. **Create and activate a virtual environment (optional but recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

2. **Install the dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit Dashboard:**
```bash
streamlit run app.py
```

## 🖥 Local Hardware Requirements
- **RAM**: Minimum 16GB (Recommended)
- **GPU**: NVIDIA with at least 8GB VRAM (Will run super-fast!). 
- **CPU Fallback**: Also runs on CPU, but processing images and text generation may take a bit longer.

*Built for absolute privacy and complete local persistence.*
