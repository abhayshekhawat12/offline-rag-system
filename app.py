import streamlit as st
import os
import fitz  # PyMuPDF
import base64
import time
from io import BytesIO
from PIL import Image

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

# Setup paths and configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize Streamlit Page
st.set_page_config(page_title="Offline RAG & Multi-Modal Chat", page_icon="🧠", layout="wide")

# Modern Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stSidebar {
        background-color: #1E1E2E;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 Private Local Offline RAG")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_vectorstore():
    embeddings = get_embeddings()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

try:
    vectorstore = get_vectorstore()
except Exception as e:
    st.error(f"Error loading ChromaDB: {e}")
    st.stop()

# Initialize session state
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Profile & Knowledge Base
with st.sidebar:
    st.header("👤 User Profile")
    st.markdown("![Profile](https://api.dicebear.com/7.x/bottts/svg?seed=Admin&width=100)")
    st.write("**Name:** Admin User")
    st.write("**Bio:** Local AI Enthusiast")
    
    st.divider()
    
    st.header("📂 Knowledge Base")
    uploaded_files = os.listdir(UPLOADS_DIR)
    st.write(f"**Total Uploads:** {len(uploaded_files)}")
    
    for f in uploaded_files:
        col1, col2 = st.columns([0.8, 0.2])
        col1.markdown(f"📄 `{f}`")
        if col2.button("❌", key=f"del_{f}"):
            os.remove(os.path.join(UPLOADS_DIR, f))
            # Just UI removal. Deleting from DB needs lookup by metadata ID.
            if f in st.session_state.ingested_files:
                st.session_state.ingested_files.remove(f)
            st.rerun()
            
    st.divider()
    st.header("⚙️ Settings")
    st.write("Models in Use:")
    st.code("- llama3 (Text)\n- llava (Vision)\n- all-MiniLM-L6-v2 (Embeddings)")

def process_image_with_llava(image_bytes):
    try:
        # Reduce temperature and set a timeout if possible, though ChatOllama doesn't have a direct timeout easily accessible here.
        llm = ChatOllama(model="llava", temperature=0)
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail. Extract any text visible in the image and explain the contents. Provide comprehensive visual details."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
            ]
        )
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        return f"[Error processing image: {str(e)} Please ensure 'llava' model is pulled in Ollama]"

st.header("📤 Upload Documents & Images")
uploaded_file = st.file_uploader("Upload PDF, TXT, PNG, or JPG", type=["pdf", "txt", "png", "jpg", "jpeg"])

if uploaded_file and uploaded_file.name not in st.session_state.ingested_files:
    file_path = os.path.join(UPLOADS_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    with st.spinner(f"Ingesting {uploaded_file.name}..."):
        progress_bar = st.progress(10)
        docs = []
        
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            progress_bar.progress(40)
        elif ext == 'pdf':
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            # Check if there's substantial text in the PDF
            total_text_length = sum(len(d.page_content.strip()) for d in docs)
            if not docs or total_text_length < 50:
                st.toast("⚠️ Low text detected. Assuming scanned PDF... using Vision Model (LLaVA)...")
                pdf_doc = fitz.open(file_path)
                image_docs = []
                # limit to first 3 pages to prevent massive hangs
                num_pages = min(len(pdf_doc), 3)
                for page_num in range(num_pages):
                    page = pdf_doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    st.info(f"🖼️ Reading Scanned PDF Page {page_num + 1}/{num_pages} with LLaVA...")
                    image_description = process_image_with_llava(img_bytes)
                    image_docs.append(Document(page_content=f"[SCANNED PDF PAGE {page_num + 1}]\nFile: {uploaded_file.name}\nDescription & Text:\n{image_description}", metadata={"source": file_path, "page": page_num}))
                docs = image_docs
                if len(pdf_doc) > 3:
                     st.warning("⚠️ Only the first 3 scanned pages were processed to save time.")
            progress_bar.progress(40)
        elif ext in ['png', 'jpg', 'jpeg']:
            progress_bar.progress(20)
            st.info("🖼️ Extracting context from Image using LLaVA (Offline Vision Model)...")
            image_description = process_image_with_llava(uploaded_file.getvalue())
            docs = [Document(page_content=f"[IMAGE UPLOAD] File name: {uploaded_file.name}\nDescription & Extracted Text:\n{image_description}", metadata={"source": file_path})]
            progress_bar.progress(60)
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)
        progress_bar.progress(80)
        
        # Add to Chroma
        if splits:
            st.info(f"🧠 Generating embeddings for {len(splits)} chunks... This might take a bit.")
            batch_size = 20
            total_splits = len(splits)
            for i in range(0, total_splits, batch_size):
                batch = splits[i:i+batch_size]
                vectorstore.add_documents(batch)
                progress = 80 + int(((i + len(batch)) / total_splits) * 19)
                progress_bar.progress(min(progress, 99))
            
        progress_bar.progress(100)
        time.sleep(1)
        progress_bar.empty()
        
    st.session_state.ingested_files.add(uploaded_file.name)
    st.success(f"✅ Automatically ingested `{uploaded_file.name}` into local vector store!")
    st.rerun()

st.divider()

# Question Answering Chat Interface & Generate Study Guide
col_chat, col_qa = st.columns([0.65, 0.35])

with col_chat:
    st.header("💬 Chat with your Data")

    # Display chat history (only Q&A related to chat, not study guide)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Retrieving answers..."):
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                relevant_docs = retriever.invoke(prompt)
                
                if relevant_docs:
                    context = "\n\n".join([d.page_content for d in relevant_docs])
                    
                    llm = ChatOllama(model="llama3", temperature=0.1)
                    system_prompt = f"""You are an intelligent and helpful AI assistant.
Answer the user's question deeply and accurately using the context below. Focus mainly on extracting insights and detailed answers from this context.
If no relevant information is present, let the user know gently, but try to use logic to answer if partially available.

Context Information:
{context}"""
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=prompt)
                    ]
                    
                    def stream_response():
                        for chunk in llm.stream(messages):
                            yield chunk.content
                            
                    reply = st.write_stream(stream_response())
                else:
                    reply = "I don't have any uploaded documents to reference. Please upload a document first."
                    st.markdown(reply)
                
        st.session_state.messages.append({"role": "assistant", "content": reply})

with col_qa:
    st.header("📝 Auto Q&A Generator")
    st.info("Click below to scan all the knowledge base context and produce a structured list of potential Questions and Answers.")
    if st.button("🚀 Generate Study Guide", use_container_width=True):
        with st.spinner("Analyzing context & Generating Study Guide..."):
            try:
                # To prevent loading an extremely large amount of text, we can get top chunks or recent chunks
                # Here we just get elements via similarity to a generic query to get a robust view, or just get from Chroma directly.
                all_docs_dict = vectorstore.get()
                docs_content = all_docs_dict.get('documents', [])
                
                if not docs_content:
                    st.warning("⚠️ No documents found. Please upload some files first!")
                else:
                    # Limit context size even more (approx ~500 words) to make the model read and start generating instantly
                    combined_text = "\n".join(docs_content)[:2500] 
                    
                    # Temperature to 0.1 for maximum determinism and speed
                    llm = ChatOllama(model="llama3", temperature=0.1)
                    guided_prompt = f"""You are an ultra-fast study guide generator.
Based strictly on the text below, create exactly 2 Very Short, 2 Short, and 2 Long questions and answers. Provide NO extra or conversational text. Output ONLY the Q&A.

Text:
{combined_text}

Format:
### ⚡ Very Short Q&A
**Q1:** [Question]
*A1:* [1 sentence answer]
(Q2 similarly)

### 📝 Short Q&A
**Q3:** [Question]
*A3:* [2 sentences answer]
(Q4 similarly)

### 📖 Long Detailed Q&A
**Q5:** [Question]
*A5:* [Paragraph answer]
(Q6 similarly)
"""
                    msg = HumanMessage(content=guided_prompt)
                    st.markdown("### 📚 Generated Study Guide")
                    
                    def stream_guide():
                        for chunk in llm.stream([msg]):
                            yield chunk.content
                    
                    st.write_stream(stream_guide())
            except Exception as e:
                st.error(f"Error generating study guide: {e}")

st.divider()

# --- Footer Section ---
st.markdown("""
<br><br>
<div style='text-align: center; padding: 20px; background-color: #1E1E2E; border-radius: 10px; border: 1px solid #333;'>
    <h3 style='color: #4CAF50;'>🏆 About This Project</h3>
    <p style='font-size: 16px; margin-bottom: 5px;'><b>Leader Name:</b> Abhay Shekhawat &nbsp;|&nbsp; <b>Email:</b> abhayshekhawat57@gmail.com</p>
    <p style='font-size: 16px; margin-bottom: 20px; color: #b0b0b0;'><b>College:</b> Arya College of Engineering and IT</p>
    
    <hr style='border-top: 1px solid #444; margin: 15px 0;'>
    
    <div style='text-align: left; font-size: 15px; color: #FAFAFA; line-height: 1.8; display: inline-block; text-align: left;'>
        <b>⚙️ How to Use:</b><br>
        👉 <b>Step 1:</b> Upload your PDF, Text file, or Image.<br>
        👉 <b>Step 2:</b> Ask a question in the chat or click "Generate Study Guide".<br>
        👉 <b>Step 3:</b> Get your answers instantly!
    </div>
</div>
""", unsafe_allow_html=True)

# Vercel entrypoint compatibility
def handler(request=None):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "text/html"},
        "body": "<h1>🧠 Private Local Offline RAG</h1><p>Please run locally using Streamlit: <code>streamlit run app.py</code></p>"
    }

app = handler
