import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import google.generativeai as genai

# Step 1: Load PDF
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

# Step 2: Split into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

# Step 3: Generate embeddings
def generate_embeddings(chunks, embedding_model, batch_size=16):
    texts = [doc.page_content for doc in chunks]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings.extend(embedding_model.encode(batch))
    return np.array(embeddings), texts

# Step 4: Create FAISS index
def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Step 5: Retrieve relevant chunks
def retrieve_documents(query, index, embedding_model, texts, k=10):
    query_vec = embedding_model.encode([query])
    _, indices = index.search(np.array(query_vec), k)
    return [texts[i] for i in indices[0]]

# Step 6: Load SentenceTransformer
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Step 7: Set up Gemini model
@st.cache_resource
def load_gemini_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")

def generate_answer_with_gemini(query, context, gemini_model):
    prompt = f"""You are a helpful assistant.
Using the context below, answer the user's question in detail.

Context:
{context}

Question:
{query}

Answer:"""
    response = gemini_model.generate_content(prompt)
    return response.text

# Streamlit UI
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")
st.title("📘 PDF Q&A System using Gemini Flash ✨")

api_key = "AIzaSyCzNgqTXs5wMeoa6s53XhhaFf4H_relLOs"
uploaded_file = st.file_uploader("📎 Upload a PDF document", type="pdf")

if api_key and uploaded_file:
    os.makedirs("temp_pdf", exist_ok=True)
    temp_path = os.path.join("temp_pdf", uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Processing document..."):
        docs = load_pdf(temp_path)
        chunks = split_documents(docs)
        embedding_model = load_embedding_model()
        embeddings, texts = generate_embeddings(chunks, embedding_model)
        index = create_faiss_index(embeddings)
        gemini = load_gemini_model(api_key)
        os.remove(temp_path)

    query = st.text_input("💬 Ask a question from the document")

    if query:
        with st.spinner("🤖 Generating answer..."):
            relevant_chunks = retrieve_documents(query, index, embedding_model, texts)
            context = " ".join(relevant_chunks)
            answer = generate_answer_with_gemini(query, context, gemini)

        st.subheader("📌 Answer")
        st.write(answer)

        with st.expander("📚 Show Retrieved Chunks"):
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {i+1}:** {chunk}")

