import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
import chromadb
import numpy as np
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# Page Configuration
st.set_page_config(page_title="📚 Professional Learning Platform", layout="wide")

# Initialize ChromaDB client
chroma_client = chromadb.Client()
collection_name = "document_embeddings"

# Initialize session state variables
if 'course_content' not in st.session_state:
    st.session_state.course_content = None
if 'course_generated' not in st.session_state:
    st.session_state.course_generated = False
if 'is_generating' not in st.session_state:
    st.session_state.is_generating = False
if 'completed_questions' not in st.session_state:
    st.session_state.completed_questions = set()
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'extracted_texts' not in st.session_state:
    st.session_state.extracted_texts = []
if 'employer_queries' not in st.session_state:
    st.session_state.employer_queries = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []
if 'chroma_collection' not in st.session_state:
    st.session_state.chroma_collection = None

# Sidebar Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Session Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.session_state.chroma_collection = None
    chroma_client.delete_collection(collection_name)
    st.rerun()

# 🔐 OpenAI API Key Input
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")

# 📄 Multi-File Uploader for PDFs
uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to extract text from PDF
def extract_pdf_text(pdf_file):
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

# Function to chunk text
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks

# Function to generate embeddings
def generate_embedding(text, client):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Process uploaded files and store in ChromaDB
if uploaded_files and openai_api_key:
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names:
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames
        st.session_state.chroma_collection = None

        # Create or reset ChromaDB collection
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass
        collection = chroma_client.create_collection(name=collection_name)
        st.session_state.chroma_collection = collection

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Extract text and store embeddings
        with st.spinner("Processing PDF files and generating embeddings..."):
            for pdf_file in uploaded_files:
                extracted_text = extract_pdf_text(pdf_file)
                if extracted_text:
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text
                    })
                    st.session_state.uploaded_files.append(pdf_file)

                    # Chunk the text and generate embeddings
                    chunks = chunk_text(extracted_text)
                    embeddings = []
                    ids = []
                    metadatas = []
                    for i, chunk in enumerate(chunks):
                        embedding = generate_embedding(chunk, client)
                        if embedding:
                            embeddings.append(embedding)
                            ids.append(f"{pdf_file.name}_{i}")
                            metadatas.append({"filename": pdf_file.name, "chunk_index": i})

                    # Store in ChromaDB
                    if embeddings:
                        collection.add(
                            embeddings=embeddings,
                            documents=chunks,
                            metadatas=metadatas,
                            ids=ids
                        )

        if st.session_state.extracted_texts:
            st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed successfully!")
else:
    st.info("📥 Please enter your OpenAI API key and upload PDF files to begin.")

# 🎯 GPT Model and Role selection
model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_options, index=0)

role_options = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
role = st.sidebar.selectbox("Select Your Role", role_options)

learning_focus_options = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
learning