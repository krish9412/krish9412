import streamlit as st
import pdfplumber
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import os

# Initialize session state
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'course_content' not in st.session_state:
    st.session_state.course_content = None

# Streamlit app layout
st.title("PDF Course Generator and Q&A")
st.sidebar.header("Configuration")

# API Key input
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# Model selection
model_options = ["gpt-4o-mini"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_options, index=0)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def process_pdf(file):
    """Extract text from a PDF file and create document embeddings."""
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # Create documents
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=openai_api_key
        )
        
        # Create vector store
        vectorstore = Chroma.from_documents(documents, embeddings)
        
        # Store retriever in session state
        st.session_state.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        return "PDF processed successfully. You can now generate a course or ask questions."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def generate_course_content():
    """Generate course content based on processed PDF."""
    try:
        if not openai_api_key:
            return "API key is required to generate course content."
        
        if not st.session_state.retriever:
            return "Please process a PDF first."
        
        # Use OpenAI to generate course structure
        llm = OpenAI(
            model=selected_model,
            temperature=0.7,
            api_key=openai_api_key
        )
        
        # Retrieve relevant documents
        docs = st.session_state.retriever.get_relevant_documents("Summarize the main topics")
        
        # Generate course structure
        course_prompt = f"""
        Based on the following document content, create a course structure with:
        - Course Title
        - Course Description
        - 3 Modules, each with:
          - Module Title
          - 2 Learning Objectives
          - Content Summary (100 words)
        
        Document Content:
        {docs[0].page_content[:1000]}
        
        Provide the response in JSON format.
        """
        
        course_content = llm(course_prompt)
        course_content = eval(course_content)  # Assuming LLM returns valid JSON string
        
        st.session_state.course_content = course_content
        return course_content
    except Exception as e:
        return f"Error generating course content: {str(e)}"

def generate_rag_answer(question, course_content=None):
    """Generate an answer using Retrieval-Augmented Generation."""
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        if not st.session_state.retriever:
            return "Document embeddings are not available. Please process documents first."
        
        # Include course content for additional context if available
        course_context = ""
        if course_content:
            course_context = f"""
            Course Title: {course_content.get('course_title', '')}
            Course Description: {course_content.get('course_description', '')}
            
            Module Information:
            """
            for i, module in enumerate(course_content.get('modules', []), 1):
                course_context += f"""
                Module {i}: {module.get('title', '')}
                Learning Objectives: {', '.join(module.get('learning_objectives', []))}
                Content Summary: {module.get('content', '')[:200]}...
                """
        
        # Custom prompt template for RAG
        template = """
        You are an AI assistant for a professional learning platform. Answer the following question
        based on the provided document content. Be specific, accurate, and helpful.
        
        Question: {question}
        
        {context}
        
        Additional Course Information: {course_context}
        
        Provide a comprehensive answer using information from the documents and course contents.
        If the question cannot be answered based on the provided information, say so politely.
        Reference specific documents when appropriate in your answer.
        """
        
        prompt = PromptTemplate(
            input_variables=["question", "context", "course_context"],
            template=template
        )
        
        # Create the QA chain
        llm = OpenAI(
            model=selected_model,
            temperature=0.5,
            api_key=openai_api_key
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.retriever,
            chain_type_kwargs={
                "prompt": prompt,
            }
        )
        
        # Run the query
        result = qa_chain.run(
            question=question,
            course_context=course_context
        )
        
        return result
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Main app logic
if uploaded_file:
    with st.spinner("Processing PDF..."):
        result = process_pdf(uploaded_file)
        st.write(result)

if st.button("Generate Course"):
    with st.spinner("Generating course content..."):
        course_content = generate_course_content()
        if isinstance(course_content, dict):
            st.subheader("Generated Course Content")
            st.json(course_content)
        else:
            st.error(course_content)

# Question input
question = st.text_input("Ask a question about the PDF or course content")
if question:
    with st.spinner("Generating answer..."):
        answer = generate_rag_answer(question, st.session_state.course_content)
        st.write("**Answer:**")
        st.write(answer)
