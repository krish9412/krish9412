import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
import time
from openai import OpenAI
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# LangChain & ChromaDB imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Page Configuration
st.set_page_config(page_title="📚 Enhanced Professional Learning Platform", layout="wide")

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
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()

# Sidebar Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    # Clean up temp directory contents
    if 'temp_dir' in st.session_state and os.path.exists(st.session_state.temp_dir):
        for filename in os.listdir(st.session_state.temp_dir):
            file_path = os.path.join(st.session_state.temp_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                st.sidebar.error(f"Error deleting {file_path}: {e}")
    
    # Reset all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize essential variables
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.session_state.temp_dir = tempfile.mkdtemp()
    st.session_state.chat_history = []
    
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


# Function to create embeddings and setup vector store
def setup_vector_store(documents):
    try:
        if not openai_api_key:
            st.warning("OpenAI API key is required for vector embeddings.")
            return None
        
        # Create a text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Save text chunks to temporary files
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = text_splitter.split_text(doc["text"])
            
            # Create document metadata for each chunk
            for j, chunk in enumerate(chunks):
                # Save chunk to temporary file
                chunk_file_path = os.path.join(st.session_state.temp_dir, f"doc_{i}_chunk_{j}.txt")
                with open(chunk_file_path, "w", encoding="utf-8") as f:
                    f.write(chunk)
                
                # Create metadata for the chunk
                metadata = {
                    "source": doc["filename"],
                    "chunk_id": j,
                    "document_id": i
                }
                
                # Add to collection of chunks with metadata
                all_chunks.append({"path": chunk_file_path, "metadata": metadata})
        
        # Load documents with metadata
        loaded_docs = []
        for chunk_info in all_chunks:
            loader = TextLoader(chunk_info["path"], encoding="utf-8")
            doc = loader.load()[0]
            doc.metadata = chunk_info["metadata"]
            loaded_docs.append(doc)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=loaded_docs,
            embedding=embeddings,
            persist_directory=os.path.join(st.session_state.temp_dir, "chroma_db")
        )
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error setting up vector store: {str(e)}")
        return None


# Process uploaded files and add to session state
if uploaded_files and openai_api_key:
    # Clear previous uploads if list has changed
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names:
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames
        
        # Extract text from each PDF and store in session state
        with st.spinner("Processing PDF files and creating embeddings..."):
            for pdf_file in uploaded_files:
                extracted_text = extract_pdf_text(pdf_file)
                if extracted_text:
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text
                    })
                    st.session_state.uploaded_files.append(pdf_file)
            
            # Setup vector store with the extracted texts
            if st.session_state.extracted_texts:
                st.session_state.vector_store = setup_vector_store(st.session_state.extracted_texts)
                
                if st.session_state.vector_store:
                    # Initialize conversation memory
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )
                    
                    # Setup LangChain QA chain
                    llm = ChatOpenAI(
                        openai_api_key=openai_api_key,
                        model_name="gpt-4o-mini",  # Default model, can be changed by user
                        temperature=0.5
                    )
                    
                    # Create template for RAG
                    template = """You are a knowledgeable AI assistant for a professional learning platform.
                    Use the following pieces of context to answer the question at the end.
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    
                    Context: {context}
                    
                    Chat History: {chat_history}
                    
                    Question: {question}
                    
                    Answer:"""
                    
                    QA_CHAIN_PROMPT = PromptTemplate(
                        input_variables=["context", "question", "chat_history"],
                        template=template
                    )
                    
                    # Create the chain
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever(
                            search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
                        ),
                        return_source_documents=True,
                        chain_type_kwargs={
                            "prompt": QA_CHAIN_PROMPT,
                            "memory": memory
                        }
                    )
                
                st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed successfully!")
else:
    st.info("📥 Please enter your OpenAI API key and upload PDF files to begin.")

# 🎯 GPT Model and Role selection
model_options = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_options, index=0)

role_options = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
role = st.sidebar.selectbox("Select Your Role", role_options)

learning_focus_options = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
learning_focus = st.sidebar.multiselect("Select Learning Focus", learning_focus_options)

# Display uploaded files in sidebar
if st.session_state.uploaded_file_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Uploaded Files")
    for i, filename in enumerate(st.session_state.uploaded_file_names):
        st.sidebar.text(f"{i+1}. {filename}")


# Enhanced RAG function using vector embeddings
def generate_rag_answer(question, documents=None, course_content=None):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        # Update the QA chain model if user changed it
        if st.session_state.qa_chain and hasattr(st.session_state.qa_chain, 'llm'):
            st.session_state.qa_chain.llm = ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name=selected_model,
                temperature=0.5
            )
        
        # Course context to include in the query
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
        
        # Enhanced question with course context
        enhanced_question = f"{question}\n\nAdditional context: {course_context}" if course_context else question
        
        # Check if we have a vector store and QA chain setup
        if st.session_state.qa_chain and st.session_state.vector_store:
            # Run the query through the QA chain
            result = st.session_state.qa_chain({"query": enhanced_question})
            
            # Get the answer and source documents
            answer = result.get("result", "")
            source_docs = result.get("source_documents", [])
            
            # Format source information
            sources_info = ""
            seen_sources = set()
            if source_docs:
                sources_info = "\n\nInformation sourced from:\n"
                for doc in source_docs:
                    source = doc.metadata.get("source", "Unknown")
                    if source not in seen_sources:
                        seen_sources.add(source)
                        sources_info += f"- {source}\n"
            
            # Add sources to the answer
            answer_with_sources = f"{answer}{sources_info}"
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer_with_sources})
            
            return answer_with_sources
        
        else:
            # Fallback to direct OpenAI API if vector store is not available
            # Create a context from all document texts (with file attribution)
            combined_context = ""
            if documents:
                for i, doc in enumerate(documents[:3]):  # Limit to first 3 documents to avoid token issues
                    context_chunk = doc["text"][:2000]  # Limit each doc to 2000 chars
                    combined_context += f"\nDocument {i+1} ({doc['filename']}):\n{context_chunk}\n"
            
            prompt = f"""
            You are an AI assistant for a professional learning platform. Answer the following question 
            based on the provided document content. Be specific, accurate, and helpful.
            
            Question: {question}
            
            Document Content: {combined_context}
            
            Course Information: {course_context}
            
            Provide a comprehensive answer using information from the documents and course contents.
            If the question cannot be answered based on the provided information, say so politely.
            Reference specific documents when appropriate in your answer.
            """
            
            # Create OpenAI client correctly
            client = OpenAI(api_key=openai_api_key)
            response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            
            # Return generated answers
            answer = response.choices[0].message.content
            
            # Add to chat history
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            
            return answer
        
    except Exception as e:
        error_message = f"Error generating answer: {str(e)}"
        st.error(error_message)
        return error_message


# Employee Queries Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employer Queries")

new_query = st.sidebar.text_area("Add a new question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_query:
        # Generate proper answers automatically if documents are available
        answer = ""
        if st.session_state.extracted_texts:
            with st.spinner("Generating answer using vector embeddings..."):
                answer = generate_rag_answer(
                    new_query, 
                    st.session_state.extracted_texts,
                    st.session_state.course_content if st.session_state.course_generated else None
                )
        else:
            answer = "Please upload and process documents first to enable question answering."
        
        st.session_state.employer_queries.append({
            "question": new_query,
            "answer": answer,
            "answered": bool(answer),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

# Functions to check answer and update progress
def check_answer(question_id, user_answer, correct_answer):
    if user_answer == correct_answer:
        st.success("🎉 Correct! Well done!")
        # Add to completed questions set if not already there
        st.session_state.completed_questions.add(question_id)
        return True
    else:
        st.error(f"Not quite. The correct answer is: {correct_answer}")
        return False

# Generate Progress Report
def generate_progress_report():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Add report header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Professional Learning Progress Report")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, f"User Role: {role}")
    c.drawString(100, 710, f"Learning Focus: {', '.join(learning_focus)}")
    c.drawString(100, 690, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    # Calculate progress
    completed = len(st.session_state.completed_questions)
    total = st.session_state.total_questions
    progress_percentage = (completed / total * 100) if total > 0 else 0
    
    c.drawString(100, 670, f"Progress: {completed}/{total} questions completed ({progress_percentage:.1f}%)")
    
    # Add course details
    if st.session_state.course_generated and st.session_state.course_content:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, 630, "Course Information")
        
        c.setFont("Helvetica", 12)
        c.drawString(100, 610, f"Course Title: {st.session_state.course_content.get('course_title', 'N/A')}")
        
        # Add module completion status
        c.setFont("Helvetica-Bold", 14)
        c.drawString(100, 570, "Module Completion")
        
        c.setFont("Helvetica", 12)
        y_position = 550
        modules = st.session_state.course_content.get("modules", [])
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            
            # Count completed questions for this module
            module_questions = 0
            module_completed = 0
            quiz = module.get('quiz', {})
            questions = quiz.get('questions', [])
            
            for q_idx, _ in enumerate(questions, 1):
                question_id = f"module_{i}_question_{q_idx}"
                module_questions += 1
                if question_id in st.session_state.completed_questions:
                    module_completed += 1
            
            module_progress = (module_completed / module_questions * 100) if module_questions > 0 else 0
            c.drawString(100, y_position, f"Module {i}: {module_title} - {module_completed}/{module_questions} ({module_progress:.1f}%)")
            y_position -= 20
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Course Generation function
def generate_course():
    # Set generation flag to True when starting
    st.session_state.is_generating = True
    st.session_state.course_generated = False
    st.rerun()  # Trigger rerun to show loading state

# Function to actually generate the course content
def perform_course_generation():
    try:
        # Use vector store for more efficient document understanding
        documents_context = ""
        if st.session_state.vector_store:
            # Extract key concepts from the documents using the vector store
            key_concept_queries = [
                "What are the main topics covered in these documents?",
                "What are the key principles taught in these materials?",
                "What practical skills are developed through these documents?",
                "What are the theoretical foundations presented in these materials?",
                "What are the most important takeaways from these documents?"
            ]
            
            combined_insights = ""
            for query in key_concept_queries:
                with st.spinner(f"Analyzing documents - {query}"):
                    insight = generate_rag_answer(query, st.session_state.extracted_texts)
                    combined_insights += f"\n\n{insight}"
            
            documents_context = combined_insights
        else:
            # Fallback to standard extraction
            combined_docs = ""
            for i, doc in enumerate(st.session_state.extracted_texts):
                doc_summary = f"\n--- DOCUMENT {i+1}: {doc['filename']} ---\n"
                doc_summary += doc['text'][:3000]  # Limit each doc to avoid token limits
                combined_docs += doc_summary + "\n\n"
            
            documents_context = combined_docs
        
        professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
        
        # Get a document summary first
        with st.spinner("Creating comprehensive document summary..."):
            summary_query = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
            document_summary = generate_rag_answer(summary_query, st.session_state.extracted_texts)
        
        with st.spinner("Generating course outline and structure..."):
            # First, generate just the course outline
            outline_prompt = f"""
            Design an outline for a comprehensive professional learning course based on the documents provided.
            Context: {professional_context}
            Document Summary: {document_summary}
            
            Create a thoughtful course outline with:
            1. A compelling course title
            2. A brief course description
            3. 5-8 module titles with 3-4 learning objectives each
            
            Return the response in the following JSON format:
            {{
                "course_title": "Your Course Title",
                "course_description": "Brief description of the course",
                "modules": [
                    {{
                        "title": "Module 1 Title",
                        "learning_objectives": ["Objective 1", "Objective 2", "Objective 3"]
                    }}
                ]
            }}
            """
            
            # Create OpenAI client for the outline
            client = OpenAI(api_key=openai_api_key)
            outline_response = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": outline_prompt}],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Parse the outline
            outline_content = json.loads(outline_response.choices[0].message.content)
        
        # Now generate detailed content for each module one by one
        complete_course = {
            "course_title": outline_content.get("course_title", "Professional Course"),
            "course_description": outline_content.get("course_description", ""),
            "modules": []
        }
        
        # Generate content for each module
        for i, module_outline in enumerate(outline_content.get("modules", [])):
            module_title = module_outline.get("title", f"Module {i+1}")
            module_objectives = module_outline.get("learning_objectives", [])
            
            with st.spinner(f"Generating content for Module {i+1}: {module_title}..."):
                # Generate detailed content for this specific module
                module_prompt = f"""
                Create detailed content for module "{module_title}" in a professional learning course.
                
                Context: {professional_context}
                Learning Objectives: {', '.join(module_objectives)}
                Document Summary: {document_summary}
                
                Provide:
                1. Detailed module content (at least 500 words) including examples, case studies, and practical applications
                2. A quiz with 3-5 thought-provoking questions for this module
                
                Return the response in the following JSON format:
                {{
                    "content": "Detailed module content...",
                    "quiz": {{
                        "questions": [
                            {{
                                "question": "Question text?",
                                "options": ["Option A", "Option B", "Option C", "Option D"],
                                "correct_answer": "Option A"
                            }}
                        ]
                    }}
                }}
                """
                
                # Create OpenAI client for module content
                module_response = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": module_prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                
                # Parse the module content
                module_content = json.loads(module_response.choices[0].message.content)
                
                # Add to complete course
                complete_course["modules"].append({
                    "title": module_title,
                    "learning_objectives": module_objectives,
                    "content": module_content.get("content", ""),
                    "quiz": module_content.get("quiz", {"questions": []})
                })
                
                # Add a small delay to avoid rate limits
                time.sleep(1)
        
        # Store the complete course
        st.session_state.course_content = complete_course
        st.session_state.course_generated = True
        
        # Count total questions for progress tracking
        total_questions = 0
        for module in st.session_state.course_content.get("modules", []):
            quiz = module.get("quiz", {})
            total_questions += len(quiz.get("questions", []))
        st.session_state.total_questions = total_questions
        
    except Exception as e:
        st.error(f"Error generating course: {e}")
    
    # Always reset the generation flag when done
    st.session_state.is_generating = False

# Main contents area with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources", "🔍 Knowledge Explorer"])

# Check if we're in the middle of generating a course and need to continue
if st.session_state.is_generating:
    with st.spinner("Generating your personalized course using vector embeddings and advanced RAG..."):
        # Reset completed questions when generating a new course
        st.session_state.completed_questions = set()
        perform_course_generation()
        st.success("✅ Your Comprehensive Course is Ready!")
        st.rerun()  # Refresh the UI after completion

with tab1:
    # Display Course Content
    if st.session_state.course_generated and st.session_state.course_content:
        course = st.session_state.course_content
        
        # Course Header with appreciation
        st.title(f"🌟 {course.get('course_title', 'Professional Course')}")
        st.markdown(f"*Specially designed for {role}s focusing on {', '.join(learning_focus)}*")
        st.write(course.get('course_description', 'A structured course to enhance your skills.'))
        
        # Tracking the Progress
        completed = len(st.session_state.completed_questions)
        total = st.session_state.total_questions
        progress_percentage = (completed / total * 100) if total > 0 else 0
        
        st.progress(progress_percentage / 100)
        st.write(f"**Progress:** {completed}/{total} questions completed ({progress_percentage:.1f}%)")
        # Add Download Report button
        st.download_button("📥 Download Progress Report", generate_progress_report(), "progress_report.pdf")
        
        st.markdown("---")
        st.subheader("📋 Course Overview")
        
        # Safely access module titles
        modules = course.get("modules", [])
        if modules:
            modules_list = [module.get('title', f'Module {i+1}') for i, module in enumerate(modules)]
            for i, module_title in enumerate(modules_list, 1):
                st.write(f"**Module {i}:** {module_title}")
        else:
            st.warning("No modules were found in the course content.")
        
        st.markdown("---")
        
        # Detailed Module Contents with improved formatting
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {module_title}"):
                # Module Learning Objectives
                st.markdown("### 🎯 Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                if objectives:
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                else:
                    st.write("No learning objectives specified.")
                
                # Module Content with better readability
                st.markdown("### 📖 Module Content:")
                module_content = module.get('content', 'No content available for this module.')
                
                # Split the content into paragraphs and add proper formatting
                paragraphs = module_content.split('\n\n')
                for para in paragraphs:
                    if para.strip().startswith('#'):
                        # Handle markdown headers
                        st.markdown(para)
                    elif para.strip().startswith('*') and para.strip().endswith('*'):
                        # Handle emphasized text
                        st.markdown(para)
                    elif para.strip().startswith('1.') or para.strip().startswith('- '):
                        # Handle lists
                        st.markdown(para)
                    else:
                        # Regular paragraphs
                        st.write(para)
                        st.write("")  # Add spacing between paragraphs
                
                # Key Takeaways section
                st.markdown("### 💡 Key Takeaways:")
                st.info("The content in this module will help you develop practical skills that you can apply immediately in your professional context.")
                
                # Module Quiz with improved UI
                st.markdown("### 📝 Module Quiz:")
                quiz = module.get('quiz', {})
                questions = quiz.get('questions', [])
                
                if questions:
                    for q_idx, q in enumerate(questions, 1):
                        question_id = f"module_{i}_question_{q_idx}"
                        question_text = q.get('question', f'Question {q_idx}')
                        
                        # Create quiz question container
                        quiz_container = st.container()
                        with quiz_container:
                            st.markdown(f"**Question {q_idx}:** {question_text}")
                            
                            options = q.get('options', [])
                            if options:
                                # Create a unique key for each radio button
                                option_key = f"quiz_{i}_{q_idx}"
                                user_answer = st.radio("Select your answer:", options, key=option_key)
                                
                                # Create a unique key for each submit button
                                submit_key = f"submit_{i}_{q_idx}"
                                
                                # Show completion status for this question
                                if question_id in st.session_state.completed_questions:
                                    st.success("✓ Question completed")
                                else:
                                    if st.button(f"Check Answer", key=submit_key):
                                        correct_answer = q.get('correct_answer', '')
                                        check_answer(question_id, user_answer, correct_answer)
                            else:
                                st.write("No options available for this question.")
                        
                        st.markdown("---")
                else:
                    st.write("No quiz questions available for this module.")

    else:
        # Welcome screen when no course is generated yet
        st.title("Welcome to Enhanced Professional Learning Platform")
        st.markdown("""
        ## Transform your professional development with AI-powered learning system
        
        Upload multiple PDF documents, and I'll create a comprehensive, integrated learning course just for you!
        
        ### How it works:
        1. Enter your OpenAI API key in the sidebar
        2. Select your professional role and learning focus
        3. Upload multiple PDF documents related to your area of interest
        4. Click "Generate Course" to create your personalized learning journey that combines insights from all documents
        
        ### Advanced Features:
        - **Vector Embeddings**: Your documents are analyzed using AI embeddings for deeper understanding
        - **ChromaDB Vector Store**: Efficient storage and retrieval of document knowledge
        - **LangChain Integration**: Advanced RAG (Retrieval Augmented Generation) for accurate answers
        - **Interactive Quizzes**: Test your knowledge with auto-generated quizzes
        - **Progress Tracking**: Monitor your learning journey with detailed reports
        
        Get ready to enhance your skills and accelerate your professional growth!
        """)
        
        # Generate Course Button - only if not currently generating
        if st.session_state.extracted_texts and openai_api_key and not st.session_state.is_generating:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate My Course", use_container_width=True):
                    generate_course()
        elif st.session_state.is_generating:
            st.info("Generating your personalized course... Please wait.")

with tab2:
    st.title("💬 Employer Queries")
    st.markdown("""
    This section allows employers to ask questions and get AI-generated answers about the course content or related topics.
    Submit your questions in the sidebar, and our AI will automatically generate answers based on the uploaded documents using vector embeddings for more accurate responses.
    """)
    
    if not st.session_state.employer_queries:
        st.info("No questions have been submitted yet. Add a question in the sidebar to get started.")
    else:
        for i, query in enumerate(st.session_state.employer_queries):
            with st.expander(f"Question {i+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {i+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                st.write(f"**Time:** {query.get('timestamp', 'N/A')}")
                
                if query.get('answered'):
                    st.write(f"**Answer:** {query['answer']}")
                else:
                    st.info("Generating answer...")
                    # Generate answer on-demand if not already answered
                    if st.session_state.extracted_texts:
                        try:
                            answer = generate_rag_answer(
                                query['question'], 
                                st.session_state.extracted_texts,
                                st.session_state.course_content if st.session_state.course_generated else None
                            )
                            st.session_state.employer_queries[i]['answer'] = answer
                            st.session_state.employer_queries[i]['answered'] = True
                            st.session_state.employer_queries[i]['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.rerun()
                        except Exception as e:
                            error_msg = f"Error generating answer: {str(e)}. Please try resetting the application."
                            st.error(error_msg)
                            st.session_state.employer_queries[i]['answer'] = error_msg
                            st.session_state.employer_queries[i]['answered'] = True
                    else:
                        st.warning("No documents uploaded yet. Please upload documents to generate answers.")

with tab3:
    st.title("📑 Document Sources")
    
    if not st.session_state.extracted_texts:
        st.info("No documents have been uploaded yet. Please upload PDF files in the sidebar to see their content here.")
    else:
        st.write(f"**{len(st.session_state.extracted_texts)} documents uploaded:**")
        
        for i, doc in enumerate(st.session_state.extracted_texts):
            with st.expander(f"Document {i+1}: {doc['filename']}"):
                # Display document preview (first 1000 characters)
                preview_text = doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text']
                st.markdown("### Document Preview:")
                st.text_area("Content Preview:", value=preview_text, height=300, disabled=True)
                
                # Add document summary using AI
                if st.button(f"Generate Summary for {doc['filename']}", key=f"sum_{i}"):
                    with st.spinner("Generating document summary using vector embeddings..."):
                        summary_query = f"Create a comprehensive summary of this document highlighting key concepts, theories, and practical applications:"
                        summary = generate_rag_answer(summary_query, [doc])
                        st.markdown("### AI-Generated Summary:")
                        st.write(summary)
                        
                # Add document analysis options
                analysis_options = ["Key Concepts", "Main Arguments", "Practical Applications", "Critical Analysis"]
                selected_analysis = st.selectbox(f"Analyze document {i+1}", ["Select analysis type"] + analysis_options, key=f"analysis_{i}")
                
                if selected_analysis != "Select analysis type":
                    with st.spinner(f"Performing {selected_analysis} analysis..."):
                        analysis_query = f"Provide a detailed {selected_analysis.lower()} analysis of this document:"
                        analysis = generate_rag_answer(analysis_query, [doc])
                        st.markdown(f"### {selected_analysis} Analysis:")
                        st.write(analysis)

with tab4:
    st.title("🔍 Knowledge Explorer")
    st.markdown("""
    This interactive tool allows you to explore the knowledge contained in your uploaded documents.
    Ask questions, discover connections, and get AI-powered insights using vector embeddings technology.
    """)
    
    # Interactive chat interface
    st.subheader("💬 Ask anything about your documents")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**You:** {chat['content']}")
            else:
                st.markdown(f"**AI:** {chat['content']}")
    
    # Input for new questions
    user_question = st.text_input("Your question:", key="knowledge_question")
    col1, col2 = st.columns([1, 5])
    
    with col1:
        send_button = st.button("Send", use_container_width=True)
    
    with col2:
        clear_chat = st.button("Clear Chat History", use_container_width=True)
        
    if clear_chat:
        st.session_state.chat_history = []
        st.rerun()
        
    if send_button and user_question:
        if not st.session_state.extracted_texts:
            st.error("Please upload documents first to enable the Knowledge Explorer.")
        else:
            with st.spinner("Searching knowledge base..."):
                response = generate_rag_answer(user_question, st.session_state.extracted_texts)
                st.rerun()
    
    # Knowledge visualization section
    st.markdown("---")
    st.subheader("📊 Knowledge Visualization")
    
    if st.session_state.extracted_texts:
        visualization_options = ["Topic Clusters", "Concept Map", "Key Terms Analysis", "Document Relationships"]
        viz_type = st.selectbox("Select visualization type:", visualization_options)
        
        if st.button("Generate Visualization"):
            with st.spinner(f"Generating {viz_type}..."):
                if viz_type == "Key Terms Analysis":
                    # Generate key terms analysis
                    key_terms_query = "Identify and list the top 10-15 most important terms or concepts across all documents with a brief explanation of each."
                    key_terms = generate_rag_answer(key_terms_query, st.session_state.extracted_texts)
                    
                    st.subheader("Key Terms Analysis")
                    st.write(key_terms)
                    
                elif viz_type == "Topic Clusters":
                    # Generate topic clusters
                    topics_query = "Identify 5-7 main topic clusters across all documents and list the key concepts within each cluster."
                    topics = generate_rag_answer(topics_query, st.session_state.extracted_texts)
                    
                    st.subheader("Topic Clusters")
                    st.write(topics)
                    
                elif viz_type == "Document Relationships":
                    # Generate document relationships
                    relationships_query = "Analyze how the documents relate to each other. Identify shared themes, complementary information, and any contradictions."
                    relationships = generate_rag_answer(relationships_query, st.session_state.extracted_texts)
                    
                    st.subheader("Document Relationships")
                    st.write(relationships)
                    
                else:  # Concept Map
                    # Generate concept map data
                    concept_map_query = "Create a concept map showing the main ideas and their relationships across all documents."
                    concept_map = generate_rag_answer(concept_map_query, st.session_state.extracted_texts)
                    
                    st.subheader("Concept Map Description")
                    st.write(concept_map)
    else:
        st.info("Please upload documents to enable knowledge visualization features.")