import streamlit as st
import os
import tempfile
import json
import io
import requests
import pdfplumber
import uuid
from openai import OpenAI 
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

# Page Configuration
st.set_page_config(page_title="📚 Professional Learning Platform", layout="wide")

# Setup Temporary Directory for ChromaDB
TEMP_DIR = tempfile.mkdtemp()
CHROMA_DB_DIR = os.path.join(TEMP_DIR, "chroma_db")

# Initializing sessions state variables
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
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False

# Sidebars Appearance
st.sidebar.title("🎓 Professional Learning System")

# Clear Sessions Button & Session Management
if st.sidebar.button("🔄 Reset Application"):
    # Clean up ChromaDB
    if os.path.exists(CHROMA_DB_DIR):
        import shutil
        try:
            shutil.rmtree(CHROMA_DB_DIR)
        except:
            pass
    
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.extracted_texts = []
    st.session_state.uploaded_files = []
    st.session_state.uploaded_file_names = []
    st.session_state.embeddings_created = False
    st.rerun()

# 🔐 OpenAI API Key Inputs
openai_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")

# 📄 Multi-File Uploader for PDFs
uploaded_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to extract text from PDF using PDFPlumber
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

# Function to create embeddings and setup vector store with LangChain & ChromaDB
def create_embeddings_from_pdfs(pdf_files, openai_api_key):
    if not pdf_files or not openai_api_key:
        return None
    
    try:
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Create text splitter for chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Process PDFs and create documents
        all_documents = []
        
        for i, pdf_file in enumerate(pdf_files):
            # Create a temporary file to work with PyPDFLoader
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                pdf_file.seek(0)
                temp_file.write(pdf_file.read())
                temp_file_path = temp_file.name
            
            try:
                # Use LangChain's PyPDFLoader
                loader = PyPDFLoader(temp_file_path)
                pdf_documents = loader.load()
                
                # Add source metadata
                for doc in pdf_documents:
                    doc.metadata["source"] = pdf_file.name
                    doc.metadata["doc_id"] = i
                
                # Split into chunks
                split_docs = text_splitter.split_documents(pdf_documents)
                all_documents.extend(split_docs)
                
            except Exception as e:
                st.error(f"Error processing {pdf_file.name}: {e}")
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        # Create or get ChromaDB client and collection
        if not os.path.exists(CHROMA_DB_DIR):
            os.makedirs(CHROMA_DB_DIR)
        
        # Create vector store from documents
        vector_store = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        
        return vector_store
        
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

# Process uploaded files and add to session state
if uploaded_files and openai_api_key:
    # Clear previous uploads if list has changed
    current_filenames = [file.name for file in uploaded_files]
    if current_filenames != st.session_state.uploaded_file_names:
        st.session_state.extracted_texts = []
        st.session_state.uploaded_files = []
        st.session_state.uploaded_file_names = current_filenames
        st.session_state.embeddings_created = False
        
        # Extract text from each PDF and store in session state
        with st.spinner("Processing PDF files..."):
            for pdf_file in uploaded_files:
                extracted_text = extract_pdf_text(pdf_file)
                if extracted_text:
                    st.session_state.extracted_texts.append({
                        "filename": pdf_file.name,
                        "text": extracted_text
                    })
                    st.session_state.uploaded_files.append(pdf_file)
        
        # Create embeddings with ChromaDB and LangChain
        with st.spinner("Creating document embeddings with LangChain..."):
            vector_store = create_embeddings_from_pdfs(
                st.session_state.uploaded_files, 
                openai_api_key
            )
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.embeddings_created = True
                st.sidebar.success(f"✅ {len(st.session_state.extracted_texts)} PDF files processed and embedded!")
            else:
                st.sidebar.error("Failed to create embeddings. Please check logs and try again.")
            
    elif not st.session_state.embeddings_created and st.session_state.extracted_texts:
        # Create embeddings if not already created
        with st.spinner("Creating document embeddings with LangChain..."):
            vector_store = create_embeddings_from_pdfs(
                st.session_state.uploaded_files, 
                openai_api_key
            )
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.embeddings_created = True
                st.sidebar.success("✅ Document embeddings created successfully!")
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

# Enhanced RAG function using LangChain and ChromaDB
def generate_rag_answer(question, documents=None, course_content=None):
    try:
        if not openai_api_key:
            return "API key is required to generate answers."
        
        # Use vector store if available
        if st.session_state.vector_store and st.session_state.embeddings_created:
            # Create LangChain ChatOpenAI instance
            llm = ChatOpenAI(
                model_name=selected_model,
                openai_api_key=openai_api_key,
                temperature=0.5
            )
            
            # Create retrieval chain
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Add course content context if available
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
            
            # Custom prompt with course context
            from langchain.prompts import PromptTemplate
            
            prompt_template = """You are an AI assistant for a professional learning platform. 
            Use the following pieces of context to answer the question at the end.
            
            {context}
            
            Additional Course Information: {course_info}
            
            Question: {question}
            
            Provide a comprehensive answer using information from the context and course contents.
            If the question cannot be answered based on the provided information, say so politely.
            Reference specific documents when appropriate in your answer.
            """
            
            qa_prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "course_info"]
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": qa_prompt},
                return_source_documents=True
            )
            
            # Run chain with course context
            result = qa_chain({"question": question, "course_info": course_context})
            return result["result"]
            
        else:
            # Fallback to direct text search if vector store not available
            if not documents:
                return "Document text is not available. Please process documents first."
                
            # Create a context from all document texts (with file attribution)
            combined_context = ""
            for i, doc in enumerate(documents[:3]):  # Limit to first 3 documents to avoid token issues
                context_chunk = doc["text"][:2000]  # Limit each doc to 2000 chars
                combined_context += f"\nDocument {i+1} ({doc['filename']}):\n{context_chunk}\n"
            
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
            return response.choices[0].message.content
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Employee Queries Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employer Queries")

new_query = st.sidebar.text_area("Add a new question:", height=100)
if st.sidebar.button("Submit Question"):
    if new_query:
        # Generate proper answers automatically if documents are available
        answer = ""
        if st.session_state.embeddings_created:
            with st.spinner("Generating answer with LangChain & ChromaDB..."):
                answer = generate_rag_answer(
                    new_query, 
                    None,  # Don't need documents with vector store
                    st.session_state.course_content if st.session_state.course_generated else None
                )
        elif st.session_state.extracted_texts:
            with st.spinner("Generating answer with direct text search..."):
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
            "answered": bool(answer)
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
    c.drawString(100, 750, "Training Progress Report")
    c.drawString(100, 730, f"User Role: {role}")
    c.drawString(100, 710, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    completed = len(st.session_state.completed_questions)
    total = st.session_state.total_questions
    c.drawString(100, 690, f"Progress: {completed}/{total} questions completed ({completed/total*100:.1f}%)")
    
    # Add document information
    c.drawString(100, 650, "Documents Studied:")
    y_pos = 630
    for i, filename in enumerate(st.session_state.uploaded_file_names):
        c.drawString(120, y_pos, f"{i+1}. {filename}")
        y_pos -= 20
    
    # Add module completion information if course is generated
    if st.session_state.course_generated and st.session_state.course_content:
        course = st.session_state.course_content
        c.drawString(100, y_pos-20, "Course Modules:")
        y_pos -= 40
        
        for i, module in enumerate(course.get("modules", []), 1):
            module_title = module.get('title', f'Module {i}')
            c.drawString(120, y_pos, f"{i}. {module_title}")
            y_pos -= 20
    
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

# Function to actually generate the course content using LangChain
def perform_course_generation():
    try:
        # Create LangChain documents from vector store for course generation
        relevant_docs = None
        
        if st.session_state.vector_store and st.session_state.embeddings_created:
            # Get most relevant documents from vector store for course creation
            query = f"Create a professional training course for {role} focusing on {', '.join(learning_focus)}"
            relevant_docs = st.session_state.vector_store.similarity_search(query, k=15)
            
            # Extract text from relevant docs
            combined_docs = ""
            for i, doc in enumerate(relevant_docs):
                doc_text = doc.page_content
                doc_source = doc.metadata.get("source", f"Document {i}")
                combined_docs += f"\n--- FROM {doc_source} ---\n{doc_text}\n\n"
        else:
            # Fallback to direct document text
            combined_docs = ""
            for i, doc in enumerate(st.session_state.extracted_texts):
                doc_summary = f"\n--- DOCUMENT {i+1}: {doc['filename']} ---\n"
                doc_summary += doc['text'][:3000]  # Limit each doc to avoid token limits
                combined_docs += doc_summary + "\n\n"
        
        professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
        
        # Get a document summary first using LangChain if available
        if st.session_state.vector_store and st.session_state.embeddings_created:
            llm = ChatOpenAI(
                model_name=selected_model,
                openai_api_key=openai_api_key,
                temperature=0.7
            )
            
            from langchain.chains.summarize import load_summarize_chain
            from langchain.prompts import PromptTemplate
            
            summary_prompt_template = """Write a comprehensive summary of the following documents,
            highlighting key concepts, theories, and practical applications:
            
            {text}
            
            COMPREHENSIVE SUMMARY:"""
            
            summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["text"])
            
            if relevant_docs:
                summary_chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=summary_prompt,
                    combine_prompt=summary_prompt,
                    verbose=False
                )
                
                document_summary = summary_chain.run(relevant_docs)
            else:
                document_summary = "No document summary available."
        else:
            # Fallback to OpenAI direct call
            summary_query = "Create a comprehensive summary of these documents highlighting key concepts, theories, and practical applications across all materials."
            document_summary = generate_rag_answer(summary_query, st.session_state.extracted_texts)
        
        prompt = f"""
        Design a comprehensive professional learning course based on the multiple documents provided.
        Context: {professional_context}
        Document Summary: {document_summary}
        
        Document Contents: {combined_docs[:5000]}
        
        Create an engaging, thorough and well-structured course by:
        1. Analyzing all provided documents and identifying common themes, complementary concepts, and unique insights from each source
        2. Creating an inspiring course title that reflects the integrated knowledge from all documents
        3. Writing a detailed course description (at least 300 words) that explains how the course synthesizes information from multiple sources
        4. Developing 5-8 comprehensive modules that build upon each other in a logical sequence
        5. Providing 4-6 clear learning objectives for each module with specific examples and practical applications
        6. Creating detailed, well-explained content for each module (at least 500 words per module) including:
           - Real-world examples and case studies
           - Practical applications of concepts
           - Visual explanations where appropriate
           - Step-by-step guides for complex procedures
           - Comparative analysis when sources present different perspectives
        7. Including a quiz with 3-5 thought-provoking questions per module for better understanding
        
        Return the response in the following JSON format:
        {{
            "course_title": "Your Course Title",
            "course_description": "Detailed description of the course",
            "modules": [
                {{
                    "title": "Module 1 Title",
                    "learning_objectives": ["Objective 1", "Objective 2", "Objective 3"],
                    "content": "Module content text with detailed explanations, examples, and practical applications",
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
            ]
        }}
        
        Make the content exceptionally practical, actionable, and tailored to the professional context.
        Provide detailed explanations, real-world examples, and practical applications in each module content.
        Where document sources provide different perspectives or approaches to the same topic, compare and contrast them.
        """
        
        try:
            # Try using LangChain first
            if st.session_state.vector_store and st.session_state.embeddings_created:
                llm = ChatOpenAI(
                    model_name=selected_model,
                    openai_api_key=openai_api_key,
                    temperature=0.7
                )
                response_content = llm.predict(prompt)
            else:
                # Fallback to direct OpenAI API
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                response_content = response.choices[0].message.content
            
            try:
                st.session_state.course_content = json.loads(response_content)
                st.session_state.course_generated = True
                
                # Count total questions for progress tracking
                total_questions = 0
                for module in st.session_state.course_content.get("modules", []):
                    quiz = module.get("quiz", {})
                    total_questions += len(quiz.get("questions", []))
                st.session_state.total_questions = total_questions
                
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON response: {e}")
                st.text(response_content)
        
        except Exception as e:
            st.error(f"API Error: {e}")
            st.error("Please check your API key and model selection.")
            
    except Exception as e:
        st.error(f"Error: {e}")
    
    # Always reset the generation flag when done
    st.session_state.is_generating = False

# Main contents area with tabs
tab1, tab2, tab3 = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources"])

# Check if we're in the middle of generating a course and need to continue
if st.session_state.is_generating:
    with st.spinner("Generating your personalized course from multiple documents..."):
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
        st.title("Welcome to Professional Learning Platform")
        st.markdown("""
        ## Transform your professional development with AI-powered learning system
        
        Upload multiple PDF documents, and I'll create a comprehensive, integrated learning course just for you!
        
        ### How it works:
        1. Enter your OpenAI API key in the sidebar
        2. Select your professional role and learning focus
        3. Upload multiple PDF documents related to your area of interest
        4. Click "Generate Course" to create your personalized learning journey that combines insights from all documents
        
        This enhanced version now uses:
        - ChromaDB for vector search
        - LangChain embeddings for semantic understanding
        - Advanced RAG techniques for more accurate question answering
        
        Get ready to enhance your skills and accelerate your professional growth!
        """
        
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
    Submit your questions in the sidebar, and our AI will automatically generate answers based on the uploaded documents.
    
    The system now uses LangChain and ChromaDB for more accurate semantic search and retrieval!
    """)
    
    if not st.session_state.employer_queries:
        st.info("No questions have been submitted yet. Add a question in the sidebar to get started.")
    else:
        for i, query in enumerate(st.session_state.employer_queries):
            with st.expander(f"Question {i+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {i+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                
                if query['answered']:
                    st.write(f"**Answer:** {query['answer']}")
                else:
                    st.info("Generating answer...")
                    # Generate answer on-demand if not already answered
                    if st.session_state.embeddings_created:
                        try:
                            answer = generate_rag_answer(
                                query['question'], 
                                None,  # Don't need documents with vector store
                                st.session_state.course_content if st.session_state.course_generated else None
                            )
                            st.session_state.employer_queries[i]['answer'] = answer
                            st.session_state.employer_queries[i]['answered'] = True
                            st.rerun()
                        except Exception as e:
                            error_msg = f"Error generating answer: {str(e)}. Please try resetting the application."
                            st.error(error_msg)
                            st.session_state.employer_queries[i]['answer'] = error_msg
                            st.session_state.employer_queries[i]['answered'] = True
                    elif st.session_state.extracted_texts:
                        try:
                            answer = generate_rag_answer(
                                query['question'], 
                                st.session_state.extracted_texts,
                                st.session_state.course_content if st.session_state.course_generated else None
                            )
                            st.session_state.employer_queries[i]['answer'] = answer
                            st.session_state.employer_queries[i]['answered'] = True
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
        
        # Add document stats and visualization
        if st.session_state.embeddings_created:
            st.success("✅ Documents have been embedded with LangChain and stored in ChromaDB!")
            
            # Stats about documents
            total_chars = sum(len(doc['text']) for doc in st.session_state.extracted_texts)
            avg_chars = total_chars / len(st.session_state.extracted_texts)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Documents", len(st.session_state.extracted_texts))
                st.metric("Total Characters", f"{total_chars:,}")
            with col2:
                st.metric("Avg Characters per Doc", f"{avg_chars:,.0f}")
                
                # If we have relevant docs from the vector store
                if st.session_state.vector_store:
                    try:
                        # Get document count from ChromaDB
                        chroma_count = len(st.session_state.vector_store.get())
                        st.metric("Embedded Chunks", chroma_count)
                    except:
                        pass
        
        # Display documents with options to view or get summaries
        for i, doc in enumerate(st.session_state.extracted_texts):
            with st.expander(f"Document {i+1}: {doc['filename']}"):
                tab_preview, tab_summary = st.tabs(["Preview", "AI Summary"])
                
                with tab_preview:
                    # Display document preview (first 1000 characters)
                    preview_text = doc['text'][:1000] + "..." if len(doc['text']) > 1000 else doc['text']
                    st.markdown("### Document Preview:")
                    st.text_area("Content Preview:", value=preview_text, height=300, disabled=True)
                
                with tab_summary:
                    # Add document summary using AI
                    if st.button(f"Generate Summary for {doc['filename']}", key=f"sum_{i}"):
                        with st.spinner("Generating document summary..."):
                            summary_query = f"Create a comprehensive summary of this document highlighting key concepts, theories, and practical applications:"
                            
                            if st.session_state.embeddings_created:
                                # Use LangChain for summary if available
                                try:
                                    llm = ChatOpenAI(
                                        model_name=selected_model,
                                        openai_api_key=openai_api_key,
                                        temperature=0.5
                                    )
                                    
                                    # Create a document for summarization
                                    from langchain.schema import Document
                                    from langchain.chains.summarize import load_summarize_chain
                                    
                                    doc_to_summarize = Document(
                                        page_content=doc['text'][:5000],
                                        metadata={"source": doc['filename']}
                                    )
                                    
                                    chain = load_summarize_chain(llm, chain_type="stuff")
                                    summary = chain.run([doc_to_summarize])
                                except Exception as e:
                                    summary = generate_rag_answer(summary_query, [doc])
                            else:
                                summary = generate_rag_answer(summary_query, [doc])
                                
                            st.markdown("### AI-Generated Summary:")
                            st.write(summary)

# Add a new tab for AI-powered analysis
tab4 = st.tabs(["🧠 AI Analysis"])[0]

with tab4:
    st.title("🧠 AI Document Analysis")
    
    if not st.session_state.extracted_texts:
        st.info("Upload PDF documents first to enable AI analysis.")
    elif not openai_api_key:
        st.warning("Please enter your OpenAI API key to use AI analysis features.")
    else:
        st.write("Use AI to extract key insights from your uploaded documents.")
        
        analysis_options = st.selectbox(
            "Select Analysis Type:",
            ["Document Comparison", "Key Concepts Extraction", "Learning Recommendations", "Content Gap Analysis"]
        )
        
        if analysis_options == "Document Comparison":
            st.subheader("Document Comparison")
            
            if len(st.session_state.extracted_texts) < 2:
                st.warning("Please upload at least two documents for comparison.")
            else:
                doc_options = [doc['filename'] for doc in st.session_state.extracted_texts]
                col1, col2 = st.columns(2)
                
                with col1:
                    doc1 = st.selectbox("Select first document:", doc_options, key="doc1")
                with col2:
                    doc2 = st.selectbox("Select second document:", doc_options, key="doc2")
                
                if st.button("Compare Documents"):
                    with st.spinner("Analyzing documents..."):
                        # Find the selected documents
                        doc1_content = next((doc['text'] for doc in st.session_state.extracted_texts if doc['filename'] == doc1), "")
                        doc2_content = next((doc['text'] for doc in st.session_state.extracted_texts if doc['filename'] == doc2), "")
                        
                        # Use LangChain for comparison
                        if st.session_state.embeddings_created:
                            try:
                                llm = ChatOpenAI(
                                    model_name=selected_model,
                                    openai_api_key=openai_api_key,
                                    temperature=0.5
                                )
                                
                                comparison_prompt = f"""
                                Compare and contrast these two documents:
                                
                                DOCUMENT 1 ({doc1}): {doc1_content[:2000]}...
                                
                                DOCUMENT 2 ({doc2}): {doc2_content[:2000]}...
                                
                                Provide a detailed analysis that includes:
                                1. Main topics covered in each document
                                2. Key similarities between documents
                                3. Important differences in approach or content
                                4. How these documents complement each other
                                5. Recommendations on which document is more suitable for different learning objectives
                                """
                                
                                comparison_result = llm.predict(comparison_prompt)
                            except:
                                # Fallback to direct OpenAI API
                                client = OpenAI(api_key=openai_api_key)
                                response = client.chat.completions.create(
                                    model=selected_model,
                                    messages=[{"role": "user", "content": f"Compare and contrast these two documents: Document 1 ({doc1}) and Document 2 ({doc2}). Identify key similarities, differences, and how they complement each other."}],
                                    temperature=0.5
                                )
                                comparison_result = response.choices[0].message.content
                        else:
                            # Use direct API call
                            client = OpenAI(api_key=openai_api_key)
                            response = client.chat.completions.create(
                                model=selected_model,
                                messages=[{"role": "user", "content": f"Compare and contrast these two documents: Document 1 ({doc1}) and Document 2 ({doc2}). Identify key similarities, differences, and how they complement each other."}],
                                temperature=0.5
                            )
                            comparison_result = response.choices[0].message.content
                        
                        st.markdown("### Comparison Results:")
                        st.write(comparison_result)
        
        elif analysis_options == "Key Concepts Extraction":
            st.subheader("Key Concepts Extraction")
            
            if st.button("Extract Key Concepts"):
                with st.spinner("Analyzing documents..."):
                    # Use LangChain and vector store for better concept extraction
                    if st.session_state.embeddings_created:
                        try:
                            # Get summary from vector store
                            llm = ChatOpenAI(
                                model_name=selected_model,
                                openai_api_key=openai_api_key,
                                temperature=0.3
                            )
                            
                            # Use retrieval to get key documents first
                            retriever = st.session_state.vector_store.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 10}
                            )
                            
                            docs = retriever.get_relevant_documents("identify important concepts topics themes across all documents")
                            
                            concept_prompt = """
                            Based on the documents, identify and explain the 5-10 most important concepts, themes, or topics.
                            For each concept:
                            1. Provide a clear name/title
                            2. Give a detailed explanation
                            3. Note which documents discuss this concept
                            4. Explain its practical importance
                            """
                            
                            # Use a chain to process the documents
                            from langchain.chains.summarize import load_summarize_chain
                            from langchain.prompts import PromptTemplate
                            
                            concept_chain = load_summarize_chain(
                                llm,
                                chain_type="stuff",
                                prompt=PromptTemplate(
                                    template=concept_prompt + "\n\nDocuments: {text}\n\nConcepts:",
                                    input_variables=["text"]
                                )
                            )
                            
                            concepts_result = concept_chain.run(docs)
                        except Exception as e:
                            # Fallback to direct API
                            client = OpenAI(api_key=openai_api_key)
                            concept_prompt = "Extract and explain the 7-10 most important concepts from these documents."
                            
                            # Combine some text from each document
                            combined_text = ""
                            for doc in st.session_state.extracted_texts[:3]:  # Limit to 3 docs
                                combined_text += f"\nFrom {doc['filename']}:\n{doc['text'][:1500]}\n"
                            
                            response = client.chat.completions.create(
                                model=selected_model,
                                messages=[{"role": "user", "content": concept_prompt + combined_text}],
                                temperature=0.3
                            )
                            concepts_result = response.choices[0].message.content
                    else:
                        # Use direct OpenAI API
                        client = OpenAI(api_key=openai_api_key)
                        concept_prompt = "Extract and explain the 7-10 most important concepts from these documents."
                        
                        # Combine some text from each document
                        combined_text = ""
                        for doc in st.session_state.extracted_texts[:3]:  # Limit to 3 docs
                            combined_text += f"\nFrom {doc['filename']}:\n{doc['text'][:1500]}\n"
                        
                        response = client.chat.completions.create(
                            model=selected_model,
                            messages=[{"role": "user", "content": concept_prompt + combined_text}],
                            temperature=0.3
                        )
                        concepts_result = response.choices[0].message.content
                    
                    st.markdown("### Key Concepts:")
                    st.write(concepts_result)
        
        elif analysis_options == "Learning Recommendations":
            st.subheader("Learning Recommendations")
            st.write("Get personalized learning recommendations based on your role and documents.")
            
            if st.button("Generate Learning Recommendations"):
                with st.spinner("Creating personalized recommendations..."):
                    # Use combined approach for recommendations
                    professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
                    
                    if st.session_state.embeddings_created:
                        try:
                            llm = ChatOpenAI(
                                model_name=selected_model,
                                openai_api_key=openai_api_key,
                                temperature=0.7
                            )
                            
                            recommendation_prompt = f"""
                            Based on the uploaded documents and the user's professional context:
                            {professional_context}
                            
                            Provide detailed learning recommendations including:
                            1. 5-7 specific skills they should focus on developing
                            2. Which documents are most relevant for their role and why
                            3. Suggested learning path with specific topics to master first
                            4. Practical projects they could undertake to apply this knowledge
                            5. How these skills will benefit them in their professional role
                            
                            Be specific, practical, and tailored to their context.
                            """
                            
                            # Use retrieval to get context
                            retriever = st.session_state.vector_store.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 10}
                            )
                            docs = retriever.get_relevant_documents(f"Learning recommendations for {role} {' '.join(learning_focus)}")
                            
                            # Create a context from the docs
                            docs_context = "\n".join([doc.page_content for doc in docs[:5]])
                            
                            rec_result = llm.predict(recommendation_prompt + "\n\nDocument context: " + docs_context)
                        except:
                            # Fallback
                            client = OpenAI(api_key=openai_api_key)
                            response = client.chat.completions.create(
                                model=selected_model,
                                messages=[{"role": "user", "content": f"Create learning recommendations for a {role} focused on {', '.join(learning_focus)} based on the uploaded documents."}],
                                temperature=0.7
                            )
                            rec_result = response.choices[0].message.content
                    else:
                        # Use direct API
                        client = OpenAI(api_key=openai_api_key)
                        response = client.chat.completions.create(
                            model=selected_model,
                            messages=[{"role": "user", "content": f"Create learning recommendations for a {role} focused on {', '.join(learning_focus)} based on the uploaded documents."}],
                            temperature=0.7
                        )
                        rec_result = response.choices[0].message.content
                    
                    st.markdown("### Your Personalized Learning Recommendations:")
                    st.write(rec_result)
                    
        elif analysis_options == "Content Gap Analysis":
            st.subheader("Content Gap Analysis")
            st.write("Identify missing topics or areas not well covered in your documents.")
            
            if st.button("Perform Gap Analysis"):
                with st.spinner("Analyzing content gaps..."):
                    professional_context = f"Role: {role}, Focus: {', '.join(learning_focus)}"
                    
                    if st.session_state.embeddings_created:
                        try:
                            llm = ChatOpenAI(
                                model_name=selected_model,
                                openai_api_key=openai_api_key,
                                temperature=0.5
                            )
                            
                            gap_prompt = f"""
                            Considering the user's professional context ({professional_context}),
                            analyze the uploaded documents and identify important gaps in content that would be valuable for this user.
                            
                            Provide a detailed gap analysis including:
                            1. Key topics that are missing or insufficiently covered
                            2. Important practical skills not addressed
                            3. Missing theoretical foundations that would be valuable
                            4. Suggestions for additional resources or topics to study
                            5. How filling these gaps would benefit their professional development
                            
                            Be specific about what's missing and why it matters for their role and focus areas.
                            """
                            
                            # Use retrieval for context
                            retriever = st.session_state.vector_store.as_retriever(
                                search_type="similarity",
                                search_kwargs={"k": 15}
                            )
                            docs = retriever.get_relevant_documents(f"Important topics for {role} {' '.join(learning_focus)}")
                            
                            # Create a context from the docs
                            docs_context = "\n".join([doc.page_content for doc in docs[:7]])
                            
                            gap_result = llm.predict(gap_prompt + "\n\nDocument context: " + docs_context)
                        except:
                            # Fallback
                            client = OpenAI(api_key=openai_api_key)
                            response = client.chat.completions.create(
                                model=selected_model,
                                messages=[{"role": "user", "content": f"For a {role} focused on {', '.join(learning_focus)}, analyze what important topics are missing from the uploaded documents."}],
                                temperature=0.5
                            )
                            gap_result = response.choices[0].message.content
                    else:
                        # Use direct API
                        client = OpenAI(api_key=openai_api_key)
                        response = client.chat.completions.create(
                            model=selected_model,
                            messages=[{"role": "user", "content": f"For a {role} focused on {', '.join(learning_focus)}, analyze what important topics are missing from the uploaded documents."}],
                            temperature=0.5
                        )
                        gap_result = response.choices[0].message.content
                    
                    st.markdown("### Content Gap Analysis:")
                    st.write(gap_result)

# Add about section in footer
st.markdown("---")
st.markdown("### About This Learning Platform")
st.markdown("""
This professional learning platform uses cutting-edge AI technologies:
- **ChromaDB**: For powerful vector search and document retrieval
- **LangChain**: For advanced embeddings and AI chains
- **PDFPlumber**: For extracting text from PDF documents
- **Streamlit**: For building the interactive web interface
- **ReportLab**: For generating progress reports
- **OpenAI API**: For generating course content and answering questions

Powered by natural language processing to deliver personalized learning experiences!
""")
