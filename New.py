import streamlit as st
import uuid
import json
import io
import pdfplumber
from openai import OpenAI
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Initialize Vector Search and LLM components
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI as LangchainLLM
from langchain.memory import ConversationBufferMemory

# App configuration
st.set_page_config(
    page_title="Professional Learning Hub",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
def init_session_state():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    # User preferences
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ""
    if 'user_role' not in st.session_state:
        st.session_state.user_role = ""
    if 'interests' not in st.session_state:
        st.session_state.interests = []
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "gpt-4o-mini"
    
    # Course data
    if 'learning_materials' not in st.session_state:
        st.session_state.learning_materials = []
    if 'file_names' not in st.session_state:
        st.session_state.file_names = []
    if 'course_data' not in st.session_state:
        st.session_state.course_data = None
    if 'generation_in_progress' not in st.session_state:
        st.session_state.generation_in_progress = False
    if 'is_course_ready' not in st.session_state:
        st.session_state.is_course_ready = False
    
    # Quiz tracking
    if 'answered_questions' not in st.session_state:
        st.session_state.answered_questions = set()
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    
    # Q&A tracking
    if 'user_questions' not in st.session_state:
        st.session_state.user_questions = []
    
    # Vector search components
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )


# PDF Processing functions
def extract_text_from_pdf(pdf_document):
    """Extract all text content from uploaded PDF"""
    pdf_document.seek(0)
    extracted_text = ""
    
    try:
        with pdfplumber.open(pdf_document) as pdf:
            for page in pdf.pages:
                page_content = page.extract_text() or ""
                extracted_text += page_content + "\n"
                
        return extracted_text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return ""


def create_knowledge_base():
    """Create a searchable knowledge base from processed documents"""
    if not st.session_state.learning_materials:
        st.warning("No learning materials have been processed yet.")
        return False
        
    if not st.session_state.api_key:
        st.warning("Please enter your OpenAI API key first.")
        return False
        
    try:
        # Create a text splitter for chunking
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Process all learning materials
        all_chunks = []
        for doc in st.session_state.learning_materials:
            document_chunks = text_splitter.split_text(doc["content"])
            all_chunks.extend(document_chunks)
            
        if not all_chunks:
            st.warning("No content was extracted from the documents.")
            return False
            
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
        st.session_state.knowledge_base = FAISS.from_texts(
            all_chunks, 
            embeddings
        )
        
        st.success("✅ Knowledge base successfully created!")
        return True
        
    except Exception as e:
        st.error(f"Error creating knowledge base: {str(e)}")
        return False


# Course generation functions
def generate_learning_path():
    """Generate personalized learning course using OpenAI"""
    if not st.session_state.knowledge_base:
        st.error("Knowledge base not initialized. Please upload and process documents first.")
        return
        
    if not st.session_state.api_key:
        st.error("OpenAI API key is required for course generation.")
        return
        
    try:
        # Prepare document context
        document_context = ""
        for i, doc in enumerate(st.session_state.learning_materials):
            document_context += f"\n--- Document {i+1}: {doc['filename']} ---\n"
            document_context += doc['content'][:2500]  # Limit context size
        
        # Get user preferences
        user_profile = f"Role: {st.session_state.user_role}, Interests: {', '.join(st.session_state.interests)}"
        
        # Generate document summary using retrieval
        summary_query = "Generate a comprehensive summary of these documents, identifying key themes, concepts, and actionable insights."
        summary = ask_knowledge_base(summary_query)
        
        # Prompt for course creation
        course_prompt = f"""
        Create a comprehensive professional learning course based on the uploaded documents.
        
        User profile: {user_profile}
        
        Document summary: {summary}
        
        Document samples: {document_context[:4000]}
        
        Design a well-structured course that:
        1. Has a compelling title that captures the essence of the learning materials
        2. Includes a detailed course description (300+ words) explaining the value and application of this knowledge
        3. Contains 5-7 logical modules that build on each other
        4. For each module, provide:
           - A clear title and 4-5 specific learning objectives
           - Comprehensive content (500+ words) with practical applications, examples, and guidelines
           - A quiz with 3-5 challenging multiple-choice questions
        
        Return your response in this JSON format:
        {{
            "title": "Course Title",
            "description": "Detailed course description",
            "modules": [
                {{
                    "title": "Module Title",
                    "objectives": ["Objective 1", "Objective 2", "..."],
                    "content": "Module content with detailed explanations and examples",
                    "assessment": {{
                        "questions": [
                            {{
                                "text": "Question text?",
                                "choices": ["Option A", "Option B", "Option C", "Option D"],
                                "answer": "Correct option"
                            }}
                        ]
                    }}
                }}
            ]
        }}
        
        Make the content extremely practical and actionable for {st.session_state.user_role} professionals.
        Include comparative perspectives where the documents present different approaches.
        """
        
        # Call OpenAI API
        client = OpenAI(api_key=st.session_state.api_key)
        response = client.chat.completions.create(
            model=st.session_state.model_choice,
            messages=[{"role": "user", "content": course_prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        course_data = json.loads(response.choices[0].message.content)
        st.session_state.course_data = course_data
        st.session_state.is_course_ready = True
        
        # Count total questions for progress tracking
        question_count = 0
        for module in course_data.get("modules", []):
            questions = module.get("assessment", {}).get("questions", [])
            question_count += len(questions)
        st.session_state.question_count = question_count
        
    except Exception as e:
        st.error(f"Error generating course: {str(e)}")
        
    st.session_state.generation_in_progress = False


def ask_knowledge_base(question):
    """Generate answer to question using RAG"""
    try:
        if not st.session_state.knowledge_base:
            return "Please upload and process documents first."
            
        if not st.session_state.api_key:
            return "OpenAI API key is required."
            
        # Create retrieval chain
        llm = LangchainLLM(
            openai_api_key=st.session_state.api_key,
            model_name=st.session_state.model_choice
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.knowledge_base.as_retriever(search_kwargs={"k": 4}),
            memory=st.session_state.memory
        )
        
        result = qa_chain({"question": question})
        return result["answer"]
        
    except Exception as e:
        return f"Error retrieving answer: {str(e)}"


def check_quiz_answer(question_id, user_choice, correct_answer):
    """Verify user's quiz answer and update progress"""
    if user_choice == correct_answer:
        st.success("✅ Correct answer! Well done!")
        st.session_state.answered_questions.add(question_id)
        return True
    else:
        st.error(f"Not quite right. The correct answer is: {correct_answer}")
        return False


def create_progress_report():
    """Generate downloadable PDF progress report"""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    
    # Add report header
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(72, 750, "Learning Progress Report")
    
    pdf.setFont("Helvetica", 12)
    pdf.drawString(72, 725, f"Date: {datetime.now().strftime('%B %d, %Y')}")
    pdf.drawString(72, 710, f"Role: {st.session_state.user_role}")
    pdf.drawString(72, 695, f"Interests: {', '.join(st.session_state.interests)}")
    
    # Add course info
    if st.session_state.course_data:
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(72, 665, f"Course: {st.session_state.course_data.get('title', 'Unnamed Course')}")
        
        # Add progress statistics
        completed = len(st.session_state.answered_questions)
        total = st.session_state.question_count
        completion_rate = (completed / total * 100) if total > 0 else 0
        
        pdf.setFont("Helvetica", 12)
        pdf.drawString(72, 640, f"Overall Progress: {completed} of {total} questions completed ({completion_rate:.1f}%)")
        
        # Add module breakdown
        y_position = 610
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(72, y_position, "Module Progress:")
        y_position -= 20
        
        # List each module
        for i, module in enumerate(st.session_state.course_data.get("modules", [])):
            module_title = module.get("title", f"Module {i+1}")
            pdf.setFont("Helvetica", 11)
            pdf.drawString(90, y_position, f"• {module_title}")
            y_position -= 15
    
    pdf.save()
    buffer.seek(0)
    return buffer


# UI layout and components
def sidebar_ui():
    """Create sidebar UI elements"""
    st.sidebar.title("🧠 Learning Hub")
    st.sidebar.markdown("---")
    
    # Reset application
    if st.sidebar.button("🔄 Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        init_session_state()
        st.rerun()
    
    # API Key input
    api_key = st.sidebar.text_input("🔑 OpenAI API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key
    
    # User profile settings
    st.sidebar.subheader("Profile Settings")
    
    roles = ["Executive", "Manager", "Developer", "Designer", "Marketer", 
             "HR Professional", "New Graduate", "Consultant"]
    user_role = st.sidebar.selectbox("Professional Role", roles)
    if user_role:
        st.session_state.user_role = user_role
    
    interest_options = ["Leadership", "Technical Skills", "Communication", 
                        "Project Management", "Innovation", "Team Building", 
                        "Data Analysis", "Strategic Planning"]
    interests = st.sidebar.multiselect("Learning Interests", interest_options)
    if interests:
        st.session_state.interests = interests
    
    # AI model selection
    models = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
    model_choice = st.sidebar.selectbox("AI Model", models)
    if model_choice:
        st.session_state.model_choice = model_choice
    
    # Document upload section
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Upload Learning Materials")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    # Process uploaded files
    if uploaded_files and st.session_state.api_key:
        current_files = [file.name for file in uploaded_files]
        
        # Check if files have changed
        if current_files != st.session_state.file_names:
            st.session_state.learning_materials = []
            st.session_state.file_names = current_files
            
            with st.sidebar.status("Processing documents..."):
                for pdf_file in uploaded_files:
                    text_content = extract_text_from_pdf(pdf_file)
                    if text_content:
                        st.session_state.learning_materials.append({
                            "filename": pdf_file.name,
                            "content": text_content
                        })
            
            if st.session_state.learning_materials:
                st.sidebar.success(f"✅ Processed {len(st.session_state.learning_materials)} documents")
                create_knowledge_base()
    
    # Display uploaded files
    if st.session_state.file_names:
        st.sidebar.markdown("---")
        st.sidebar.caption("📚 Uploaded Documents:")
        for i, filename in enumerate(st.session_state.file_names):
            st.sidebar.text(f"{i+1}. {filename}")
    
    # Course generation button
    if st.session_state.knowledge_base and st.session_state.api_key:
        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Generate Learning Path"):
            st.session_state.generation_in_progress = True
            st.session_state.is_course_ready = False
            st.rerun()
    
    # Ask question section
    st.sidebar.markdown("---")
    st.sidebar.subheader("❓ Ask About Materials")
    
    question = st.sidebar.text_area("Your question:")
    if st.sidebar.button("Ask Question"):
        if question:
            if st.session_state.knowledge_base and st.session_state.api_key:
                with st.sidebar.status("Generating answer..."):
                    answer = ask_knowledge_base(question)
                
                st.session_state.user_questions.append({
                    "question": question,
                    "answer": answer
                })
                st.sidebar.success("Question answered!")
                st.rerun()
            else:
                st.sidebar.error("Please upload documents and enter API key first")


def main_content():
    """Create main UI content area"""
    # Check if we're generating a course
    if st.session_state.generation_in_progress:
        with st.status("Generating personalized learning path...", expanded=True):
            st.write("Analyzing documents and creating course structure...")
            generate_learning_path()
            st.success("✅ Learning path created successfully!")
        st.rerun()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        "📚 Learning Path", 
        "❓ Q&A History", 
        "📑 Document Library"
    ])
    
    with tab1:
        # Learning path content
        if not st.session_state.is_course_ready:
            st.header("Welcome to Your Learning Hub")
            st.write("Upload your learning materials to get started. We'll analyze them and create a personalized learning path for you.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("📤 **Step 1:** Upload PDF documents via the sidebar")
            with col2:
                st.info("🚀 **Step 2:** Click 'Generate Learning Path' to create your personalized course")
                
            # Show example content
            if st.button("See Example Learning Path"):
                st.markdown("""
                ### Example: Digital Leadership Transformation
                
                This learning path combines multiple resources to help you develop essential leadership skills in digital environments:
                
                **Module Examples:**
                - Digital Leadership Foundations
                - Change Management in Digital Transformation
                - Building High-Performing Remote Teams
                - Data-Driven Decision Making
                - Innovation Culture & Design Thinking
                
                *Upload your documents to generate your own personalized learning path!*
                """)
        
        elif st.session_state.course_data:
            course = st.session_state.course_data
            
            # Course header
            st.title(f"📚 {course.get('title', 'Your Learning Path')}")
            st.caption(f"Personalized for {st.session_state.user_role} focused on {', '.join(st.session_state.interests)}")
            
            # Course description
            st.markdown(course.get('description', 'A personalized learning journey.'))
            
            # Progress tracker
            completed = len(st.session_state.answered_questions)
            total = st.session_state.question_count
            progress_pct = (completed / total * 100) if total > 0 else 0
            
            st.progress(progress_pct / 100)
            st.write(f"**Progress:** {completed}/{total} assessments completed ({progress_pct:.1f}%)")
            
            # Download progress report
            st.download_button(
                "📊 Download Progress Report",
                create_progress_report(),
                "learning_progress.pdf",
                "application/pdf"
            )
            
            st.markdown("---")
            
            # Module navigation
            st.subheader("📋 Course Overview")
            modules = course.get("modules", [])
            for i, module in enumerate(modules, 1):
                st.write(f"**Module {i}:** {module.get('title', f'Module {i}')}")
            
            st.markdown("---")
            
            # Module details with expandable sections
            for i, module in enumerate(modules, 1):
                module_title = module.get('title', f'Module {i}')
                with st.expander(f"Module {i}: {module_title}"):
                    # Learning objectives
                    st.markdown("### 🎯 Learning Objectives")
                    objectives = module.get('objectives', [])
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                    
                    # Module content
                    st.markdown("### 📖 Content")
                    content = module.get('content', 'No content available.')
                    
                    # Format content nicely
                    for paragraph in content.split('\n\n'):
                        if paragraph.strip().startswith('#'):
                            st.markdown(paragraph)
                        else:
                            st.write(paragraph)
                    
                    # Quiz section
                    st.markdown("### 📝 Assessment")
                    questions = module.get('assessment', {}).get('questions', [])
                    
                    for q_idx, question in enumerate(questions, 1):
                        question_id = f"m{i}_q{q_idx}"
                        question_text = question.get('text', f'Question {q_idx}')
                        choices = question.get('choices', [])
                        answer = question.get('answer', '')
                        
                        # Show question or completion status
                        if question_id not in st.session_state.answered_questions:
                            st.write(f"**Question {q_idx}:** {question_text}")
                            
                            if choices:
                                user_choice = st.radio(
                                    f"Select your answer for question {q_idx}:",
                                    choices,
                                    key=f"quiz_{question_id}"
                                )
                                
                                if st.button(f"Submit Answer", key=f"submit_{question_id}"):
                                    check_quiz_answer(question_id, user_choice, answer)
                            else:
                                st.warning("No choices available for this question.")
                        else:
                            st.success(f"✅ Completed Question {q_idx}: {question_text}")
    
    with tab2:
        # Q&A History
        st.header("❓ Question & Answer History")
        
        if not st.session_state.user_questions:
            st.info("Ask questions about the learning materials using the sidebar.")
        else:
            for i, qa in enumerate(st.session_state.user_questions, 1):
                with st.expander(f"Q{i}: {qa['question'][:80]}..."):
                    st.markdown("**Question:**")
                    st.write(qa['question'])
                    st.markdown("**Answer:**")
                    st.write(qa['answer'])
    
    with tab3:
        # Document Library
        st.header("📑 Document Library")
        
        if not st.session_state.learning_materials:
            st.info("No documents have been uploaded yet. Use the sidebar to upload PDF files.")
        else:
            for i, doc in enumerate(st.session_state.learning_materials, 1):
                with st.expander(f"Document {i}: {doc['filename']}"):
                    st.write(f"**Filename:** {doc['filename']}")
                    preview_len = min(800, len(doc['content']))
                    st.text_area(
                        "Preview:",
                        doc['content'][:preview_len] + ('...' if len(doc['content']) > preview_len else ''),
                        height=200,
                        disabled=True
                    )


# Main application
def main():
    init_session_state()
    sidebar_ui()
    main_content()


if __name__ == "__main__":
    main()
