import streamlit as st
import os
import tempfile
import json
import io
import pdfplumber
import uuid
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

# LangChain imports for document retrieval and embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Configure the Streamlit page layout
st.set_page_config(page_title="📚 Professional Learning Platform", layout="wide")

# Initialize session state variables for managing app state
if 'course_data' not in st.session_state:
    st.session_state.course_data = None
if 'course_ready' not in st.session_state:
    st.session_state.course_ready = False
if 'generating_course' not in st.session_state:
    st.session_state.generating_course = False
if 'answered_questions' not in st.session_state:
    st.session_state.answered_questions = set()
if 'total_quiz_questions' not in st.session_state:
    st.session_state.total_quiz_questions = 0
if 'processed_docs' not in st.session_state:
    st.session_state.processed_docs = []
if 'queries_list' not in st.session_state:
    st.session_state.queries_list = []
if 'unique_session' not in st.session_state:
    st.session_state.unique_session = str(uuid.uuid4())
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []
if 'uploaded_doc_names' not in st.session_state:
    st.session_state.uploaded_doc_names = []
if 'doc_vector_db' not in st.session_state:
    st.session_state.doc_vector_db = None

# Sidebar setup
st.sidebar.title("🎓 Professional Learning System")

# Reset button to clear session state
if st.sidebar.button("🔄 Reset Application"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.unique_session = str(uuid.uuid4())
    st.session_state.processed_docs = []
    st.session_state.uploaded_docs = []
    st.session_state.uploaded_doc_names = []
    st.session_state.doc_vector_db = None
    st.rerun()

# Input for OpenAI API key, with support for secrets.toml
api_key = st.secrets["openai"]["api_key"] if "openai" in st.secrets else st.sidebar.text_input("🔑 Enter your OpenAI API key", type="password")

# File uploader for PDFs
pdf_files = st.sidebar.file_uploader("📝 Upload Training PDFs", type=['pdf'], accept_multiple_files=True)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            extracted_text = ""
            for page in pdf.pages:
                text = page.extract_text() or ""
                extracted_text += text + "\n"
        return extracted_text
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
        return ""

# Process uploaded PDFs and store in session state
if pdf_files and api_key:
    current_doc_names = [file.name for file in pdf_files]
    if current_doc_names != st.session_state.uploaded_doc_names:
        st.session_state.processed_docs = []
        st.session_state.uploaded_docs = []
        st.session_state.uploaded_doc_names = current_doc_names
        doc_list = []
        
        with st.spinner("Processing your PDF documents..."):
            for pdf_file in pdf_files:
                text = extract_text_from_pdf(pdf_file)
                if text:
                    st.session_state.processed_docs.append({
                        "name": pdf_file.name,
                        "content": text
                    })
                    st.session_state.uploaded_docs.append(pdf_file)
                    doc_list.append(Document(page_content=text, metadata={"name": pdf_file.name}))
                    
        if st.session_state.processed_docs:
            st.sidebar.success(f"✅ {len(st.session_state.processed_docs)} PDFs processed successfully!")
            
            try:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                doc_chunks = splitter.split_documents(doc_list)
                
                embedding_model = OpenAIEmbeddings(api_key=api_key)
                # Use in-memory Chroma store for Streamlit Cloud
                st.session_state.doc_vector_db = Chroma.from_documents(
                    documents=doc_chunks,
                    embedding=embedding_model,
                    persist_directory=None
                )
            except Exception as e:
                st.error(f"Failed to initialize vector database: {e}. Please try resetting the app.")
                st.session_state.doc_vector_db = None
else:
    st.info("📥 Please provide your OpenAI API key and upload PDFs to start.")

# Model and role selection in sidebar
model_choices = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
selected_model = st.sidebar.selectbox("Select OpenAI Model", model_choices, index=0)

role_choices = ["Manager", "Executive", "Developer", "Designer", "Marketer", "Human Resources", "Other", "Fresher"]
user_role = st.sidebar.selectbox("Select Your Role", role_choices)

focus_areas = ["Leadership", "Technical Skills", "Communication", "Project Management", "Innovation", "Team Building", "Finance"]
selected_focus = st.sidebar.multiselect("Select Learning Focus", focus_areas)

# Display uploaded files in sidebar
if st.session_state.uploaded_doc_names:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Uploaded Files")
    for idx, doc_name in enumerate(st.session_state.uploaded_doc_names):
        st.sidebar.text(f"{idx+1}. {doc_name}")

# Function to generate answers using retrieved documents
def answer_with_retrieval(query, course_info=None):
    try:
        if not api_key:
            return "Please provide an OpenAI API key to proceed."
        
        if not st.session_state.doc_vector_db:
            return "Document vector database not initialized. Process some documents first."
            
        # Retrieve relevant document chunks with similarity scores
        doc_retriever = st.session_state.doc_vector_db.as_retriever(search_kwargs={"k": 3})
        search_results = st.session_state.doc_vector_db.similarity_search_with_score(query, k=3)
        
        # Check if any documents were retrieved
        if not search_results:
            return "No relevant documents found for this query. Please ask a question related to the uploaded document content."
        
        # Extract documents and their scores
        relevant_docs = [result[0] for result in search_results]
        scores = [result[1] for result in search_results]
        
        # Debug: Display similarity scores to understand retrieval performance
        st.write("**Debug: Similarity Scores for Retrieved Documents:**")
        for i, score in enumerate(scores):
            st.write(f"Document {i+1}: Score = {score}")
        
        # Set a less strict threshold for relevance (lower score means more relevant in Chroma)
        relevance_threshold = 0.9  # Increased from 0.8 to be less strict
        if min(scores) > relevance_threshold:
            return "The query does not appear to be relevant to the uploaded documents. Please ask a question related to the document content."
        
        # Build the document context
        doc_context = ""
        for doc in relevant_docs:
            doc_context += f"\nSource: {doc.metadata.get('name', 'Unknown')}\nText: {doc.page_content[:500]}...\n"
        
        # Explicitly exclude course information for employer queries
        course_details = "Course information is not used for employer queries as per the requirement."
        
        # Construct the prompt with explicit context
        full_prompt = (
            "You are an AI assistant for a professional learning platform. Provide a detailed and accurate answer to the following query "
            "using ONLY the document excerpts provided below. Do not use any external knowledge, course information, or assumptions beyond the provided documents. "
            "Be thorough and reference the documents where applicable.\n\n"
            f"Query: {query}\n\n"
            f"Document Excerpts:\n{doc_context}\n\n"
            "Answer the query comprehensively using only the provided document excerpts. If the information is insufficient, state so politely and suggest what might help."
        )
        
        # Initialize and call the LLM
        llm = ChatOpenAI(api_key=api_key, model=selected_model, temperature=0.5)
        response = llm.invoke(full_prompt)
        answer_text = response.content
        
        # Add references to the answer
        if relevant_docs:
            answer_text += "\n\n**Sources Referenced:**\n"
            for doc in relevant_docs:
                doc_name = doc.metadata.get("name", "Unknown")
                answer_text += f"- Source: {doc_name}\n"
        
        return answer_text
    except Exception as e:
        return f"Failed to generate answer: {str(e)}"

# Sidebar section for employer queries
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Employer Queries")

new_question = st.sidebar.text_area("Add a new question (related to uploaded documents):", height=100)
if st.sidebar.button("Submit Question"):
    if new_question:
        response = ""
        if st.session_state.doc_vector_db:
            with st.spinner("Generating response..."):
                try:
                    response = answer_with_retrieval(new_question, course_info=None)
                except Exception as e:
                    response = f"Failed to generate response: {str(e)}"
        else:
            response = "Please upload and process documents to enable answering questions."
        
        st.session_state.queries_list.append({
            "question": new_question,
            "response": response,
            "answered": bool(response)
        })
        st.sidebar.success("Question submitted and answered!")
        st.rerun()

# Function to validate quiz answers with enhanced normalization and debugging
def validate_answer(q_id, user_response, correct_response):
    # Ensure both inputs are strings and normalize them
    user_response_str = str(user_response).strip().lower() if user_response is not None else ""
    correct_response_str = str(correct_response).strip().lower() if correct_response is not None else ""
    
    # Debug: Show the values being compared
    st.write("**Debug: Answer Comparison**")
    st.write(f"User Response (normalized): '{user_response_str}'")
    st.write(f"Correct Answer (normalized): '{correct_response_str}'")
    
    if user_response_str == correct_response_str:
        st.success("🎉 Correct answer! Great job!")
        st.session_state.answered_questions.add(q_id)
        return True
    else:
        st.error(f"Incorrect. The correct answer is: {correct_response}")
        return False

# Function to generate a progress report PDF
def create_progress_report():
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    page_width, page_height = letter

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(page_width / 2, page_height - 50, "Professional Learning Platform")
    pdf.setFont("Helvetica", 12)
    pdf.drawCentredString(page_width / 2, page_height - 70, "Training Progress Report")
    pdf.line(50, page_height - 80, page_width - 50, page_height - 80)

    pdf.setFont("Helvetica", 12)
    y_pos = page_height - 110

    pdf.drawString(50, y_pos, f"User Role: {user_role}")
    y_pos -= 20
    pdf.drawString(50, y_pos, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    y_pos -= 20

    if st.session_state.course_data:
        course_name = st.session_state.course_data.get('course_title', 'Not Available')
        pdf.drawString(50, y_pos, f"Course: {course_name}")
        y_pos -= 20
    if selected_focus:
        pdf.drawString(50, y_pos, f"Learning Focus: {', '.join(selected_focus)}")
        y_pos -= 20

    y_pos -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y_pos, "Progress Overview:")
    pdf.setFont("Helvetica", 12)
    y_pos -= 20
    completed_count = len(st.session_state.answered_questions)
    total_count = st.session_state.total_quiz_questions
    progress_percent = (completed_count / total_count * 100) if total_count > 0 else 0
    pdf.drawString(50, y_pos, f"Questions Answered: {completed_count}/{total_count}")
    y_pos -= 20
    pdf.drawString(50, y_pos, f"Progress: {progress_percent:.1f}%")
    y_pos -= 20

    y_pos -= 10
    if progress_percent >= 75:
        summary_text = "Excellent progress! You're nearly finished!"
    elif progress_percent >= 50:
        summary_text = "Good work! You're more than halfway done!"
    elif progress_percent > 0:
        summary_text = "Nice start! Keep going to complete more sections!"
    else:
        summary_text = "Let's begin! Answer quiz questions to track progress."
    pdf.drawString(50, y_pos, f"Summary: {summary_text}")
    y_pos -= 20

    if st.session_state.answered_questions:
        y_pos -= 10
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y_pos, "Answered Questions:")
        pdf.setFont("Helvetica", 12)
        y_pos -= 20
        for q_id in sorted(st.session_state.answered_questions):
            try:
                module_idx = q_id.split('_')[1]
                question_idx = q_id.split('_')[3]
                pdf.drawString(50, y_pos, f"Module {module_idx}, Question {question_idx}")
                y_pos -= 15
                if y_pos < 50:
                    pdf.showPage()
                    y_pos = page_height - 50
            except IndexError:
                continue

    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawCentredString(page_width / 2, 30, f"Generated by Professional Learning Platform on {datetime.now().strftime('%Y-%m-%d')}")
    
    pdf.showPage()
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer

# Function to initiate course generation
def start_course_creation():
    st.session_state.generating_course = True
    st.session_state.course_ready = False
    st.rerun()

# Function to generate the course content
def generate_course_content():
    try:
        combined_content = ""
        for idx, doc in enumerate(st.session_state.processed_docs):
            doc_text = f"\n--- Document {idx+1}: {doc['name']} ---\n"
            doc_text += doc['content'][:3000]
            combined_content += doc_text + "\n\n"
        
        user_context = f"Role: {user_role}, Focus Areas: {', '.join(selected_focus)}"
        
        doc_summary_query = "Summarize these documents comprehensively, focusing on key concepts, theories, and practical applications."
        doc_summary = answer_with_retrieval(doc_summary_query)
        
        course_prompt = f"""
        Create a detailed professional learning course based on the provided documents.
        User Context: {user_context}
        Document Summary: {doc_summary}
        
        Document Content: {combined_content[:5000]}
        
        Develop a structured and engaging course by:
        1. Analyzing the documents to find common themes, concepts, and unique insights.
        2. Crafting a compelling course title that reflects the integrated knowledge.
        3. Writing a course description (minimum 300 words) explaining the synthesis of information.
        4. Creating 5-8 modules that progress logically.
        5. Defining 4-6 learning objectives per module with practical examples.
        6. Providing detailed module content (minimum 500 words each) with:
           - Real-world examples and case studies
           - Practical applications
           - Visual explanations where relevant
           - Step-by-step guides for complex topics
           - Comparisons of differing perspectives from documents
        7. Adding a quiz per module with 3-5 questions to test understanding.
        
        For each quiz question:
        - Provide exactly 4 options labeled as "A", "B", "C", "D" (e.g., ["A", "B", "C", "D"]).
        - Ensure each option is a single letter (A, B, C, or D) with no additional text or spaces.
        - Ensure the correct_answer is exactly one of the options (e.g., "A", "B", "C", or "D").
        - Ensure the correct_answer matches the option exactly (e.g., "A" not "a" or "A ").
        - Ensure the correct_answer is a string, not a number or other type.
        
        Return the result in JSON format:
        {{
            "course_title": "Course Title",
            "course_description": "Detailed description",
            "modules": [
                {{
                    "title": "Module Title",
                    "learning_objectives": ["Objective 1", "Objective 2"],
                    "content": "Detailed content with examples",
                    "quiz": {{
                        "questions": [
                            {{
                                "question": "Question text?",
                                "options": ["A", "B", "C", "D"],
                                "correct_answer": "A"
                            }}
                        ]
                    }}
                }}
            ]
        }}
        
        Ensure the content is practical, actionable, and tailored to the user context.
        """
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": course_prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        course_json = response.choices[0].message.content
        try:
            st.session_state.course_data = json.loads(course_json)
            st.session_state.course_ready = True
            total_questions = 0
            for module in st.session_state.course_data.get("modules", []):
                quiz = module.get("quiz", {})
                total_questions += len(quiz.get("questions", []))
            st.session_state.total_quiz_questions = total_questions
        except json.JSONDecodeError as e:
            st.error(f"Error parsing course JSON: {e}")
            st.text(course_json)
        
    except Exception as e:
        st.error(f"Course generation failed: {e}")
    
    st.session_state.generating_course = False

# Main tabs for navigation
content_tab, queries_tab, docs_tab = st.tabs(["📚 Course Content", "❓ Employer Queries", "📑 Document Sources"])

if st.session_state.generating_course:
    with st.spinner("Creating your personalized course..."):
        st.session_state.answered_questions = set()
        generate_course_content()
        st.success("✅ Course Generated Successfully!")
        st.rerun()

with content_tab:
    if st.session_state.course_ready and st.session_state.course_data:
        course = st.session_state.course_data
        
        st.title(f"🌟 {course.get('course_title', 'Professional Course')}")
        st.markdown(f"*Designed for {user_role}s focusing on {', '.join(selected_focus)}*")
        st.write(course.get('course_description', 'A course to boost your professional skills.'))
        
        completed = len(st.session_state.answered_questions)
        total = st.session_state.total_quiz_questions
        progress = (completed / total * 100) if total > 0 else 0
        
        st.progress(progress / 100)
        st.write(f"**Progress:** {completed}/{total} questions completed ({progress:.1f}%)")
        st.download_button("📥 Download Progress Report", create_progress_report(), "progress_report.pdf")
        
        st.markdown("---")
        st.subheader("📋 Course Overview")
        
        modules = course.get("modules", [])
        if modules:
            module_titles = [module.get('title', f'Module {i+1}') for i, module in enumerate(modules)]
            for i, title in enumerate(module_titles, 1):
                st.write(f"**Module {i}:** {title}")
        else:
            st.warning("No modules found in the course.")
        
        st.markdown("---")
        
        for i, module in enumerate(modules, 1):
            module_title = module.get('title', f'Module {i}')
            with st.expander(f"📚 Module {i}: {module_title}"):
                st.markdown("### 🎯 Learning Objectives:")
                objectives = module.get('learning_objectives', [])
                if objectives:
                    for obj in objectives:
                        st.markdown(f"- {obj}")
                else:
                    st.write("No objectives specified.")
                
                st.markdown("### 📖 Module Content:")
                content = module.get('content', 'No content available.')
                sections = content.split('\n\n')
                for section in sections:
                    if section.strip().startswith('#'):
                        st.markdown(section)
                    elif section.strip().startswith('*') and section.strip().endswith('*'):
                        st.markdown(section)
                    elif section.strip().startswith('1.') or section.strip().startswith('- '):
                        st.markdown(section)
                    else:
                        st.write(section)
                        st.write("")
                
                st.markdown("### 💡 Key Takeaways:")
                st.info("This module equips you with actionable skills for your professional growth.")
                
                st.markdown("### 📝 Module Quiz:")
                quiz = module.get("quiz", {})
                questions = quiz.get("questions", [])
                
                if questions:
                    for q_idx, q in enumerate(questions, 1):
                        q_id = f"module_{i}_question_{q_idx}"
                        q_text = q.get('question', f'Question {q_idx}')
                        
                        quiz_section = st.container()
                        with quiz_section:
                            st.markdown(f"**Question {q_idx}:** {q_text}")
                            options = q.get('options', [])
                            
                            # Debug: Display raw quiz data
                            st.write("**Debug: Raw Quiz Data**")
                            st.write(f"Question: {q_text}")
                            st.write(f"Options: {options}")
                            st.write(f"Correct Answer: {q.get('correct_answer', 'Not specified')}")
                            
                            if options:
                                option_key = f"quiz_{i}_{q_idx}"
                                user_answer = st.radio("Select your answer:", options, key=option_key)
                                submit_key = f"submit_{i}_{q_idx}"
                                if q_id in st.session_state.answered_questions:
                                    st.success("✓ Question completed")
                                else:
                                    if st.button(f"Check Answer", key=submit_key):
                                        correct_answer = q.get('correct_answer', '')
                                        validate_answer(q_id, user_answer, correct_answer)
                            else:
                                st.write("No options available for this question.")
                        
                        st.markdown("---")
                else:
                    st.write("No quiz questions available for this module.")

    else:
        st.title("Welcome to Professional Learning Platform")
        st.markdown("""
        ## Elevate your professional skills with AI-driven learning
        
        Upload your PDF documents, and I'll craft a tailored course for you!
        
        ### Steps to Begin:
        1. Ensure your OpenAI API key is set up (in secrets.toml for Streamlit Cloud or via the sidebar).
        2. Select your role and focus areas.
        3. Upload PDF documents relevant to your learning goals.
        4. Click "Generate Course" to start your learning journey.
        
        Let's boost your professional growth!
        """)
        
        if st.session_state.processed_docs and api_key and not st.session_state.generating_course:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Generate My Course", use_container_width=True):
                    start_course_creation()
        elif st.session_state.generating_course:
            st.info("Generating your course... Please wait.")

with queries_tab:
    st.title("💬 Employer Queries")
    st.markdown("""
    Employers can ask questions here to get AI-generated insights based solely on the uploaded documents.
    Submit your query in the sidebar, and the AI will respond with answers derived only from the document content.
    Unrelated questions will not be answered.
    """)
    
    if not st.session_state.queries_list:
        st.info("No queries submitted yet. Add a question in the sidebar to begin.")
    else:
        for idx, query in enumerate(st.session_state.queries_list):
            with st.expander(f"Question {idx+1}: {query['question'][:50]}..." if len(query['question']) > 50 else f"Question {idx+1}: {query['question']}"):
                st.write(f"**Question:** {query['question']}")
                if query['answered']:
                    st.write(f"**Answer:** {query['response']}")
                else:
                    st.info("Generating answer...")
                    if st.session_state.doc_vector_db:
                        try:
                            answer = answer_with_retrieval(query['question'], course_info=None)
                            st.session_state.queries_list[idx]['response'] = answer
                            st.session_state.queries_list[idx]['answered'] = True
                            st.rerun()
                        except Exception as e:
                            error_msg = f"Failed to generate answer: {str(e)}. Try resetting the app."
                            st.error(error_msg)
                            st.session_state.queries_list[idx]['response'] = error_msg
                            st.session_state.queries_list[idx]['answered'] = True
                    else:
                        st.warning("No documents uploaded. Please upload PDFs to enable answers.")

with docs_tab:
    st.title("📑 Document Sources")
    
    if not st.session_state.processed_docs:
        st.info("No documents uploaded yet. Upload PDFs in the sidebar to view them here.")
    else:
        st.write(f"**{len(st.session_state.processed_docs)} documents uploaded:**")
        
        for idx, doc in enumerate(st.session_state.processed_docs):
            with st.expander(f"Document {idx+1}: {doc['name']}"):
                preview = doc['content'][:1000] + "..." if len(doc['content']) > 1000 else doc['content']
                st.markdown("### Document Preview:")
                st.text_area("Content Preview:", value=preview, height=300, disabled=True)
                
                if st.button(f"Generate Summary for {doc['name']}", key=f"summary_{idx}"):
                    with st.spinner("Generating summary..."):
                        summary_query = f"Summarize the document '{doc['name']}' focusing on key concepts and practical applications."
                        summary = answer_with_retrieval(summary_query, course_info=None)
                        st.markdown("### AI-Generated Summary:")
                        st.write(summary)
