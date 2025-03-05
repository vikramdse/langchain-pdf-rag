import streamlit as st
import requests
import os
from typing import Optional
import time

# API Configuration
API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False


def upload_pdf(file):
    """Upload PDF file to the backend"""
    if file is None:
        return False, "No file selected"
    
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code == 201:
            return True, "PDF uploaded successfully"
        else:
            return False, f"Error: {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"


def ask_question(question: str) -> Optional[str]:
    """Send question to backend and get answer"""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": question}
        )
        
        if response.status_code == 200:
            return response.json().get("answer")
        else:
            error_msg = response.json().get("detail", "Unknown error")
            return f"Error: {error_msg}"
    except Exception as e:
        return f"Connection error: {str(e)}"


# App header
st.title("📚 PDF Question Answering System")
st.markdown("""
Upload a PDF document and ask questions about its content.
This application uses LangChain, OpenAI embeddings, and a RAG architecture to provide accurate answers.
""")

# Sidebar for PDF upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing PDF..."):
            success, message = upload_pdf(uploaded_file)
            if success:
                st.session_state.pdf_uploaded = True
                st.success(message)
            else:
                st.error(message)
    
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. Upload a PDF document
    2. The system processes and indexes the content
    3. Ask questions about the document
    4. Get answers based on the document's content
    """)

# Main content area
if st.session_state.pdf_uploaded:
    st.header("Ask Questions")
    question = st.text_input("Enter your question about the document:")
    
    if st.button("Ask") and question:
        with st.spinner("Getting answer..."):
            answer = ask_question(question)
            if answer:
                # Add to conversation history
                st.session_state.conversation.append({"question": question, "answer": answer})
    
    # Display conversation history
    if st.session_state.conversation:
        st.subheader("Conversation History")
        for i, exchange in enumerate(st.session_state.conversation):
            with st.container():
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.markdown(f"**Q{i+1}:**")
                with col2:
                    st.markdown(exchange["question"])
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.markdown(f"**A{i+1}:**")
                with col2:
                    st.markdown(exchange["answer"])
                
                st.divider()
else:
    st.info("Please upload and process a PDF document to start asking questions.")