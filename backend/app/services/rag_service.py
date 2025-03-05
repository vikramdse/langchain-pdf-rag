import os
import shutil
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from app.config.settings import Settings
import pathlib
import aiofiles
import asyncio


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.setup_directories()
        self._setup_rag_chain()
    
    def _setup_rag_chain(self):
        """Initialize the RAG chain components"""
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            api_key=self.settings.OPENAI_API_KEY,
            model=self.settings.EMBEDDINGS_MODEL
        )
        
        # Initialize vector store if it exists
        if any(self.settings.CHROMA_DIR.iterdir()):
            self.vectorstore = self._load_existing_db()
        else:
            self.vectorstore = None  # Will be created when a PDF is uploaded
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.settings.OPENAI_API_KEY,
            model=self.settings.LLM_MODEL,
            temperature=self.settings.TEMPERATURE
        )
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_template("""
        You are an expert assistant. Answer the question based only on the following context:

        {context}

        Question: {question}                                               
        """)
        
        # Setup retriever and chain if vectorstore exists
        if self.vectorstore:
            self._setup_retriever_and_chain()
    
    def _setup_retriever_and_chain(self):
        """Setup retriever and RAG chain"""
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.settings.TOP_K_RETRIEVAL}
        )
        
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
        )
    
    def _load_existing_db(self):
        """Load existing Chroma database"""
        return Chroma(
            persist_directory=str(self.settings.CHROMA_DIR),
            embedding_function=self.embeddings
        )
    
    def _create_or_update_db(self, pdf_path):
        """Create or update the vector database with the uploaded PDF"""
        try:
            # Load PDF
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(docs)
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=self.embeddings,
                persist_directory=str(self.settings.CHROMA_DIR)
            )
            
            # Setup retriever and chain
            self._setup_retriever_and_chain()
            
            return True
        except Exception as e:
            print(f"Error creating/updating database: {e}")
            return False
    
    async def process_pdf(self, file: UploadFile):
        """Process an uploaded PDF file"""
        try:
            # Create upload path
            pdf_path = self.settings.UPLOADS_DIR / file.filename
            
            # Write uploaded file to disk
            async with aiofiles.open(pdf_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            
            # Process in background to not block the API
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._create_or_update_db, pdf_path
            )
            
            return result
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return False
    
    async def answer_question(self, question: str):
        """Answer a question using the RAG chain"""
        if not self.vectorstore:
            raise ValueError("No documents have been uploaded yet")
        
        # Run the query
        response = await self.rag_chain.ainvoke(question)
        return response.content