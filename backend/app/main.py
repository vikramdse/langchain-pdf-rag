from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.models.models import QuestionRequest, QuestionResponse
from app.services.rag_service import RAGService
from app.config.settings import Settings



app = FastAPI(
    title="LangChain PDF RAG API",
    description="API for PDF document retrieval and question answering",
    version="1.0.0"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Settings dependency
def get_settings():
    return Settings()

# Service dependency
def get_rag_service(settings: Settings = Depends(get_settings)):
    return RAGService(settings)


@app.post("/upload", status_code=201)
async def upload_pdf(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service)
):
    """Upload a PDF file and process it for RAG"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    success = await rag_service.process_pdf(file)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to process PDF")
    
    return {"message": "PDF uploaded and processed successfully"}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Ask a question about the uploaded PDF documents"""
    try:
        answer = await rag_service.answer_question(request.question)
        return QuestionResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)