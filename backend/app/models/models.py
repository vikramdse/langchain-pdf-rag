from pydantic import BaseModel, Field
from typing import List, Optional

class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the document")

class QuestionResponse(BaseModel):
    answer: str = Field(..., description="The answer to the question")

class PDFProcessingResponse(BaseModel):
    success: bool
    message: str