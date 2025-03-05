from pydantic_settings import BaseSettings
import os
import pathlib
from typing import Optional


class Settings(BaseSettings):
    # API configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LangChain PDF RAG"

    # OpenAI API settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")


    # Directory Settings
    PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).parent.parent.parent
    DATA_DIR: pathlib.Path = PROJECT_ROOT / "data"
    CHROMA_DIR: pathlib.Path = DATA_DIR / "chroma_db"
    UPLOADS_DIR: pathlib.Path = PROJECT_ROOT / "uploads"

    # LLM Settings
    EMBEDDINGS_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o"
    TEMPERATURE: float = 0

    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 3

    class Config:
        env_file = ".env"

    def setup_directories(self):
        """Create neccessary directories if they don't exist"""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.CHROMA_DIR.mkdir(exist_ok=True)
        self.UPLOADS_DIR.mkdir(exist_ok=True)
        return True
    

