"""
Configuration file for Tender Analysis System
"""
import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

class Config:
    # API Configuration
    API_TITLE = "Tender Analysis & Management System"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "LLM-powered tender document analysis and management"
    DEBUG = os.getenv("DEBUG", True)
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, huggingface, ollama
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2048))
    
    # Vector Database Configuration
    VECTOR_DB = os.getenv("VECTOR_DB", "chroma")  # chroma, pinecone, faiss
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 384))
    
    # Pinecone Configuration (if using Pinecone)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "tender-analysis")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tender_analysis.db")
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    MONGODB_DB = os.getenv("MONGODB_DB", "tender_analysis")
    
    # File Upload Configuration
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
    PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/processed")
    EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", "./data/embeddings")
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
    
    # NLP Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/tender_analysis.log")
    
    # Processing Configuration
    ENABLE_OCR = os.getenv("ENABLE_OCR", True)
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
    TIMEOUT = int(os.getenv("TIMEOUT", 300))

config = Config()
