"""Configuration and constants for the Sales Copilot application."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration and constants."""
    
    # API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Database Configuration
    DATABASE_PATH = os.getenv('DATABASE_PATH', './data/sales_copilot.db')
    FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH', './data/faiss_index')
    
    # Directory Paths
    DATA_DIRECTORY = os.getenv('DATA_DIRECTORY', './data')
    SRC_DIRECTORY = os.getenv('SRC_DIRECTORY', './src')
    
    # Model Configuration
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '256'))
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    
    # Query Results Configuration
    MAX_QUERY_RESULTS = int(os.getenv('MAX_QUERY_RESULTS', '50'))
    MAX_CHUNKS = int(os.getenv('MAX_CHUNKS', '20'))
    
    @classmethod
    def get_data_directory(cls) -> str:
        """Get the absolute path to the data directory."""
        if os.path.isabs(cls.DATA_DIRECTORY):
            return cls.DATA_DIRECTORY
        
        # Get project root directory (one level up from src)
        project_root = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(project_root, cls.DATA_DIRECTORY.lstrip('./'))
    
    @classmethod
    def get_file_path(cls, filename: str) -> str:
        """Get full path to a file in the data directory."""
        return os.path.join(cls.get_data_directory(), filename)
