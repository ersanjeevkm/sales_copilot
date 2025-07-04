"""Setup and batch ingestion utilities for the call transcript processing system."""

import os

from src.config import Config
from src.storage import DatabaseManager
from src.text_processor import TextProcessor
from src.embeddings import EmbeddingManager
from src.ingestion import IngestionPipeline


class BatchIngestor:
    """Utility class for batch ingestion of call transcript files."""
    
    def __init__(self, data_directory: str = None):
        """
        Initialize the batch ingestor.
        
        Args:
            data_directory: Path to directory containing .txt files. 
                          Defaults to data directory from config.
        """
        if data_directory is None:
            self.data_directory = Config.get_data_directory()
        else:
            self.data_directory = data_directory
        
        # Validate required configuration
        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == 'your_openai_api_key_here':
            raise ValueError(
                "OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file. "
                "You can find the .env file at the project root and replace 'your_openai_api_key_here' "
                "with your actual OpenAI API key."
            )
            
        # Initialize components
        self.db_manager = DatabaseManager(Config.DATABASE_PATH)
        self.text_processor = TextProcessor()
        self.embedding_manager = EmbeddingManager(
            openai_api_key=Config.OPENAI_API_KEY,
            index_path=Config.FAISS_INDEX_PATH
        )
        self.ingestion_pipeline = IngestionPipeline(
            self.db_manager,
            self.text_processor, 
            self.embedding_manager
        )
    
    def read_and_batch_ingest_txt_files(self, file_pattern: str = "*.txt") -> dict:
        """
        Read all .txt files from the data directory and perform batch ingestion.
        
        Args:
            file_pattern: Pattern to match files (default: "*.txt")
            
        Returns:
            Dict containing ingestion results with statistics and per-file details
        """
        print(f"Starting batch ingestion from directory: {self.data_directory}")
        print(f"Looking for files matching pattern: {file_pattern}")
        
        if not os.path.exists(self.data_directory):
            return {
                'success': False,
                'error': f"Data directory not found: {self.data_directory}",
                'results': []
            }
        
        # Initialize database and vector index
        try:
            self.db_manager.init_database()
            print("Database and vector index initialized successfully")
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to initialize database/vector index: {str(e)}",
                'results': []
            }
        
        # Perform batch ingestion
        result = self.ingestion_pipeline.ingest_directory(
            self.data_directory, 
            file_pattern
        )
        
        # Print summary
        if result['success']:
            print(f"\nBatch ingestion completed successfully!")
            print(f"Total files processed: {len(result.get('results', []))}")
            for file_result in result.get('results', []):
                if file_result.get('success'):
                    print(f"{file_result.get('filename', 'Unknown')}: {file_result.get('chunks_created', 0)} chunks")
                else:
                    print(f"{file_result.get('filename', 'Unknown')}: {file_result.get('error', 'Unknown error')}")
        else:
            print(f"\nBatch ingestion failed!")
            if 'error' in result:
                print(f"Error: {result['error']}")
            if 'results' in result:
                for file_result in result.get('results', []):
                    if not file_result.get('success'):
                        print(f"{file_result.get('filename', 'Unknown')}: {file_result.get('error', 'Unknown error')}")
        
        return result
        
def main():
    """Main function to demonstrate batch ingestion."""
    print("Starting batch ingestion of call transcript files...")
    
    try:
        # Create batch ingestor instance
        ingestor = BatchIngestor()
        
        # Perform batch ingestion of all .txt files in data directory
        result = ingestor.read_and_batch_ingest_txt_files()
        
        if result['success']:
            print(f"\nAll files processed successfully!")
            return 0
        else:
            print(f"\nSome files failed to process.")
            if 'error' in result:
                print(f"Main error: {result['error']}")
            return 1
            
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())