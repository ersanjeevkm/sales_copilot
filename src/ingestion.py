"""Ingestion pipeline for processing and storing call transcripts."""

import os
import uuid
from datetime import datetime
from typing import Optional
from .storage import DatabaseManager, CallTranscript
from .text_processor import TextProcessor
from .embeddings import EmbeddingManager


class IngestionPipeline:
    """Handles the complete ingestion pipeline for call transcripts."""
    
    def __init__(self, 
                 db_manager: DatabaseManager,
                 text_processor: TextProcessor,
                 embedding_manager: EmbeddingManager):
        self.db_manager = db_manager
        self.text_processor = text_processor
        self.embedding_manager = embedding_manager
    
    def _ingest_file_with_cursor(self, file_path: str, cursor) -> dict:
        """
        Internal method to ingest a single file with an existing database cursor.
        
        Returns:
            Dict with success status, call_id, and any error messages
        """
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': f"File not found: {file_path}",
                'call_id': None
            }
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                return {
                    'success': False,
                    'error': "File is empty",
                    'call_id': None
                }
            
            # Generate unique call ID
            call_id = str(uuid.uuid4())
            filename = os.path.basename(file_path)
            
            # Process into chunks, Extract participants
            chunks, participants = self.text_processor.create_chunks(call_id, content)

            # Create call transcript object
            call = CallTranscript(
                call_id=call_id,
                filename=filename,
                participants=participants,
                created_at=datetime.now().isoformat(),
                metadata={
                    'source_path': file_path,
                    'file_size': len(content),
                    'ingestion_timestamp': datetime.now().isoformat()
                }
            )
            
            # Store call in database
            if not self.db_manager.store_call(call, cursor):
                return {
                    'success': False,
                    'error': "Failed to store call in database",
                    'call_id': call_id
                }
            
            if not chunks:
                return {
                    'success': False,
                    'error': "Failed to create text chunks",
                    'call_id': call_id
                }
            
            # Store chunks in database
            for chunk in chunks:
                if not self.db_manager.store_chunk(chunk, cursor):
                    print(f"Warning: Failed to store chunk {chunk.chunk_id}")
            
            # Add chunks to vector index
            if not self.embedding_manager.add_chunks(chunks):
                print("Warning: Failed to add chunks to vector index")
            
            return {
                'success': True,
                'call_id': call_id,
                'filename': filename,
                'participants': participants,
                'chunks_created': len(chunks),
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Error processing file: {str(e)}",
                'call_id': None
            }

    def ingest_file(self, file_path: str) -> dict:
        """
        Ingest a single call transcript file.
        
        Returns:
            Dict with success status, call_id, and any error messages
        """
        # Use database connection for single file operation
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            result = self._ingest_file_with_cursor(file_path, cursor)
            if result['success']:
                conn.commit()
            return result
    
    def ingest_directory(self, directory_path: str, file_pattern: str = "*.txt") -> dict:
        """
        Ingest all files matching pattern in a directory.
        
        Returns:
            Dict with overall results and per-file details
        """
        if not os.path.exists(directory_path):
            return {
                'success': False,
                'error': f"Directory not found: {directory_path}",
                'results': []
            }
        
        import glob
        
        # Find matching files
        pattern = os.path.join(directory_path, file_pattern)
        files = glob.glob(pattern)
        
        if not files:
            return {
                'success': False,
                'error': f"No files found matching pattern: {file_pattern}",
                'results': []
            }
        
        results = []
        successful = 0
        failed = 0
        
        # Open database connection once for all files
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            for file_path in files:
                print(f"Processing: {os.path.basename(file_path)}")
                result = self._ingest_file_with_cursor(file_path, cursor)
                results.append(result)
                
                if result['success']:
                    successful += 1
                    print(f"Successfully ingested with {result.get('chunks_created', 0)} chunks")
                else:
                    failed += 1
                    print(f"Failed: {result.get('error', 'Unknown error')}")
            
            # Commit all changes at once
            conn.commit()
        
        return {
            'success': failed == 0,
            'total_files': len(files),
            'successful': successful,
            'failed': failed,
            'results': results
        }
        
    def get_ingestion_stats(self) -> dict:
        """Get statistics about the ingested data."""
        total_calls = self.db_manager.get_call_count()
        index_stats = self.embedding_manager.get_index_stats()
        
        return {
            'total_calls': total_calls,
            'total_chunks': index_stats['total_chunks'],
            'index_size_mb': index_stats['index_size_mb'],
            'vector_dimension': index_stats['dimension']
        }
