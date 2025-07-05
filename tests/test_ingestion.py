"""Tests for the ingestion pipeline."""

import unittest
import tempfile
import os
import shutil
from unittest.mock import Mock, patch, MagicMock

from src.ingestion import IngestionPipeline
from src.storage import DatabaseManager, CallTranscript, TextChunk
from src.text_processor import TextProcessor
from src.embeddings import EmbeddingManager
from tests.test_config import (
    TestConfig, 
    create_test_transcript, 
    cleanup_test_files,
    SAMPLE_CALL_TRANSCRIPT,
    EMPTY_TRANSCRIPT,
    INVALID_FORMAT_TRANSCRIPT
)


class TestIngestionPipeline(unittest.TestCase):
    """Test cases for IngestionPipeline class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock dependencies
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_text_processor = Mock(spec=TextProcessor)
        self.mock_embedding_manager = Mock(spec=EmbeddingManager)
        
        # Create ingestion pipeline with mocks
        self.ingestion_pipeline = IngestionPipeline(
            db_manager=self.mock_db_manager,
            text_processor=self.mock_text_processor,
            embedding_manager=self.mock_embedding_manager
        )
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ingest_file_success(self):
        """Test successful file ingestion."""
        # Create test file
        test_file = create_test_transcript(SAMPLE_CALL_TRANSCRIPT, "test_call.txt")
        
        # Mock dependencies
        mock_chunks = [Mock(spec=TextChunk, chunk_id="chunk1")]
        mock_participants = ["AE (Jordan)", "Prospect (Priya)"]
        
        self.mock_text_processor.create_chunks.return_value = (mock_chunks, mock_participants)
        self.mock_db_manager.store_call.return_value = True
        self.mock_db_manager.store_chunk.return_value = True
        self.mock_embedding_manager.add_chunks.return_value = True
        
        # Mock database connection as context manager
        mock_conn = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_conn
        mock_context_manager.__exit__.return_value = None
        self.mock_db_manager.get_connection.return_value = mock_context_manager
        
        # Test ingestion
        result = self.ingestion_pipeline.ingest_file(test_file)
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['call_id'])
        self.assertEqual(result['filename'], "test_call.txt")
        self.assertEqual(result['chunks_created'], 1)
        
        # Verify method calls
        self.mock_text_processor.create_chunks.assert_called_once()
        self.mock_db_manager.store_call.assert_called_once()
        self.mock_embedding_manager.add_chunks.assert_called_once_with(mock_chunks)
        
        # Cleanup
        cleanup_test_files(test_file)
    
    def test_ingest_file_not_found(self):
        """Test ingestion of non-existent file."""
        # Mock database connection as context manager (even though file won't be found)
        mock_conn = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_conn
        mock_context_manager.__exit__.return_value = None
        self.mock_db_manager.get_connection.return_value = mock_context_manager
        
        result = self.ingestion_pipeline.ingest_file("/nonexistent/file.txt")
        
        self.assertFalse(result['success'])
        self.assertIn("File not found", result['error'])
        self.assertIsNone(result['call_id'])
    
    def test_ingest_file_storage_failure(self):
        """Test handling of database storage failure."""
        test_file = create_test_transcript(SAMPLE_CALL_TRANSCRIPT, "test_call.txt")
        
        # Mock text processor
        mock_chunks = [Mock(spec=TextChunk)]
        mock_participants = ["AE", "Prospect"]
        self.mock_text_processor.create_chunks.return_value = (mock_chunks, mock_participants)
        
        # Mock database failure
        self.mock_db_manager.store_call.return_value = False
        
        # Mock database connection as context manager
        mock_conn = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_conn
        mock_context_manager.__exit__.return_value = None
        self.mock_db_manager.get_connection.return_value = mock_context_manager
        
        result = self.ingestion_pipeline.ingest_file(test_file)
        
        self.assertFalse(result['success'])
        self.assertIn("Failed to store call in database", result['error'])
        
        cleanup_test_files(test_file)


class TestIngestionPipelineIntegration(unittest.TestCase):
    """Integration tests for IngestionPipeline with real dependencies."""
    
    def setUp(self):
        """Set up test environment with real dependencies."""
        self.test_db_path = TestConfig.get_test_db_path()
        
        # Create real instances (but avoid external API calls)
        self.db_manager = DatabaseManager(self.test_db_path)
        self.text_processor = TextProcessor(chunk_size=100)  # Smaller chunks for testing
        
        # Mock embedding manager to avoid OpenAI API calls
        self.mock_embedding_manager = Mock(spec=EmbeddingManager)
        self.mock_embedding_manager.add_chunks.return_value = True
        
        self.ingestion_pipeline = IngestionPipeline(
            db_manager=self.db_manager,
            text_processor=self.text_processor,
            embedding_manager=self.mock_embedding_manager
        )
    
    def tearDown(self):
        """Clean up after tests."""
        cleanup_test_files(self.test_db_path)
    
    def test_end_to_end_file_ingestion(self):
        """Test complete file ingestion with real text processing and database storage."""
        test_file = create_test_transcript(SAMPLE_CALL_TRANSCRIPT, "integration_test.txt")
        
        result = self.ingestion_pipeline.ingest_file(test_file)
        
        # Verify successful ingestion
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['call_id'])
        self.assertEqual(result['filename'], "integration_test.txt")
        self.assertGreater(result['chunks_created'], 0)
        
        # Verify data was stored in database
        call = self.db_manager.get_call_by_id(result['call_id'])
        self.assertIsNotNone(call)
        self.assertEqual(call.filename, "integration_test.txt")
        
        # Verify embedding manager was called
        self.mock_embedding_manager.add_chunks.assert_called_once()
        
        cleanup_test_files(test_file)


if __name__ == '__main__':
    unittest.main()
