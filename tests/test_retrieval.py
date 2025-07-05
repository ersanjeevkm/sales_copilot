"""Simplified tests for the retrieval and analysis engine."""

import unittest
from unittest.mock import Mock, MagicMock

from src.retrieval import SalesAnalysisToolEngine
from src.storage import DatabaseManager, TextChunk
from src.embeddings import EmbeddingManager
from tests.test_config import cleanup_test_files


class TestSalesAnalysisToolEngine(unittest.TestCase):
    """Test cases for SalesAnalysisToolEngine class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create mock dependencies
        self.mock_db_manager = Mock(spec=DatabaseManager)
        self.mock_embedding_manager = Mock(spec=EmbeddingManager)
        
        # Create tool engine with mocks
        self.tool_engine = SalesAnalysisToolEngine(
            openai_api_key="test_key",
            db_manager=self.mock_db_manager,
            embedding_manager=self.mock_embedding_manager,
            llm_model="gpt-4o-mini"
        )
        
        # Mock OpenAI client
        self.tool_engine.client = Mock()
    
    def test_retrieve_and_generate_success(self):
        """Test successful RAG pipeline execution."""
        query = "What are the main pain points mentioned?"
        
        # Mock search results
        mock_search_results = [
            {'chunk_id': 'chunk1', 'similarity_score': 0.85}
        ]
        self.mock_embedding_manager.search.return_value = mock_search_results
        
        # Mock chunks from database
        mock_chunks = [
            Mock(spec=TextChunk, 
                 chunk_id='chunk1', 
                 content='Slow onboarding is a major issue.',
                 call_id='call1',
                 chunk_index=1,
                 speakers=['AE', 'Prospect'],
                 timestamp='00:01:00')
        ]
        self.mock_db_manager.get_chunks_by_ids.return_value = mock_chunks
        
        # Mock calls from database
        from src.storage import CallTranscript
        mock_calls = [
            CallTranscript(
                call_id='call1',
                filename='test_call.txt',
                participants=['AE', 'Prospect'],
                created_at='2024-01-01T10:00:00Z'
            )
        ]
        self.mock_db_manager.get_calls_by_ids.return_value = mock_calls
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "The main pain point is slow onboarding."
        self.tool_engine.client.chat.completions.create.return_value = mock_response
        
        result = self.tool_engine.retrieve_and_generate(query)
        
        # Assertions
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertIsInstance(result['sources'], list)
        
        # Verify method calls
        self.mock_embedding_manager.search.assert_called_once()
        self.mock_db_manager.get_chunks_by_ids.assert_called_once()
    
    def test_retrieve_and_generate_no_results(self):
        """Test RAG pipeline when no relevant chunks are found."""
        query = "What is the weather like?"
        
        # Mock empty search results
        self.mock_embedding_manager.search.return_value = []
        
        result = self.tool_engine.retrieve_and_generate(query)
        
        self.assertIn("couldn't find", result['answer'].lower())
        self.assertEqual(result['sources'], [])
    
    def test_retrieve_and_generate_with_context(self):
        """Test retrieval and context building functionality."""
        query = "pricing discussion"
        
        # Mock search results
        mock_search_results = [
            {'chunk_id': 'chunk1', 'similarity_score': 0.90}
        ]
        self.mock_embedding_manager.search.return_value = mock_search_results
        
        # Mock chunks
        mock_chunks = [
            Mock(spec=TextChunk, 
                 chunk_id='chunk1',
                 content='We discussed the pricing model.',
                 call_id='call1',
                 chunk_index=1,
                 speakers=['AE', 'Prospect'],
                 timestamp='00:02:00')
        ]
        self.mock_db_manager.get_chunks_by_ids.return_value = mock_chunks
        
        # Mock calls from database
        from src.storage import CallTranscript
        mock_calls = [
            CallTranscript(
                call_id='call1',
                filename='test_call.txt',
                participants=['AE', 'Prospect'],
                created_at='2024-01-01T10:00:00Z'
            )
        ]
        self.mock_db_manager.get_calls_by_ids.return_value = mock_calls
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "The pricing discussion covered various models."
        self.tool_engine.client.chat.completions.create.return_value = mock_response
        
        result = self.tool_engine.retrieve_and_generate(query, max_chunks=1)
        
        # Assertions
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertEqual(len(result['sources']), 1)
        
        # Verify method calls
        self.mock_embedding_manager.search.assert_called_once_with(query, k=1)
        self.mock_db_manager.get_chunks_by_ids.assert_called_once_with(['chunk1'])


class TestSalesAnalysisToolEngineIntegration(unittest.TestCase):
    """Integration tests with real database but mocked embeddings."""
    
    def setUp(self):
        """Set up test environment."""
        from tests.test_config import TestConfig
        
        self.test_db_path = TestConfig.get_test_db_path()
        
        # Create real database manager
        self.db_manager = DatabaseManager(self.test_db_path)
        
        # Mock embedding manager to avoid API calls
        self.mock_embedding_manager = Mock(spec=EmbeddingManager)
        
        self.tool_engine = SalesAnalysisToolEngine(
            openai_api_key="test_key",
            db_manager=self.db_manager,
            embedding_manager=self.mock_embedding_manager
        )
        
        # Mock OpenAI client
        self.tool_engine.client = Mock()
    
    def tearDown(self):
        """Clean up after tests."""
        cleanup_test_files(self.test_db_path)
    
    def test_end_to_end_query_processing(self):
        """Test complete query processing with real database."""
        # First, add some test data to the database
        from src.storage import CallTranscript, TextChunk
        import uuid
        import json
        
        call_id = str(uuid.uuid4())
        call = CallTranscript(
            call_id=call_id,
            filename="test_call.txt",
            participants=["AE", "Prospect"],
            created_at="2024-01-01T10:00:00Z"
        )
        
        # Store call in database
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO calls (call_id, filename, participants, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (call.call_id, call.filename, json.dumps(call.participants), call.created_at, json.dumps(call.metadata)))
            
            # Add a chunk
            cursor.execute("""
                INSERT INTO chunks (chunk_id, call_id, content, chunk_index, speakers, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("chunk1", call_id, "Pricing discussion content", 1, json.dumps(["AE", "Prospect"]), "00:01:00"))
            
            conn.commit()
        
        # Mock embedding search to return our chunk
        self.mock_embedding_manager.search.return_value = [
            {'chunk_id': 'chunk1', 'similarity_score': 0.85}
        ]
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is about pricing discussions."
        self.tool_engine.client.chat.completions.create.return_value = mock_response
        
        # Test the query
        result = self.tool_engine.retrieve_and_generate("What about pricing?")
        
        # Assertions
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertEqual(len(result['sources']), 1)
        # Sources are formatted as strings with filename and timestamp
        self.assertIn('test_call.txt', result['sources'][0])
        self.assertIn('00:01:00', result['sources'][0])


if __name__ == '__main__':
    unittest.main()
