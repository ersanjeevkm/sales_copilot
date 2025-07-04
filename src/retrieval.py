"""LLM-powered tool engine for sales analysis: RAG, summarization, SQL queries, and file ingestion."""

from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from .storage import DatabaseManager, TextChunk
from .embeddings import EmbeddingManager
from .prompts import PromptTemplates
from .config import Config
from .ingestion import IngestionPipeline
from .text_processor import TextProcessor


class SalesAnalysisToolEngine:
    """Handles LLM-powered tools for sales call analysis: RAG, summarization, SQL queries, and file ingestion."""
    
    def __init__(self, 
                 openai_api_key: str,
                 db_manager: DatabaseManager,
                 embedding_manager: EmbeddingManager,
                 llm_model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.llm_model = llm_model
        
        # Initialize ingestion pipeline for file ingestion
        self.text_processor = TextProcessor()
        self.ingestion_pipeline = IngestionPipeline(
            db_manager=db_manager,
            text_processor=self.text_processor,
            embedding_manager=embedding_manager
        )
    
    def retrieve_and_generate(self, query: str, max_chunks: int = 20) -> Dict:
        """
        Main RAG pipeline: retrieve relevant chunks and generate response.
        
        Returns:
            Dict with 'answer', 'sources', and 'confidence'
        """
        # 1. Retrieve relevant chunks
        search_results = self.embedding_manager.search(query, k=max_chunks)
        
        if not search_results:
            return {
                'answer': "I couldn't find any relevant information in the call transcripts to answer your question.",
                'sources': [],
                'confidence': 0.0
            }
        
        # 2. Get full chunk content from database
        chunk_ids = [result['chunk_id'] for result in search_results]
        chunks = self.db_manager.get_chunks_by_ids(chunk_ids)
        
        # Create a mapping of chunk_id to chunk for easy lookup
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        relevant_chunks = []
        for result in search_results:
            chunk = chunk_map.get(result['chunk_id'])
            if chunk:
                relevant_chunks.append({
                    'chunk': chunk,
                    'similarity_score': result['similarity_score'],
                    'metadata': result
                })
        
        # 3. Build context and generate response
        context = self._build_context(relevant_chunks)
        user_prompt = PromptTemplates.get_query_analysis_prompt(query=query, context=context)
        answer = self._generate_response(
            system_prompt=PromptTemplates.SALES_ANALYST_SYSTEM,
            user_prompt=user_prompt,
            temperature=0.2
        )
        
        # 4. Format sources
        sources = self._format_sources(relevant_chunks)
        
        # 5. Calculate confidence (average similarity score)
        confidence = sum(r['similarity_score'] for r in relevant_chunks) / len(relevant_chunks) if relevant_chunks else 0.0
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence
        }
    
    def summarize_call(self, call_identifier: str) -> Dict:
        """
        Generate a summary for a specific call.
        
        Args:
            call_identifier: Either a call_id or filename (e.g., "1_demo_call.txt")
        """
        import os
        
        call = None
        filename = None
        
        # First try to get call by ID
        call = self.db_manager.get_call_by_id(call_identifier)
        
        # If still not found, assume it's a filename and try to find the file
        if not call:
            filename = call_identifier
            # Check if the file exists in the data directory
            file_path = Config.get_file_path(filename)
            
            if not os.path.exists(file_path):
                return {
                    'answer': f"Call with identifier '{call_identifier}' not found in database or as file.",
                    'sources': [],
                    'confidence': 0.0
                }
        else:
            filename = call.filename
        
        # Read the full file content directly
        try:
            file_path = Config.get_file_path(filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Extract participants if we have call info, otherwise derive from content
            participants = call.participants if call else []
            
            prompt = PromptTemplates.get_call_summary_prompt(
                call_filename=filename,
                participants=participants,
                context=file_content
            )
            
            summary = self._generate_response(
                system_prompt=PromptTemplates.CALL_SUMMARIZER_SYSTEM,
                user_prompt=prompt,
                temperature=0.3
            )
            
            # Format sources - use filename as source
            sources = [f"Source: {filename} (Full transcript)"]
            
            return {
                'answer': summary,
                'sources': sources,
                'confidence': None
            }
            
        except FileNotFoundError:
            return {
                'answer': f"File '{filename}' not found in data directory.",
                'sources': [],
                'confidence': 0.0
            }
        except Exception as e:
            return {
                'answer': f"Error generating summary: {e}",
                'sources': [],
                'confidence': 0.0
            }
        
    def query_database(self, user_requirement: str) -> Dict:
        """
        Generate and execute SQLite queries based on user requirements.
        Only accepts SELECT queries for security reasons.
        
        Args:
            user_requirement: Natural language description of what the user wants to query
            
        Returns:
            Dict with 'answer', 'sources', and 'query_executed'
        """
        try:
            # Generate SQL query using LLM
            sql_prompt = PromptTemplates.get_sql_query_prompt(user_requirement)
            
            generated_sql = self._generate_response(
                system_prompt=PromptTemplates.SQL_GENERATOR_SYSTEM,
                user_prompt=sql_prompt,
                temperature=0.1
            ).strip()
            
            # Security check: ensure only SELECT queries
            if not generated_sql.upper().strip().startswith('SELECT'):
                return {
                    'answer': "Error: Only SELECT queries are allowed for security reasons.",
                    'sources': [],
                    'query_executed': None
                }
            
            # Additional security checks for dangerous keywords
            dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'CREATE', 'TRUNCATE']
            sql_upper = generated_sql.upper()
            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    return {
                        'answer': f"Error: Query contains potentially dangerous keyword '{keyword}'. Only SELECT queries are allowed.",
                        'sources': [],
                        'query_executed': None
                    }
            
            # Execute the query
            results = self.db_manager.execute_query(generated_sql)
            
            if not results:
                return {
                    'answer': "Query executed successfully but returned no results.",
                    'sources': [f"Database query: {generated_sql}"],
                    'query_executed': generated_sql
                }
            
            # Format results for display
            formatted_results = self._format_query_results(results, generated_sql)
            
            return {
                'answer': formatted_results,
                'sources': [f"Database query: {generated_sql}"],
                'query_executed': generated_sql
            }
            
        except Exception as e:
            return {
                'answer': f"Error executing database query: {str(e)}",
                'sources': [],
                'query_executed': generated_sql if 'generated_sql' in locals() else None
            }
        
    def ingest_file_tool(self, filename: str) -> Dict:
        """
        Tool to ingest a single call transcript file into the system.
        
        Args:
            filename: Name of the file to ingest (e.g., "5_new_call.txt")
                     The file should be in the data directory configured in Config
            
        Returns:
            Dict with 'answer', 'sources', and ingestion details
        """
        try:
            # Get the full file path using Config
            file_path = Config.get_file_path(filename)
            
            # Check if file exists
            import os
            if not os.path.exists(file_path):
                return {
                    'answer': f"Error: File '{filename}' not found in data directory ({Config.DATA_DIR}).",
                    'sources': [],
                    'ingestion_result': None
                }
            
            # Call the ingestion pipeline
            result = self.ingestion_pipeline.ingest_file(file_path)
            
            if result['success']:
                answer = f"Successfully ingested file '{filename}'.\n"
                answer += f"Call ID: {result['call_id']}\n"
                answer += f"Participants: {', '.join(result.get('participants', []))}\n"
                answer += f"Chunks created: {result.get('chunks_created', 0)}\n"
                answer += "The file has been processed and is now available for querying."
                
                sources = [f"Ingested file: {filename}"]
                
                return {
                    'answer': answer,
                    'sources': sources,
                    'ingestion_result': result
                }
            else:
                return {
                    'answer': f"Failed to ingest file '{filename}': {result.get('error', 'Unknown error')}",
                    'sources': [],
                    'ingestion_result': result
                }
                
        except Exception as e:
            return {
                'answer': f"Error during file ingestion: {str(e)}",
                'sources': [],
                'ingestion_result': None
            }
    
    def _build_context(self, relevant_chunks: List[Dict]) -> str:
        """Build context string from relevant chunks, grouped by call_id and sorted by chunk_index."""
        if not relevant_chunks:
            return ""
        
        # Group chunks by call_id
        chunks_by_call = {}
        for item in relevant_chunks:
            chunk = item['chunk']
            call_id = chunk.call_id
            if call_id not in chunks_by_call:
                chunks_by_call[call_id] = []
            chunks_by_call[call_id].append(item)
        
        # Sort chunks within each call by chunk_index
        for call_id in chunks_by_call:
            chunks_by_call[call_id].sort(key=lambda x: x['chunk'].chunk_index)
        
        # Build context parts for each call
        call_contexts = []
        for call_id, call_chunks in chunks_by_call.items():

            call_parts = [f"Call Transcript ID: {call_id}"]
            
            for item in call_chunks:
                chunk = item['chunk']
                score = item['similarity_score']
                
                call_parts.append(
                    f"[{chunk.timestamp}] {chunk.speaker} [Relevance: {score:.2f}]:\n{chunk.content}\n"
                )
            
            call_contexts.append('\n'.join(call_parts))
        
        return '\n\n'.join(call_contexts)
    
    def _generate_response(self, 
                         system_prompt: str, 
                         user_prompt: str, 
                         temperature: float = 0.2) -> str:
        """Generate LLM response based on system and user prompts."""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _format_sources(self, relevant_chunks: List[Dict]) -> List[str]:
        """Format source information for display."""
        sources = []
        
        # Get unique call IDs to batch fetch calls
        unique_call_ids = list(set(item['chunk'].call_id for item in relevant_chunks))
        calls = self.db_manager.get_calls_by_ids(unique_call_ids)
        
        # Create a mapping of call_id to call for quick lookup
        call_map = {call.call_id: call for call in calls}
        
        for item in relevant_chunks:
            chunk = item['chunk']
            score = item['similarity_score']
            
            # Get call filename from the batch-fetched calls
            call = call_map.get(chunk.call_id)
            call_name = call.filename if call else chunk.call_id
            
            source_info = f"{call_name} [{chunk.timestamp}] (Relevance: {score:.2f})"
            sources.append(source_info)
        
        return sources
    
    
    def _format_query_results(self, results: List[Tuple], query: str) -> str:
        """Format query results for LLM processing."""
        if not results:
            return "No results found."
        
        max_results = Config.MAX_QUERY_RESULTS
        
        # Simple format for LLM consumption - just the raw data
        formatted_output = []
        for row in results[:max_results]:
            if len(row) == 1:
                formatted_output.append(str(row[0]))
            else:
                row_str = ", ".join(str(item) for item in row)
                formatted_output.append(row_str)
        
        return "\n".join(formatted_output)
