"""
Intelligent agent that routes queries to appropriate tools (RAG, Summarize, SQL).
"""

from typing import Dict, List, Optional
from openai import OpenAI
from .retrieval import SalesAnalysisToolEngine
from .config import Config
from .prompts import PromptTemplates


class SalesAnalysisAgent:
    """
    Intelligent agent that analyzes user queries and routes them to the appropriate tool:
    - RAG: For content-based questions about call transcripts
    - Summarize: For call summary requests
    - SQL: For database queries and analytics
    """

    def __init__(self, 
                 openai_api_key: str,
                 tool_engine: SalesAnalysisToolEngine,
                 llm_model: str = "gpt-4o-mini"):
        """
        Initialize the agent with access to tools.
        
        Args:
            openai_api_key: OpenAI API key
            tool_engine: Initialized SalesAnalysisToolEngine instance
            llm_model: LLM model to use for intent classification
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.tool_engine = tool_engine
        self.llm_model = llm_model
        
    def process_query(self, user_query: str, **kwargs) -> Dict:
        """
        Process a user query by determining intent and routing to appropriate tool.
        
        Args:
            user_query: The user's question or request
            **kwargs: Additional arguments that might be needed for specific tools
                     (e.g., call_identifier for summarize, max_chunks for RAG)
        
        Returns:
            Dict containing:
            - tool_used: Which tool was selected
            - result: The result from the selected tool
            - intent_confidence: Confidence in intent classification
        """
        # Step 1: Classify the intent
        intent = self._classify_intent(user_query)
        
        # Step 2: Route to appropriate tool
        if intent == "RAG":
            result = self._handle_rag_query(user_query, **kwargs)
        elif intent == "SUMMARIZE":
            result = self._handle_summarize_query(user_query, **kwargs)
        elif intent == "SQL":
            result = self._handle_sql_query(user_query, **kwargs)
        elif intent == "INGEST":
            result = self._handle_ingest_query(user_query, **kwargs)
        else:
            # Fallback to RAG if intent is unclear
            intent = "RAG"
            result = self._handle_rag_query(user_query, **kwargs)
        
        return {
            "tool_used": intent,
            "result": result,
            "query": user_query
        }
    
    def _classify_intent(self, user_query: str) -> str:
        """
        Classify the user's intent using LLM.
        
        Returns:
            One of: "RAG", "SUMMARIZE", "SQL"
        """
        try:
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": PromptTemplates.INTENT_CLASSIFIER_SYSTEM},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            intent = response.choices[0].message.content.strip().upper()
            
            # Validate the response
            if intent in ["RAG", "SUMMARIZE", "SQL", "INGEST"]:
                return intent
            else:
                # Default to RAG if classification is unclear
                return "RAG"
                
        except Exception as e:
            print(f"Error in intent classification: {e}")
            # Default to RAG if there's an error
            return "RAG"
    
    def _handle_rag_query(self, user_query: str, **kwargs) -> Dict:
        """Handle RAG (content-based) queries."""
        max_chunks = kwargs.get('max_chunks', Config.MAX_CHUNKS)
        return self.tool_engine.retrieve_and_generate(user_query, max_chunks=max_chunks)
    
    def _handle_summarize_query(self, user_query: str, **kwargs) -> Dict:
        """Handle call summarization requests."""
        # Try to extract call identifier from the query or kwargs
        call_identifier = kwargs.get('call_identifier')
        
        if not call_identifier:
            # Try to extract call identifier from the query
            call_identifier = self._extract_file_name(user_query)
        
        if not call_identifier:
            return {
                'answer': "Please specify which call you'd like me to summarize. You can reference by filename (e.g., '1_demo_call.txt') or call ID.",
                'sources': [],
                'confidence': 0.0
            }
        
        return self.tool_engine.summarize_call(call_identifier)
    
    def _handle_sql_query(self, user_query: str, **kwargs) -> Dict:
        """Handle database/analytics queries."""
        return self.tool_engine.query_database(user_query)
    
    def _handle_ingest_query(self, user_query: str, **kwargs) -> Dict:
        """Handle file ingestion requests."""
        # Try to extract filename from the query or kwargs
        filename = kwargs.get('filename')
        
        if not filename:
            # Try to extract filename from the query using existing method
            filename = self._extract_file_name(user_query)
        
        if not filename:
            return {
                'answer': "Please specify which file you'd like me to ingest. You can reference by filename (e.g., '5_new_call.txt').",
                'sources': [],
                'confidence': 0.0
            }
        
        return self.tool_engine.ingest_file_tool(filename)
    
    def _extract_file_name(self, user_query: str) -> Optional[str]:
        """
        Extract .txt filename from user query.
        
        Examples:
        - "/data/text.txt" -> "text.txt"
        - "analyze 1_demo_call.txt" -> "1_demo_call.txt"
        - "path/to/some_file.txt" -> "some_file.txt"
        """
        import re
        import os
        
        # Look for any .txt file pattern and return the first match
        txt_pattern = r'[^\s/\\]+\.txt'
        txt_match = re.search(txt_pattern, user_query)
        
        if txt_match:
            return os.path.basename(txt_match.group())
        
        return None
    