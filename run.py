#!/usr/bin/env python3
"""
Sales Analysis Agent - Interactive Terminal Interface
"""

import sys
import os
from typing import Optional

from src.agent import SalesAnalysisAgent
from src.retrieval import SalesAnalysisToolEngine
from src.config import Config
from src.storage import DatabaseManager
from src.embeddings import EmbeddingManager

def create_agent(openai_api_key: str = None) -> SalesAnalysisAgent:
    """
    Factory function to create a fully configured SalesAnalysisAgent.
    
    Args:
        openai_api_key: OpenAI API key (if None, will use from config)
    
    Returns:
        Configured SalesAnalysisAgent instance
    """
    if openai_api_key is None:
        openai_api_key = Config.OPENAI_API_KEY
    
    if not openai_api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it directly.")
    
    # Initialize components
    db_manager = DatabaseManager(Config.DATABASE_PATH)
    embedding_manager = EmbeddingManager(
        openai_api_key=openai_api_key,
        index_path=Config.FAISS_INDEX_PATH,
        embedding_model=Config.EMBEDDING_MODEL
    )
    
    tool_engine = SalesAnalysisToolEngine(
        openai_api_key=openai_api_key,
        db_manager=db_manager,
        embedding_manager=embedding_manager,
        llm_model=Config.LLM_MODEL
    )
    
    agent = SalesAnalysisAgent(
        openai_api_key=openai_api_key,
        tool_engine=tool_engine,
        llm_model=Config.LLM_MODEL
    )
    
    return agent

def format_response(response: dict) -> str:
    """Format the agent's response for terminal display."""
    tool_used = response.get('tool_used', 'Unknown')
    result = response.get('result', {})
    
    output = f"\nTool Used: {tool_used}\n"
    output += "-" * 50 + "\n"
    
    if 'answer' in result:
        output += f"Answer:\n{result['answer']}\n"
    
    if 'sources' in result and result['sources']:
        output += f"\nSources:\n"
        for i, source in enumerate(result['sources'], 1):
            output += f"   {i}. {source}\n"
    
    if 'confidence' in result:
        confidence = result['confidence']
        if confidence is not None and confidence > 0:
            output += f"\nConfidence: {confidence:.2f}\n"
    
    if 'query_executed' in result and result['query_executed']:
        output += f"\nSQL Query: {result['query_executed']}\n"
    elif 'sql_query' in result:
        output += f"\nSQL Query: {result['sql_query']}\n"
    
    if 'data' in result and result['data']:
        output += f"\nResults:\n{result['data']}\n"
    
    return output


def main():
    """Main interactive loop."""
    try:
        # Initialize the agent
        print("Initializing Sales Analysis Agent...")
        agent = create_agent()
        
        # Interactive loop
        while True:
            try:
                # Get user input
                user_input = input("Your question: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nThank you for using Sales Analysis Agent.")
                    break
                elif not user_input:
                    continue
                
                # Process the query
                print("\nProcessing your query...")
                response = agent.process_query(user_input)
                
                # Display the response
                formatted_response = format_response(response)
                print(formatted_response)
                
            except Exception as e:
                print(f"\nError processing query: {str(e)}")
                
    except Exception as e:
        print(f"Failed to initialize agent: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
