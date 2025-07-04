"""Prompt templates for LLM interactions in the sales call analysis system."""


class PromptTemplates:
    """Collection of prompt templates for different analysis tasks."""
    
    # System messages
    SALES_ANALYST_SYSTEM = "You are an expert sales analyst who provides accurate, context-based answers about sales calls."
    CALL_SUMMARIZER_SYSTEM = "You are an expert sales analyst who summarizes sales calls clearly and concisely."
    SQL_GENERATOR_SYSTEM = "You are a SQL expert who generates safe, efficient SQLite queries based on database schema and user requirements."
    
    # Intent classification system prompt
    INTENT_CLASSIFIER_SYSTEM = """You are an intent classifier for a sales call analysis system. 
You have access to four tools:

1. RAG (Retrieval-Augmented Generation): For answering questions about call content, finding specific information in transcripts, analyzing conversations
2. SUMMARIZE: For generating summaries of specific calls or transcripts
3. SQL: For database queries, analytics, aggregations, counting records, finding patterns across multiple calls
4. INGEST: For ingesting/importing new call transcript files into the system

Analyze the user's query and respond with EXACTLY ONE of these four words: RAG, SUMMARIZE, SQL, or INGEST

Examples:
- "What objections were raised in the demo call?" → RAG
- "Summarize the pricing call 5_demo_call.txt" → SUMMARIZE
- "How many calls do we have in the database?" → SQL
- "Ingest the new call file 5_demo_call.txt" → INGEST
- "What concerns did customers raise about pricing?" → RAG
- "Give me a summary of transcript file 5_pricing_call.txt" → SUMMARIZE
- "Which calls had the most objections?" → SQL
- "Import 6_pricing_call.txt into the system" → INGEST
- "What did the customer say about our product features?" → RAG
- "Create a summary for the negotiation call 5_negotiation.txt" → SUMMARIZE
- "Show me all calls from this month" → SQL
- "Add the file new_call.txt to the database" → INGEST"""
    
    # Call summary prompt template
    CALL_SUMMARY_PROMPT = """
Analyze this sales call transcript and provide a comprehensive summary.

Call: {call_filename}
Participants: {participants}

Transcript excerpts:
{context}

Please provide a summary covering:
1. Call purpose and agenda
2. Key discussion points
3. Customer concerns or objections raised
4. Next steps or commitments made
5. Overall call sentiment and outcome

Format your response as a clear, structured summary.
"""
    
    # Query-based analysis prompt template
    QUERY_ANALYSIS_PROMPT = """
You are an AI assistant that helps sales teams analyze their call transcripts. Based on the provided context from sales call transcripts, answer the user's question accurately and concisely.

Context from sales calls:
{context}

User question: {query}

Instructions:
1. Answer based ONLY on the information provided in the context
2. If the context doesn't contain enough information, say so
3. Include specific quotes or timestamps when relevant
4. Be concise but thorough
5. If multiple calls are referenced, clearly distinguish between them

Answer:
"""
    
    # Commented out prompt for negative analysis (for future use)
    NEGATIVE_ANALYSIS_PROMPT = """
Analyze these sales call excerpts and identify negative comments, concerns, or objections{topic_phrase}.

Context:
{context}

Please:
1. List the specific negative comments or concerns raised
2. Identify who raised each concern
3. Note any responses or handling by the sales team
4. Group by theme if multiple related concerns exist

Focus only on genuine concerns, objections, or negative feedback.
"""

    # SQL Query Generation Prompt Template
    SQL_QUERY_PROMPT = """
You are a SQL expert. Based on the following database schema and user requirement, generate a SQLite query.

Database Schema:

Table: calls
- call_id (TEXT, PRIMARY KEY): Unique identifier for each call
- filename (TEXT): Name of the call transcript file
- participants (TEXT): JSON array of participant names
- created_at (TEXT): Timestamp when the call was created
- metadata (TEXT): JSON object with additional call metadata

Table: chunks
- chunk_id (TEXT, PRIMARY KEY): Unique identifier for each text chunk
- call_id (TEXT, FOREIGN KEY): Reference to the call this chunk belongs to
- content (TEXT): The actual text content of the chunk
- speaker (TEXT): Name of the speaker for this chunk
- timestamp (TEXT): Timestamp within the call when this was spoken
- chunk_index (INTEGER): Sequential index of the chunk within the call

Relationships:
- One call can have many chunks (1:N relationship)
- Use JOIN operations to combine data from both tables

User Requirement: {user_requirement}

Important Rules:
1. Generate ONLY SELECT queries - no INSERT, UPDATE, DELETE, or DDL statements
2. Use proper JOIN syntax when accessing data from multiple tables
3. Include appropriate WHERE clauses for filtering
4. Use LIMIT if the result set might be very large
5. For participants field, remember it's stored as JSON - use json_extract() if needed
6. Include column aliases for better readability
7. Return only the SQL query, no explanations

SQL Query:
"""

    @classmethod
    def get_call_summary_prompt(cls, call_filename: str, participants: list, context: str) -> str:
        """Generate a call summary prompt with the provided parameters."""
        return cls.CALL_SUMMARY_PROMPT.format(
            call_filename=call_filename,
            participants=', '.join(participants),
            context=context
        )
    
    @classmethod
    def get_query_analysis_prompt(cls, query: str, context: str) -> str:
        """Generate a query analysis prompt with the provided parameters."""
        return cls.QUERY_ANALYSIS_PROMPT.format(
            query=query,
            context=context
        )
    
    @classmethod
    def get_negative_analysis_prompt(cls, context: str, topic_phrase: str = "") -> str:
        """Generate a negative analysis prompt with the provided parameters."""
        return cls.NEGATIVE_ANALYSIS_PROMPT.format(
            context=context,
            topic_phrase=topic_phrase
        )
    
    @classmethod
    def get_sql_query_prompt(cls, user_requirement: str) -> str:
        """Generate a SQL query generation prompt with the provided parameters."""
        return cls.SQL_QUERY_PROMPT.format(
            user_requirement=user_requirement
        )
