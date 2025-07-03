"""Database models and storage layer for the Sales Copilot."""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import os


@dataclass
class CallTranscript:
    """Represents a sales call transcript."""
    call_id: str
    filename: str
    content: str
    participants: List[str]
    created_at: str
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TextChunk:
    """Represents a chunk of text from a call transcript."""
    chunk_id: str
    call_id: str
    content: str
    speaker: str
    timestamp: str
    chunk_index: int
    embedding: Optional[List[float]] = None


class DatabaseManager:
    """Manages SQLite database operations for call transcripts and chunks."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ensure_directory_exists()
        self.init_database()
    
    def ensure_directory_exists(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def get_connection(self) -> sqlite3.Connection:
        """Return a database connection."""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create calls table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calls (
                    call_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    participants TEXT NOT NULL,  -- JSON array
                    created_at TEXT NOT NULL,
                    metadata TEXT  -- JSON object
                )
            ''')
            
            # Create chunks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    call_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    speaker TEXT,
                    timestamp TEXT,
                    chunk_index INTEGER NOT NULL,
                    FOREIGN KEY (call_id) REFERENCES calls (call_id)
                )
            ''')
            
            # Create index for faster retrieval
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_call_id ON chunks(call_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_index ON chunks(chunk_index)')
            
            conn.commit()
    
    def execute_query(self, query: str) -> List[Tuple]:
        """Execute a SQL query and return the results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                # For SELECT queries, return results
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                else:
                    # For INSERT, UPDATE, DELETE queries, commit and return empty list
                    conn.commit()
                    return []
        except Exception as e:
            print(f"Error executing query: {e}")
            return []
    
    def store_call(self, call: CallTranscript, cursor: sqlite3.Cursor) -> bool:
        """Store a call transcript in the database."""
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO calls 
                (call_id, filename, content, participants, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                call.call_id,
                call.filename,
                call.content,
                json.dumps(call.participants),
                call.created_at,
                json.dumps(call.metadata)
            ))
            return True
        except Exception as e:
            print(f"Error storing call: {e}")
            return False
    
    def store_chunk(self, chunk: TextChunk, cursor: sqlite3.Cursor) -> bool:
        """Store a text chunk in the database."""
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO chunks 
                (chunk_id, call_id, content, speaker, timestamp, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                chunk.chunk_id,
                chunk.call_id,
                chunk.content,
                chunk.speaker,
                chunk.timestamp,
                chunk.chunk_index
            ))
            return True
        except Exception as e:
            print(f"Error storing chunk: {e}")
            return False
