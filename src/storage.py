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
    speakers: List[str] 
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
                    speakers TEXT,  -- JSON array of speakers
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
                (call_id, filename, participants, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                call.call_id,
                call.filename,
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
                (chunk_id, call_id, content, speakers, timestamp, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                chunk.chunk_id,
                chunk.call_id,
                chunk.content,
                json.dumps(chunk.speakers),
                chunk.timestamp,
                chunk.chunk_index
            ))
            return True
        except Exception as e:
            print(f"Error storing chunk: {e}")
            return False
        
    def get_call_count(self) -> int:
        """Get total number of calls in database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM calls')
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error getting call count: {e}")
            return 0
        
        
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[TextChunk]:
        """Retrieve chunks by list of IDs."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if not chunk_ids:
                    return []
                
                # Create placeholders for the IN clause
                placeholders = ','.join(['?' for _ in chunk_ids])
                query = f'SELECT * FROM chunks WHERE chunk_id IN ({placeholders})'
                
                cursor.execute(query, chunk_ids)
                rows = cursor.fetchall()
                
                chunks = []
                for row in rows:
                    chunk = TextChunk(
                        chunk_id=row[0],
                        call_id=row[1],
                        content=row[2],
                        speakers=json.loads(row[3]) if row[3] else [],
                        timestamp=row[4],
                        chunk_index=row[5]
                    )
                    chunks.append(chunk)
                
                return chunks
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

    def get_calls_by_ids(self, call_ids: List[str]) -> List[CallTranscript]:
        """Retrieve multiple calls by list of IDs."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if not call_ids:
                    return []
                
                # Create placeholders for the IN clause
                placeholders = ','.join(['?' for _ in call_ids])
                query = f'SELECT * FROM calls WHERE call_id IN ({placeholders})'
                
                cursor.execute(query, call_ids)
                rows = cursor.fetchall()
                
                calls = []
                for row in rows:
                    call = CallTranscript(
                        call_id=row[0],
                        filename=row[1],
                        participants=json.loads(row[2]),
                        created_at=row[3],
                        metadata=json.loads(row[4]) if row[4] else {}
                    )
                    calls.append(call)
                
                return calls
        except Exception as e:
            print(f"Error retrieving calls: {e}")
            return []

    def get_call_by_id(self, call_id: str) -> Optional[CallTranscript]:
        """Retrieve a specific call by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM calls WHERE call_id = ?', (call_id,))
                row = cursor.fetchone()
                
                if row:
                    return CallTranscript(
                        call_id=row[0],
                        filename=row[1],
                        participants=json.loads(row[2]),
                        created_at=row[3],
                        metadata=json.loads(row[4]) if row[4] else {}
                    )
                return None
        except Exception as e:
            print(f"Error retrieving call: {e}")
            return None
