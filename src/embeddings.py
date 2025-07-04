"""Vector embedding and retrieval system using OpenAI embeddings and FAISS."""

import os
import numpy as np
import faiss
import pickle
from typing import List, Tuple, Optional
import openai
from openai import OpenAI
from .storage import TextChunk


class EmbeddingManager:
    """Manages vector embeddings and FAISS index for semantic search."""
    
    def __init__(self, 
                 openai_api_key: str,
                 index_path: str,
                 embedding_model: str = "text-embedding-3-small",
                 dimension: int = 1536):
        self.client = OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = f"{index_path}.metadata"
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunk_ids = []  # Store chunk IDs mapped to index positions
        
        self.ensure_directory_exists()
        self.load_index()
    
    def ensure_directory_exists(self):
        """Ensure the index directory exists."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using OpenAI API."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text.replace("\n", " ")
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * self.dimension
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in a batch."""
        try:
            # OpenAI API supports batch requests
            cleaned_texts = [text.replace("\n", " ") for text in texts]
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=cleaned_texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error getting batch embeddings: {e}")
            return [[0.0] * self.dimension] * len(texts)
    
    def add_chunks(self, chunks: List[TextChunk]) -> bool:
        """Add text chunks to the vector index."""
        if not chunks:
            return True
            
        try:
            # Get embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = self.get_embeddings_batch(texts)
            
            # Normalize embeddings for cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store chunk IDs
            for chunk in chunks:
                self.chunk_ids.append(chunk.chunk_id)
            
            # Save index
            self.save_index()
            return True
            
        except Exception as e:
            print(f"Error adding chunks to index: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[dict]:
        """
        Search for similar chunks using semantic similarity.
        
        Returns:
            List of dictonaries with chunk IDs and similarity scores.
        """
        if self.index.ntotal == 0:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.get_embedding(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search FAISS index
            scores, indices = self.index.search(query_vector, min(k, self.index.ntotal))
            
            # Return results with metadata
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunk_ids):
                    chunk_id = self.chunk_ids[idx]
                    results.append({
                        'chunk_id': chunk_id,
                        'similarity_score': float(score)
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching index: {e}")
            return []
    
    def save_index(self):
        """Save FAISS index and chunk IDs to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.chunk_ids, f)
        except Exception as e:
            print(f"Error saving index: {e}")
    
    def load_index(self):
        """Load FAISS index and chunk IDs from disk."""
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.chunk_ids = pickle.load(f)
            
        except Exception as e:
            print(f"Error loading index: {e}")
            # Reset to empty index on error
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunk_ids = []
    
    def get_index_stats(self) -> dict:
        """Get statistics about the current index."""
        return {
            'total_chunks': self.index.ntotal,
            'dimension': self.dimension,
            'index_size_mb': os.path.getsize(self.index_path) / (1024 * 1024) if os.path.exists(self.index_path) else 0
        }
