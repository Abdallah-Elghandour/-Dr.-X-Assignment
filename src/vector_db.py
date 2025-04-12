from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import json
import os

class VectorDB:
    def __init__(self, db_path: str = "vector_db.json"):
        self.db_path = db_path
        self.db = self._load_db()
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
        
    def _load_db(self) -> List[Dict]:
        """Load existing database or create new one"""
        if os.path.exists(self.db_path):
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_db(self):
        """Save database to file"""
        with open(self.db_path, 'w') as f:
            json.dump(self.db, f)
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[np.ndarray]:
        """Generate embeddings for text chunks using Nomic model"""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts)
        return embeddings
    
    def add_to_db(self, chunks: List[Dict]):
        """Add chunks with their embeddings to the database"""
        embeddings = self.generate_embeddings(chunks)

        for chunk, embedding in zip(chunks, embeddings):
            self.db.append({
                'source': chunk['source'],
                'page_number': chunk['page_number'],
                'chunk_number': chunk['chunk_number'],
                'text': chunk['text'],
                'embedding': embedding.tolist()  # Convert numpy array to list
            })
        
        self._save_db()
    
    def get_db_size(self) -> int:
        """Get number of entries in the database"""
        return len(self.db)