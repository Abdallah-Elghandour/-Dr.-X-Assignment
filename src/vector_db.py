import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import time
import pickle
from performance_metrics import PerformanceMetrics
from publication_chunker import PublicationChunker

class VectorDB:
    def __init__(self, db_path: str = "vector_db.faiss"):
        self.db_path = db_path
        self.metadata_path = db_path.replace('.faiss', '.meta')
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
        self.index = None
        self.metadata = []
        self.performance_metrics = PerformanceMetrics()
        self._load_db()
        self.chunker = PublicationChunker()
        
    def _load_db(self):
        """Load existing FAISS index and metadata or create new ones"""
        if os.path.exists(self.db_path):
            self.index = faiss.read_index(self.db_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            # Initialize empty index - dimension will be set when first embedding is added
            self.index = None
            self.metadata = []
    
    def _save_db(self):
        """Save FAISS index and metadata to files"""
        if self.index is not None:
            faiss.write_index(self.index, self.db_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Generate embeddings for text chunks using Nomic model"""
        texts = [chunk['text'] for chunk in chunks]
        
        # Start tracking performance
        self.performance_metrics.start_tracking()
  
        # Generate embeddings
        embeddings = self.model.encode(texts)
        processing_time = time.time() - self.performance_metrics.metrics["start_time"]

        total_text = " ".join(texts)
        token_count = self.chunker.get_token_count(total_text)
        # Add metrics
        self.performance_metrics.add_chunk_metrics(token_count, processing_time)
        self.performance_metrics.stop_tracking()
        self.performance_metrics.print_summary()

        return embeddings
    
    def add_to_db(self, chunks: List[Dict]):
        """Add chunks with their embeddings to the database"""
        embeddings = self.generate_embeddings(chunks)
        
        if self.index is None:
            # Initialize index with correct dimension
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings)
        
        # Store metadata
        for chunk in chunks:
            self.metadata.append({
                'source': chunk['source'],
                'page_number': chunk['page_number'],
                'chunk_number': chunk['chunk_number'],
                'text': chunk['text']
            })
        
        self._save_db()
    
    def get_db_size(self) -> int:
        """Get number of entries in the database"""
        return self.index.ntotal if self.index is not None else 0
