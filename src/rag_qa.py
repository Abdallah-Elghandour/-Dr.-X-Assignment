from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from vector_db import VectorDB
from llm_manager import LLMManager
from performance_metrics import PerformanceMetrics
from publication_chunker import PublicationChunker
import time

class RAGQA:
    def __init__(self, vector_db: VectorDB):
        """
        Initialize RAG Q&A system with quantized Llama 3.1
        :param vector_db: VectorDB instance
        """
        self.vector_db = vector_db
        self.llm_manager = LLMManager()
        self.model = vector_db.model
        
        # Initialize conversation history
        self.conversation_history = []
        self.max_history = 2  # Keep last 2 messages
        
        # Initialize performance metrics
        self.performance_metrics = PerformanceMetrics()
        self.chunker = PublicationChunker()
    
    def _get_relevant_chunks(self, question: str, top_k: int = 2) -> List[Dict]:
        """Retrieve top-k most relevant chunks from the database using FAISS"""
        question_embedding = self.model.encode(question)
        question_embedding = np.array([question_embedding]).astype('float32')
        
        if self.vector_db.index is None or self.vector_db.index.ntotal == 0:
            return []
            
        # Search using FAISS
        distances, indices = self.vector_db.index.search(question_embedding, top_k)
        
        # Get corresponding metadata
        return [self.vector_db.metadata[i] for i in indices[0]]
    
    def answer_question(self, question: str) -> str:
        """Generate answer using quantized Llama 3.1"""
        relevant_chunks = self._get_relevant_chunks(question)
        
        if not relevant_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        context = "\n\n".join([
            f"Source: {chunk['source']}, Page {chunk['page_number']}, Chunk {chunk['chunk_number']}:\n{chunk['text']}"
            for chunk in relevant_chunks
        ])
        
        # Build messages with system prompt, conversation history, and current question
        messages = [
            {"role": "system", "content": "You are an AI assistant that answers questions based on the provided context."}
        ]
        
        # Add conversation history before the current question
        messages.extend(self.conversation_history)
        
        # Add the current question with context
        messages.append({"role": "user", "content": f"Context: {context} \nQuestion: {question}"})

        # Start tracking performance
        self.performance_metrics.reset()
        self.performance_metrics.start_tracking()

        # Use the LLM manager to generate the response
        answer = self.llm_manager.generate_chat_response(
                                                        messages,
                                                        max_new_tokens=1024,
                                                        temperature=0.7,
                                                        top_p=0.9,
                                                        do_sample=True)
        
        elapsed_time = time.time() - self.performance_metrics.metrics["start_time"]
        token_count = self.chunker.get_token_count(answer)
        self.performance_metrics.add_chunk_metrics(token_count, elapsed_time)
        self.performance_metrics.stop_tracking()
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Keep only the last 2 messages (pairs of user/assistant)
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        # Print performance summary
        self.performance_metrics.print_summary()
        
        return answer
