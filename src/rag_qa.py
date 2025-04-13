from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from vector_db import VectorDB

class RAGQA:
    def __init__(self, vector_db: VectorDB, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize RAG Q&A system with quantized Llama 3
        :param vector_db: VectorDB instance
        :param model_name: Hugging Face model name for Llama 3
        """
        self.vector_db = vector_db
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Configure quantization using BitsAndBytesConfig with compatible data types
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # Load model with compatible data types
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},  
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        self.model = vector_db.model
        
        # Initialize conversation history
        self.conversation_history = []
        self.max_history = 2  # Keep last 2 messages
    
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
        """Generate answer using quantized Llama 3"""
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

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.llm.device)
        
        outputs = self.llm.generate(
            input_ids,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        answer = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Keep only the last 2 messages (pairs of user/assistant)
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        return answer
