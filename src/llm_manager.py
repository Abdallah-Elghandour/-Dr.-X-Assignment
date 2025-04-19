import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional, Union

class LLMManager:
    """
    A reusable manager for LLM operations across the application.
    Handles model loading, tokenization, and generation with consistent settings.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the LLM Manager with a quantized Llama model
        
        Args:
            model_name: Hugging Face model name/path
        """
        self.model_name = model_name
        
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Configure quantization using BitsAndBytesConfig with compatible data types
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": 0},  
            quantization_config=self.quantization_config,
            low_cpu_mem_usage=True,
        )
        
    def generate_response(self, 
                          prompt: str, 
                          max_new_tokens: int = 1024, 
                          temperature: float = 0.7, 
                          top_p: float = 0.9, 
                          do_sample: bool = True) -> str:
        """
        Generate a response from the model based on the prompt
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Return only the generated part, not including the prompt
        return response[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
    
    def generate_chat_response(self, 
                              messages: List[Dict[str, str]], 
                              max_new_tokens: int = 1024,
                              temperature: float = 0.7, 
                              top_p: float = 0.9, 
                              do_sample: bool = True) -> str:
        """
        Generate a response using the chat template format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Generated text response
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample
        )
        
        return self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)