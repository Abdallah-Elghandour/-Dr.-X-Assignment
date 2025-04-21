import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from typing import List, Dict, Optional, Union

class LLMManager:
    """
    A reusable manager for LLM operations across the application.
    Handles model loading, tokenization, and generation with consistent settings.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", model_type: str = "causal"):
        """
        Initialize the LLM Manager with a quantized model
        
        Args:
            model_name: Hugging Face model name/path
            model_type: Type of model - "causal" for text generation or "seq2seq" for translation
        """
        self.model_name = model_name
        self.model_type = model_type
        
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
        # Set pad_token_id to eos_token_id if not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load the appropriate model type
        if model_type == "causal":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": 0},  
                quantization_config=self.quantization_config,
                low_cpu_mem_usage=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        elif model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map={"": 0},
                torch_dtype=torch.float16
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def generate_response(self, 
                          prompt: str, 
                          max_new_tokens: int = 1024, 
                          temperature: float = 0.7, 
                          top_p: float = 0.9, 
                          do_sample: bool = True) -> str:
        """
        Generate a response from the model based on the prompt
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
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
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
    def generate_translation(self, 
                            text: str, 
                            target_lang_code: str,
                            max_length: int = 512) -> str:
        """
        Translate text to the target language using the NLLB model.
        
        Args:
            text: Text to translate
            target_lang_code: Target language code (e.g., 'fra_Latn', 'ara_Arab')
            max_length: Maximum length of the generated translation
            
        Returns:
            Translated text
        """
        if self.model_type != "seq2seq":
            raise ValueError("Translation requires a seq2seq model like NLLB. Initialize with model_type='seq2seq'")
            
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate translation with forced BOS token for target language
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(target_lang_code),
            max_length=max_length,
        )
        
        # Decode the translation
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        return translated_text