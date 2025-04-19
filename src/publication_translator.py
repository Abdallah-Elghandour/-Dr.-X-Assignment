import os
from openai import OpenAI
from typing import Dict, List, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from document_reader import DocumentReader
from docx import Document
from publication_chunker import PublicationChunker

class PublicationTranslator:
    """
    A class to translate publications from any language to English or Arabic.
    """
        
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the PublicationTranslator using Llama model.
        
        Args:
            model_name: Name or path of the Llama model to use.
        """
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map={"": 0},
        )
        # self.document_reader = DocumentReader()
        self.supported_target_languages = ["english", "arabic"]
    
    # def chunk_text_by_tokens(self, text, max_tokens=1024):
    #     """Split text into chunks based on token count."""
    #     tokens = self.tokenizer.encode(text)
    #     chunks = []
    #     for i in range(0, len(tokens), max_tokens):
    #         chunk_tokens = tokens[i:i + max_tokens]
    #         chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
    #         chunks.append(chunk_text)
    #     return chunks

    def translate_document(self, file_path: str, target_language: str = "english") -> str:
        """
        Translate a document to the target language while preserving structure.
        
        Args:
            file_path: Path to the document file
            target_language: Target language for translation (english or arabic)
            
        Returns:
            Dictionary with translated content and metadata
        """
        target_language = target_language.lower()
        if target_language not in self.supported_target_languages:
            raise ValueError(f"Unsupported target language: {target_language}. Supported languages: {', '.join(self.supported_target_languages)}")

        chunker = PublicationChunker()                            
        chunks = chunker.chunk_text(file_path, max_tokens=1024)
        translated_chunks = []

        for j, chunk in enumerate(chunks):
            print(f"Translating chunk {j+1}/{len(chunks)}...")
            translated_chunk = self._translate_text(chunk, target_language)
            translated_chunks.append(translated_chunk)
        # Optionally, you can split translated_chunks back into pages if needed
        translated_pages = " ".join(translated_chunks)
        
    

        
        return translated_pages
    
    def _translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text using Llama model while preserving structure and formatting.
        """
        prompt = (
            f"do not add extra content and do not add any Notes, Translate the following text to {target_language.capitalize()}\n"
            f"{text}\nTranslation:"
        )
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation.split("Translation:")[-1].strip()
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return f"[Translation Error: {str(e)}]"
    
    
    def save_translated_document(self, translated_content: str, output_path: str) -> str:
        """
        Save the translated content to a docx file.
        
        Args:
            translated_content: Dictionary with translated content
            output_path: Path to save the translated document
            
        Returns:
            Path to the saved file
        """
        # Ensure the output path has .docx extension
        if not output_path.lower().endswith('.docx'):
            output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = Document()  
        doc.add_paragraph(translated_content)
        doc.save(output_path)

        return output_path