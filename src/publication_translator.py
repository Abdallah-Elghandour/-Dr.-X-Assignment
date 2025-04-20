import os
import time
from typing import Dict, List, Union, Optional
from docx import Document
from publication_chunker import PublicationChunker
from llm_manager import LLMManager
from performance_metrics import PerformanceMetrics

class PublicationTranslator:
    """
    A class to translate publications from any language to English or Arabic.
    """
        
    def __init__(self):
        """
        Initialize the PublicationTranslator using Llama model.
        """
        self.llm_manager = LLMManager()
        self.supported_target_languages = ["english", "arabic"]
        self.performance_metrics = PerformanceMetrics()
        self.chunker = PublicationChunker()

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
                       
        chunks = self.chunker.chunk_text(file_path, max_tokens=1024)
        translated_chunks = []
        
        # Reset performance metrics for this document
        self.performance_metrics.reset()
        self.performance_metrics.start_tracking()

        for j, chunk in enumerate(chunks):
            
            print(f"Translating chunk {j+1}/{len(chunks)}...")
            translated_chunk = self._translate_text(chunk, target_language)
            translated_chunks.append(translated_chunk)

        # Stop tracking without assigning the return value
        self.performance_metrics.stop_tracking()
        self.performance_metrics.print_summary()
        
        translated_pages = " ".join(translated_chunks)
        
        return translated_pages
    
    def _translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text using Llama model while preserving structure and formatting.
        
        Returns:
            Translated text.
        """
        prompt = (
            f"Translate the following text to {target_language.capitalize()}\n"
            f"{text}\nTranslation:"
        )
        
        try:
            start_time = time.time()
            response = self.llm_manager.generate_response(
                prompt,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9
            )
            elapsed_time = time.time() - start_time
            token_count = self.chunker.get_token_count(response)
            
            self.performance_metrics.add_chunk_metrics(token_count, elapsed_time)

            print(f"Chunk metrics: {token_count / elapsed_time:.2f} tokens/second")

            return response

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
        