import os
from typing import Dict, List, Union, Optional
from docx import Document
from publication_chunker import PublicationChunker
from llm_manager import LLMManager

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
            f"Translate the following text to {target_language.capitalize()}\n"
            f"{text}\nTranslation:"
        )
        try:
            return self.llm_manager.generate_response(prompt, 
                                                    max_new_tokens=1024,
                                                    temperature=0.3,
                                                    top_p=0.9)
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