import os
from openai import OpenAI
from typing import Dict, List, Union, Optional
from document_reader import DocumentReader

class PublicationTranslator:
    """
    A class to translate publications from any language to English or Arabic
    using DeepSeek's translation capabilities while preserving document structure.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com"):
        """
        Initialize the PublicationTranslator.
        
        Args:
            api_key: DeepSeek API key. If None, will try to use the DEEPSEEK_API_KEY environment variable.
            base_url: DeepSeek API base URL.
        """
        if api_key:
            self.api_key = api_key
        elif os.environ.get("DEEPSEEK_API_KEY"):
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
        else:
            raise ValueError("DeepSeek API key must be provided either as an argument or as an environment variable.")
        
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.document_reader = DocumentReader()
        self.supported_target_languages = ["english", "arabic"]
    
    def translate_document(self, file_path: str, target_language: str = "english") -> Dict[str, Union[str, List[str]]]:
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
        
        # Read the document
        document_content = self.document_reader.read_document(file_path)
        
        # Translate each page
        translated_pages = []
        for i, page in enumerate(document_content['pages']):
            print(f"Translating page {i+1}/{len(document_content['pages'])}...")
            translated_page = self._translate_text(page, target_language)
            translated_pages.append(translated_page)
        
        # Create result dictionary
        result = {
            'source': document_content['source'],
            'pages': translated_pages,
            'target_language': target_language
        }
        
        return result
    
    def _translate_text(self, text: str, target_language: str) -> str:
        """
        Translate text using DeepSeek API while preserving structure and formatting.
        """
        system_prompt = f"""
        You are a professional translator. Translate the following text to {target_language}.
        Important guidelines:
        1. Preserve all original formatting, including paragraphs, bullet points, and tables
        2. Maintain the original document structure
        3. Keep technical terms accurate
        4. For tables, preserve the tabular format using the same delimiters
        5. Do not add or remove information
        6. Translate all content, including headers and footnotes
        """
        
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return f"[Translation Error: {str(e)}]"
    
    
    def save_translated_document(self, translated_content: Dict[str, Union[str, List[str]]], output_path: str) -> str:
        """
        Save the translated content to a text file.
        
        Args:
            translated_content: Dictionary with translated content
            output_path: Path to save the translated document
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write(f"Source: {translated_content['source']}\n")
            file.write(f"Target Language: {translated_content['target_language']}\n")
            file.write("\n" + "="*50 + "\n\n")
            
            for i, page in enumerate(translated_content['pages']):
                file.write(f"Page {i+1}:\n\n")
                file.write(page)
                file.write("\n\n" + "-"*50 + "\n\n")
        
        return output_path