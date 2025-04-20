import tiktoken
from typing import List, Dict
from document_reader import DocumentReader

class PublicationChunker:
    """
    A class to break down publications into smaller, manageable chunks using tokenization.
    """
    
    def __init__(self):
        """Initialize the PublicationChunker with a tokenizer."""
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.document_reader = DocumentReader()
    
    def chunk_publication(self, file_path: str, max_tokens: int = 500) -> List[Dict[str, str]]:
        """
        Process a publication file and break it into chunks.
        
        Args:
            file_path: Path to the publication file
            
        Returns:
            List of dictionaries containing chunk information
        """
        document = self.document_reader.read_document(file_path)
        chunks = []
        chunk_counter = 1
        
        for page_num, page_text in enumerate(document['pages'], start=1):
            # Tokenize the page text
            tokens = self.tokenizer.encode(page_text)
            
            # Split tokens into chunks
            for i in range(0, len(tokens), max_tokens):
                chunk_tokens = tokens[i:i + max_tokens]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                
                chunks.append({
                    'source': document['source'],
                    'page_number': page_num,
                    'chunk_number': chunk_counter,
                    'text': chunk_text,
                })
                chunk_counter += 1
                
        return chunks

    def chunk_text(self, file_path: str, max_tokens: int = 1024)-> list[str]:
        """Split text into chunks based on token count."""
        document = self.document_reader.read_document(file_path)
        full_text = " ".join(document['pages'])
        tokens = self.tokenizer.encode(full_text)
        chunks = []

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks

    def get_token_count(self, text: str) -> int:
        """
        Count the number of tokens in the given text using the model's tokenizer.
        """
        tokens = self.tokenizer.encode(text)
        return len(tokens)