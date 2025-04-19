import os
import torch
from rouge import Rouge
from document_reader import DocumentReader
from docx import Document
from typing import Dict, List, Union, Tuple
from publication_chunker import PublicationChunker
from llm_manager import LLMManager


class PublicationSummarizer:
    """
    A class to summarize publications and evaluate
    the quality of summaries using ROUGE metrics.
    """
    
    def __init__(self):
        """
        Initialize the PublicationSummarizer with a local Llama model.
        """
        self.rouge = Rouge()
        
        try:
            self.llm_manager = LLMManager()
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Continuing without local model. Will use prompt-based summarization.")
            self.model_loaded = False
    

    def summarize_document(self, file_path: str, technique: str = "extractive") -> Dict:
        """
        Summarize a document using the specified technique.
        
        Args:
            file_path: Path to the document
            technique: Summarization technique ('extractive', 'abstractive', or 'hybrid')
            
        Returns:
            Dictionary containing the summary and metadata
        """
        techniques = {
            "extractive": self._get_extractive_prompt,
            "abstractive": self._get_abstractive_prompt,
            "hybrid": self._get_hybrid_prompt
        }
        
        if technique not in techniques:
            raise ValueError(f"Unsupported summarization technique: {technique}")
            
        chunker = PublicationChunker()
        chunks = chunker.chunk_text(file_path, max_tokens=6000)
        summaries = []
        
        for chunk in chunks:
            prompt = techniques[technique](chunk)
            chunk_summary = self._generate_summary(prompt)
            summaries.append(chunk_summary)
        
        return {"summary": "\n\n".join(summaries)}

    def _get_extractive_prompt(self, text: str) -> str:
        return f"""
        Extract the most important sentences from the following text to create a concise summary that retains the original wording.
        Only include sentences that are essential to understanding the main points.
        
        TEXT:
        {text}

        SUMMARY:
        """

    def _get_abstractive_prompt(self, text: str) -> str:
        return f"""
        Summarize the following text in your own words. 
        Rewrite it concisely while preserving the core message, important points, and overall tone. 
        Avoid copying sentences directly from the text.
        
        TEXT:
        {text}

        SUMMARY:
        """

    def _get_hybrid_prompt(self, text: str) -> str:
        return f"""
        Create a summary of the following text using a mix of extracted sentences and rephrased content. 
        Focus on maintaining accuracy while improving clarity and flow. Prioritize key points and reduce redundancy.
        
        TEXT:
        {text}
        
        SUMMARY:
        """

    def _generate_summary(self, prompt: str) -> str:
        """
        Generate summary using the loaded model or a prompt-based approach.
        
        Args:
            prompt: The prompt for summarization
            
        Returns:
            Generated summary
        """
        if not self.model_loaded:
            raise RuntimeError("No model loaded. Cannot generate summary.")
            
        return self.llm_manager.generate_response(prompt, 
                                                max_new_tokens=500,
                                                temperature=0.7,
                                                top_p=0.9)

    def evaluate_summary(self, reference_summary: str, generated_summary: str) -> Dict:
        """
        Evaluate the quality of a summary using ROUGE metrics.
        
        Args:
            reference_summary: The reference or ground truth summary
            generated_summary: The generated summary to evaluate
            
        Returns:
            Dictionary containing ROUGE scores
        """
        try:
            scores = self.rouge.get_scores(generated_summary, reference_summary)[0]
            return {
                "rouge-1": scores["rouge-1"],
                "rouge-2": scores["rouge-2"],
                "rouge-l": scores["rouge-l"]
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {str(e)}")
            return {
                "error": str(e)
            }
    
    
    def save_summary(self, summary: str, output_path: str) -> str:
        """
        Save the summary to a docx file.
        
        Args:
            summary: The summary text
            output_path: Path to save the summary
            
        Returns:
            Path to the saved summary file
        """
        if not output_path.lower().endswith('.docx'):
            output_path = os.path.splitext(output_path)[0]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        doc = Document()  
        doc.add_paragraph(summary)
        doc.save(output_path)
        
        return output_path
