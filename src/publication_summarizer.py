import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rouge import Rouge
from document_reader import DocumentReader
from typing import Dict, List, Union, Tuple


class PublicationSummarizer:
    """
    A class to summarize publications and evaluate
    the quality of summaries using ROUGE metrics.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the PublicationSummarizer with a local Llama model.
        
        Args:
            model_name: Name or path of the Llama model
        """
        self.model_name = model_name
        self.rouge = Rouge()
        self.reader = DocumentReader()
        
        # Load model and tokenizer if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            # Use the same 4-bit quantization config as in PublicationTranslator
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
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Continuing without local model. Will use prompt-based summarization.")
            self.model_loaded = False
    

    def _chunk_text(self, text: str, max_tokens: int = 6000) -> List[str]:
        """
        Split text into chunks based on token count.
        
        Args:
            text: The text to chunk
            max_tokens: Maximum number of tokens per chunk
            
        Returns:
            List of text chunks
        """
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        # Split tokens into chunks
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks

    def summarize_document(self, file_path: str, technique: str = "extractive") -> Dict:
        """
        Summarize a document using the specified technique.
        
        Args:
            file_path: Path to the document
            technique: Summarization technique ('extractive', 'abstractive', or 'hybrid')
            
        Returns:
            Dictionary containing the summary and metadata
        """
        # Read the document
        doc_content = self.reader.read_document(file_path)
        
        # Choose summarization technique
        if technique == "extractive":
            summary = self._extractive_summarization(doc_content)
        elif technique == "abstractive":
            summary = self._abstractive_summarization(doc_content)
        elif technique == "hybrid":
            summary = self._hybrid_summarization(doc_content)
        else:
            raise ValueError(f"Unsupported summarization technique: {technique}")
        
        return {
            "source": doc_content.get("source", os.path.basename(file_path)),
            "summary": summary,
            "technique": technique
        }
    
    def _extractive_summarization(self, doc_content: Dict) -> str:
        """
        Perform extractive summarization by selecting key sentences.
        
        Args:
            doc_content: Document content dictionary
            
        Returns:
            Extractive summary
        """
        pages = doc_content.get("pages", [])
        full_text = " ".join(pages)
        
        # Chunk the text if it's too long
        chunks = self._chunk_text(full_text)

        summaries = []
        
        for chunk in chunks:
        # Create prompt for extractive summarization
            prompt = f"""
            Extract the most important sentences from the following text to create a concise summary that retains the original wording.
            Only include sentences that are essential to understanding the main points.
            
            TEXT:
            {chunk}

            SUMMARY:
            """
        
            chunk_summary = self._generate_summary(prompt)
            summaries.append(chunk_summary)
        
        # Simply concatenate the summaries
        return "\n\n".join(summaries)
    
    def _abstractive_summarization(self, doc_content: Dict) -> str:
        """
        Perform abstractive summarization by generating new text.
        
        Args:
            doc_content: Document content dictionary
            
        Returns:
            Abstractive summary
        """
        pages = doc_content.get("pages", [])
        full_text = " ".join(pages)
        
        # Chunk the text if it's too long
        chunks = self._chunk_text(full_text)
        summaries = []
        
        for chunk in chunks:
            # Create prompt for abstractive summarization
            prompt = f"""
            Summarize the following text in your own words. 
            Rewrite it concisely while preserving the core message, important points, and overall tone. 
            Avoid copying sentences directly from the text.
            
            TEXT:
            {chunk}

            SUMMARY:
            """
            
            chunk_summary = self._generate_summary(prompt)
            summaries.append(chunk_summary)
        
        # Simply concatenate the summaries
        return "\n\n".join(summaries)
    
    def _hybrid_summarization(self, doc_content: Dict) -> str:
        """
        Perform hybrid summarization combining extractive and abstractive approaches.
        
        Args:
            doc_content: Document content dictionary
            
        Returns:
            Hybrid summary
        """
        pages = doc_content.get("pages", [])
        full_text = " ".join(pages)
        
        # Chunk the text if it's too long
        chunks = self._chunk_text(full_text)
        summaries = []
        
        for chunk in chunks:
            # Create prompt for hybrid summarization
            prompt = f"""
            Create a summary of the following text using a mix of extracted sentences and rephrased content. 
            Focus on maintaining accuracy while improving clarity and flow. Prioritize key points and reduce redundancy.
            
            TEXT:
            {chunk}
            
            SUMMARY:
            """
            
            chunk_summary = self._generate_summary(prompt)
            summaries.append(chunk_summary)
        
        # Simply concatenate the summaries
        return "\n\n".join(summaries)
    
    def _generate_summary(self, prompt: str) -> str:
        """
        Generate summary using the loaded model or a prompt-based approach.
        
        Args:
            prompt: The prompt for summarization
            
        Returns:
            Generated summary
        """
        if self.model_loaded:
            # Generate summary using the model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part (after the prompt)
            summary = summary[len(prompt):].strip()
            
            return summary
    
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
        Save the summary to a file.
        
        Args:
            summary: The summary text
            output_path: Path to save the summary
            
        Returns:
            Path to the saved summary file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return output_path
