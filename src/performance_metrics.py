from typing import Dict
import time

class PerformanceMetrics:
    """
    A class to track and calculate performance metrics for NLP operations.
    """
    
    def __init__(self):
        """
        Initialize the performance metrics tracker.
        """
        self.reset()
    
    def reset(self):
        """
        Reset all performance metrics to initial values.
        """
        self.metrics = {
            "tokens_per_second": 0,
            "total_tokens": 0,
            "total_time": 0,
            "start_time": None
        }
    
    def start_tracking(self):
        """
        Start tracking the total processing time.
        """
        self.metrics["start_time"] = time.time()
    
    def add_chunk_metrics(self, tokens: int, processing_time: float):
        """
        Add metrics for a processed chunk.
        
        Args:
            tokens: Number of tokens in the chunk
            processing_time: Time taken to process the chunk in seconds
        """
        self.metrics["total_tokens"] += tokens
        self.metrics["total_time"] += processing_time
        
        # Update tokens per second
        if self.metrics["total_time"] > 0:
            self.metrics["tokens_per_second"] = self.metrics["total_tokens"] / self.metrics["total_time"]
    
    def stop_tracking(self):
        """
        Stop tracking and calculate final metrics.
        """
        if self.metrics["start_time"] is not None:
            total_elapsed = time.time() - self.metrics["start_time"]
            self.metrics["start_time"] = None
    
    def get_metrics(self) -> Dict:
        """
        Get the current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {k: v for k, v in self.metrics.items() if k != "start_time"}
    
    def print_summary(self):
        """
        Print a summary of the performance metrics.
        """
        print(f"\nPerformance metrics summary:")
        print(f"- Total tokens processed: {self.metrics['total_tokens']}")
        print(f"- Total processing time: {self.metrics['total_time']:.2f} seconds")
        print(f"- Tokens per second: {self.metrics['tokens_per_second']:.2f}\n\n")
