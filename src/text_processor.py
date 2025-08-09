"""
Text Processing Module
Handles text chunking and preprocessing for the RAG system.
"""

import re
from typing import List, Tuple


class TextChunker:
    """Class to handle text chunking with overlap."""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size (int): Size of each chunk in characters
            overlap (int): Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Input text to chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean the text first
        text = self._preprocess_text(text)
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundary within the last 50 characters
                sentence_break = text.rfind('.', start + self.chunk_size - 50, end)
                if sentence_break != -1 and sentence_break > start:
                    end = sentence_break + 1
                else:
                    # Look for word boundary
                    word_break = text.rfind(' ', start + self.chunk_size - 20, end)
                    if word_break != -1 and word_break > start:
                        end = word_break
            
            # Extract the chunk
            chunk = text[start:end].strip()
            
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Calculate next start position with overlap
            if end >= len(text):
                break
                
            start = end - self.overlap
            
            # Ensure we don't get stuck in an infinite loop
            if not isinstance(start, int):
                print(f"DEBUG: start is {start} of type {type(start)}")
                raise TypeError(f"start should be int, got {type(start)}")
            if start <= 0 and len(chunks) > 1:
                break
        
        return chunks
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before chunking.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers if they exist
        text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
        
        # Clean up common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\%\$\#\@\&\*\+\=\<\>\|\\\/\~\`]', '', text)
        
        return text.strip()
    
    def get_chunk_info(self, chunks: List[str]) -> dict:
        """
        Get information about the chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            
        Returns:
            dict: Information about chunks
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_length': 0,
                'min_chunk_length': 0,
                'max_chunk_length': 0,
                'total_characters': 0
            }
        
        lengths = [len(chunk) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(lengths) / len(lengths),
            'min_chunk_length': min(lengths),
            'max_chunk_length': max(lengths),
            'total_characters': sum(lengths)
        }


class TextPreprocessor:
    """Class for advanced text preprocessing."""
    
    @staticmethod
    def clean_financial_text(text: str) -> str:
        """
        Clean financial document text with domain-specific rules.
        
        Args:
            text (str): Raw financial text
            
        Returns:
            str: Cleaned text
        """
        # Remove table artifacts and excessive formatting
        text = re.sub(r'\s+\|\s+', ' ', text)  # Remove table separators
        text = re.sub(r'_{3,}', '', text)      # Remove underlines
        text = re.sub(r'-{3,}', '', text)      # Remove dashes
        
        # Fix common financial document formatting issues
        text = re.sub(r'\$\s+(\d)', r'$\1', text)  # Fix separated dollar signs
        text = re.sub(r'(\d)\s+%', r'\1%', text)   # Fix separated percentages
        
        # Normalize section headers
        text = re.sub(r'\n\s*([A-Z][A-Z\s]{5,})\s*\n', r'\n\n\1\n\n', text)
        
        return text


# Convenience functions
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Convenience function to chunk text.
    
    Args:
        text (str): Input text
        chunk_size (int): Size of each chunk
        overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    chunker = TextChunker(chunk_size, overlap)
    return chunker.chunk_text(text)