"""
PDF Loader Module
Handles loading and extracting text from PDF files using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
import os
from typing import Optional


class PDFLoader:
    """Class to handle PDF loading and text extraction."""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def load_pdf(self, file_path: str) -> str:
        """
        Load and extract text from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from all pages
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            Exception: If there's an error reading the PDF
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not file_path.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF")
        
        try:
            # Open the PDF document
            doc = fitz.open(file_path)
            
            # Extract text from all pages
            full_text = ""
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            doc.close()
            
            # Clean up the text
            full_text = self._clean_text(full_text)
            
            return full_text
            
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize the extracted text.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        import re
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_document_info(self, file_path: str) -> dict:
        """
        Get basic information about the PDF document.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            dict: Document information including page count, title, etc.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        try:
            doc = fitz.open(file_path)
            
            info = {
                'page_count': doc.page_count,
                'title': doc.metadata.get('title', 'Unknown'),
                'author': doc.metadata.get('author', 'Unknown'),
                'subject': doc.metadata.get('subject', 'Unknown'),
                'creator': doc.metadata.get('creator', 'Unknown'),
                'file_size': os.path.getsize(file_path)
            }
            
            doc.close()
            return info
            
        except Exception as e:
            raise Exception(f"Error getting document info: {str(e)}")


# Convenience function for direct use
def load_pdf_text(file_path: str) -> str:
    """
    Convenience function to load text from a PDF file.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    loader = PDFLoader()
    return loader.load_pdf(file_path)