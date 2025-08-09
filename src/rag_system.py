"""
RAG System Module
Combines all components into a complete RAG system.
"""

import os
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from .data_loader import PDFLoader
from .text_processor import TextChunker, TextPreprocessor
from .embedding import EmbeddingManager
from .llm import OllamaLLM, MockLLM


class FinancialRAGSystem:
    """Complete RAG system for financial document analysis."""
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "mistral",
        chunk_size: int = 500,
        overlap: int = 100,
        cache_dir: str = "cache/embeddings"
    ):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model (str): Name of the embedding model
            llm_model (str): Name of the LLM model
            chunk_size (int): Size of text chunks
            overlap (int): Overlap between chunks
            cache_dir (str): Cache directory for embeddings
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cache_dir = cache_dir
        
        # Initialize components
        self.pdf_loader = PDFLoader()
        self.text_chunker = TextChunker(chunk_size, overlap)
        self.text_processor = TextPreprocessor()
        self.embedding_manager = EmbeddingManager(embedding_model, cache_dir)
        
        # Initialize LLM (with fallback to mock)
        self.llm = OllamaLLM(llm_model)
        if not self.llm.is_ollama_running():
            print("⚠️  Ollama not detected. Using mock LLM for demonstration.")
            self.llm = MockLLM(llm_model)
        
        # Document tracking
        self.loaded_documents = {}
        self.current_index_id = None
    
    def load_document(self, file_path: str, document_id: Optional[str] = None) -> str:
        """
        Load a PDF document into the system.
        
        Args:
            file_path (str): Path to the PDF file
            document_id (str, optional): Custom document ID
            
        Returns:
            str: Document ID for the loaded document
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Generate document ID if not provided
        if document_id is None:
            file_hash = self._get_file_hash(file_path)
            document_id = f"doc_{file_hash[:8]}"
        
        try:
            print(f"Loading document: {file_path}")
            
            # Load and extract text
            raw_text = self.pdf_loader.load_pdf(file_path)
            if isinstance(raw_text, list):
                raw_text = "\n".join(raw_text)
            
            # Clean financial text
            cleaned_text = self.text_processor.clean_financial_text(raw_text)
            print(f"DEBUG: type(raw_text)={type(raw_text)}, type(cleaned_text)={type(cleaned_text)}")
            # Chunk the text
            chunks = self.text_chunker.chunk_text(cleaned_text)
            
            # Get document info
            doc_info = self.pdf_loader.get_document_info(file_path)
            chunk_info = self.text_chunker.get_chunk_info(chunks)
            
            # Store document information
            self.loaded_documents[document_id] = {
                'file_path': file_path,
                'raw_text': raw_text,
                'cleaned_text': cleaned_text,
                'chunks': chunks,
                'doc_info': doc_info,
                'chunk_info': chunk_info
            }
            
            print(f"✓ Document loaded: {document_id}")
            print(f"  - Pages: {doc_info['page_count']}")
            print(f"  - Chunks: {chunk_info['total_chunks']}")
            print(f"  - Avg chunk length: {chunk_info['avg_chunk_length']:.0f} characters")
            
            return document_id
            
        except Exception as e:
            raise Exception(f"Error loading document: {str(e)}")
    
    def build_index(self, document_ids: Optional[List[str]] = None) -> str:
        """
        Build vector index for specified documents.
        
        Args:
            document_ids (List[str], optional): List of document IDs to index
            
        Returns:
            str: Index ID
        """
        if not self.loaded_documents:
            raise ValueError("No documents loaded. Load documents first.")
        
        # Use all documents if none specified
        if document_ids is None:
            document_ids = list(self.loaded_documents.keys())
        
        # Validate document IDs
        for doc_id in document_ids:
            if doc_id not in self.loaded_documents:
                raise ValueError(f"Document ID not found: {doc_id}")
        
        try:
            print(f"Building index for {len(document_ids)} document(s)...")
            
            # Combine all chunks from specified documents
            all_chunks = []
            for doc_id in document_ids:
                chunks = self.loaded_documents[doc_id]['chunks']
                all_chunks.extend(chunks)
            
            if not all_chunks:
                raise ValueError("No chunks available for indexing")
            
            # Generate index ID
            index_hash = hashlib.md5('_'.join(sorted(document_ids)).encode()).hexdigest()
            index_id = f"index_{index_hash[:8]}"
            
            # Check if index already exists
            if self.embedding_manager.load_index(index_id):
                print(f"✓ Loaded existing index: {index_id}")
            else:
                # Build new index
                self.embedding_manager.build_index(all_chunks)
                self.embedding_manager.save_index(index_id)
                print(f"✓ Built and saved new index: {index_id}")
            
            self.current_index_id = index_id
            return index_id
            
        except Exception as e:
            raise Exception(f"Error building index: {str(e)}")
    
    def query(self, question: str, k: int = 5, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question (str): User question
            k (int): Number of chunks to retrieve
            max_tokens (int): Maximum tokens for LLM response
            
        Returns:
            Dict[str, Any]: Query results including answer and sources
        """
        if not self.current_index_id:
            raise ValueError("No index available. Build an index first.")
        
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            print(f"Processing query: {question[:100]}...")
            
            # Retrieve relevant chunks
            retrieved_chunks = self.embedding_manager.search(question, k)
            
            if not retrieved_chunks:
                return {
                    'question': question,
                    'answer': "I couldn't find any relevant information in the document to answer your question.",
                    'sources': [],
                    'confidence': 0.0
                }
            
            # Prepare context
            context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])
            
            # Generate answer using LLM
            answer = self.llm.generate_response(question, context, max_tokens)
            
            # Calculate average confidence (similarity score)
            avg_confidence = sum(score for _, score in retrieved_chunks) / len(retrieved_chunks)
            
            return {
                'question': question,
                'answer': answer,
                'sources': [
                    {
                        'text': chunk[:200] + '...' if len(chunk) > 200 else chunk,
                        'similarity': score,
                        'full_text': chunk
                    }
                    for chunk, score in retrieved_chunks
                ],
                'confidence': avg_confidence,
                'num_sources': len(retrieved_chunks)
            }
            
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and information.
        
        Returns:
            Dict[str, Any]: System status information
        """
        # Check LLM status
        llm_status = "Available"
        llm_type = "Ollama"
        if isinstance(self.llm, MockLLM):
            llm_status = "Mock (Ollama not available)"
            llm_type = "Mock"
        elif not self.llm.is_ollama_running():
            llm_status = "Not available"
        
        # Get index info
        index_info = self.embedding_manager.get_index_info()
        
        return {
            'documents_loaded': len(self.loaded_documents),
            'current_index': self.current_index_id,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'llm_status': llm_status,
            'llm_type': llm_type,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'index_info': index_info,
            'documents': {
                doc_id: {
                    'file_path': doc_data['file_path'],
                    'pages': doc_data['doc_info']['page_count'],
                    'chunks': doc_data['chunk_info']['total_chunks'],
                    'title': doc_data['doc_info'].get('title', 'Unknown')
                }
                for doc_id, doc_data in self.loaded_documents.items()
            }
        }
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash for a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def clear_cache(self):
        """Clear all cached embeddings and indexes."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
            print("✓ Cache cleared")
    
    def reset_system(self):
        """Reset the entire system."""
        self.loaded_documents.clear()
        self.current_index_id = None
        self.embedding_manager.index = None
        self.embedding_manager.chunks = None
        print("✓ System reset")


# Convenience function
def create_rag_system(**kwargs) -> FinancialRAGSystem:
    """
    Create a Financial RAG System with default or custom parameters.
    
    Returns:
        FinancialRAGSystem: Initialized RAG system
    """
    return FinancialRAGSystem(**kwargs)