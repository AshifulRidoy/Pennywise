"""
Embedding Module
Handles text embedding and vector index creation using SentenceTransformers and FAISS.
"""

import numpy as np
import faiss
import pickle
import os
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Class to manage embeddings and vector search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', cache_dir: str = 'cache/embeddings'):
        """
        Initialize the embedding manager.
        
        Args:
            model_name (str): Name of the SentenceTransformer model
            cache_dir (str): Directory to cache embeddings and index
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.index = None
        self.chunks = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("✓ Embedding model loaded successfully")
        except Exception as e:
            raise Exception(f"Error loading embedding model: {str(e)}")
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of text chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not chunks:
            return np.array([])
        
        try:
            print(f"Creating embeddings for {len(chunks)} chunks...")
            
            # Create embeddings
            embeddings = self.model.encode(
                chunks, 
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            print("✓ Embeddings created successfully")
            return embeddings
            
        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")
    
    def build_index(self, chunks: List[str], embeddings: Optional[np.ndarray] = None) -> faiss.Index:
        """
        Build FAISS index from chunks and their embeddings.
        
        Args:
            chunks (List[str]): List of text chunks
            embeddings (Optional[np.ndarray]): Precomputed embeddings
            
        Returns:
            faiss.Index: Built FAISS index
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        # Create embeddings if not provided
        if embeddings is None:
            embeddings = self.create_embeddings(chunks)
        
        # Store chunks for retrieval
        self.chunks = chunks
        
        try:
            print("Building FAISS index...")
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to index
            self.index.add(embeddings.astype('float32'))
            
            print(f"✓ FAISS index built with {self.index.ntotal} vectors")
            return self.index
            
        except Exception as e:
            raise Exception(f"Error building FAISS index: {str(e)}")
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar chunks given a query.
        
        Args:
            query (str): Search query
            k (int): Number of top results to return
            
        Returns:
            List[Tuple[str, float]]: List of (chunk, similarity_score) tuples
        """
        if self.index is None or self.chunks is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if not query.strip():
            return []
        
        try:
            # Embed the query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            
            # Search the index
            distances, indices = self.index.search(
                query_embedding.astype('float32'), k
            )
            
            # Return results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunks):  # Ensure valid index
                    # Convert L2 distance to similarity score (lower distance = higher similarity)
                    similarity = 1 / (1 + distance)
                    results.append((self.chunks[idx], similarity))
            
            return results
            
        except Exception as e:
            raise Exception(f"Error searching index: {str(e)}")
    
    def save_index(self, filename: str):
        """
        Save the FAISS index and chunks to disk.
        
        Args:
            filename (str): Base filename (without extension)
        """
        if self.index is None or self.chunks is None:
            raise ValueError("No index to save. Build index first.")
        
        try:
            index_path = os.path.join(self.cache_dir, f"{filename}.index")
            chunks_path = os.path.join(self.cache_dir, f"{filename}.chunks")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save chunks
            with open(chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            print(f"✓ Index saved: {index_path}")
            print(f"✓ Chunks saved: {chunks_path}")
            
        except Exception as e:
            raise Exception(f"Error saving index: {str(e)}")
    
    def load_index(self, filename: str) -> bool:
        """
        Load FAISS index and chunks from disk.
        
        Args:
            filename (str): Base filename (without extension)
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            index_path = os.path.join(self.cache_dir, f"{filename}.index")
            chunks_path = os.path.join(self.cache_dir, f"{filename}.chunks")
            
            if not os.path.exists(index_path) or not os.path.exists(chunks_path):
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load chunks
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            print(f"✓ Index loaded: {index_path}")
            print(f"✓ Chunks loaded: {chunks_path}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False
    
    def get_index_info(self) -> dict:
        """
        Get information about the current index.
        
        Returns:
            dict: Index information
        """
        if self.index is None or self.chunks is None:
            return {
                'index_exists': False,
                'total_vectors': 0,
                'dimension': 0,
                'total_chunks': 0
            }
        
        return {
            'index_exists': True,
            'total_vectors': self.index.ntotal,
            'dimension': self.index.d,
            'total_chunks': len(self.chunks),
            'model_name': self.model_name
        }


# Convenience functions
def create_vector_index(chunks: List[str], model_name: str = 'all-MiniLM-L6-v2') -> EmbeddingManager:
    """
    Convenience function to create a vector index from chunks.
    
    Args:
        chunks (List[str]): List of text chunks
        model_name (str): Name of the embedding model
        
    Returns:
        EmbeddingManager: Initialized embedding manager with built index
    """
    manager = EmbeddingManager(model_name)
    manager.build_index(chunks)
    return manager