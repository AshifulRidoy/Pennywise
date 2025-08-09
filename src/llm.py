"""
LLM Interface Module
Handles interaction with local LLM via Ollama.
"""

import subprocess
import json
import requests
from typing import Optional, Dict, Any


class OllamaLLM:
    """Class to interface with Ollama LLM."""
    
    def __init__(self, model_name: str = "mistral", host: str = "http://localhost:11434"):
        """
        Initialize Ollama LLM interface.
        
        Args:
            model_name (str): Name of the Ollama model to use
            host (str): Ollama server host
        """
        self.model_name = model_name
        self.host = host
        self.base_url = f"{host}/api"
        
    def is_ollama_running(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            bool: True if Ollama is running, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def is_model_available(self) -> bool:
        """
        Check if the specified model is available.
        
        Returns:
            bool: True if model is available, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'].split(':')[0] for model in models]
                return self.model_name in available_models
            return False
        except:
            return False
    
    def pull_model(self) -> bool:
        """
        Pull the model if it's not available.
        
        Returns:
            bool: True if model is now available, False otherwise
        """
        try:
            print(f"Pulling model {self.model_name}...")
            
            data = {"name": self.model_name}
            response = requests.post(
                f"{self.base_url}/pull", 
                json=data, 
                timeout=300
            )
            
            return response.status_code == 200
        except Exception as e:
            print(f"Error pulling model: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 200) -> str:
        """
        Generate response using Ollama API.
        
        Args:
            prompt (str): User query/prompt
            context (str): Context information
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Generated response
        """
        if not self.is_ollama_running():
            raise Exception("Ollama is not running. Please start Ollama first.")
        
        if not self.is_model_available():
            print(f"Model {self.model_name} not found. Attempting to pull...")
            if not self.pull_model():
                raise Exception(f"Failed to pull model {self.model_name}")
        
        # Construct the full prompt
        full_prompt = self._construct_prompt(prompt, context)
        
        try:
            data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            print(f"DEBUG: Sending prompt of length {len(full_prompt)} to Ollama")
            response = requests.post(
                f"{self.base_url}/generate",
                json=data,
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                raise Exception(f"API request failed: {response.status_code}")
                
        except requests.exceptions.Timeout:
            raise Exception("Request timed out. The model might be taking too long to respond.")
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
    
    def _construct_prompt(self, query: str, context: str) -> str:
        """
        Construct a prompt with context for financial document Q&A.
        
        Args:
            query (str): User query
            context (str): Retrieved context
            
        Returns:
            str: Formatted prompt
        """
        if context:
            prompt = f"""You are a helpful financial analyst assistant. Use the provided context from financial documents to answer the user's question accurately and concisely.

Context from financial documents:
{context}

Question: {query}

Please provide a clear, accurate answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        else:
            prompt = f"""You are a helpful financial analyst assistant. Please answer the following question about financial topics:

Question: {query}

Answer:"""
        
        return prompt
    
    def generate_with_cli(self, prompt: str, context: str = "") -> str:
        """
        Alternative method using Ollama CLI (fallback).
        
        Args:
            prompt (str): User query
            context (str): Context information
            
        Returns:
            str: Generated response
        """
        full_prompt = self._construct_prompt(prompt, context)
        
        try:
            # Use subprocess to call Ollama CLI
            result = subprocess.run(
                ["ollama", "run", self.model_name],
                input=full_prompt,
                text=True,
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise Exception(f"CLI command failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("CLI command timed out")
        except FileNotFoundError:
            raise Exception("Ollama CLI not found. Please install Ollama.")
        except Exception as e:
            raise Exception(f"Error with CLI: {str(e)}")


class MockLLM:
    """Mock LLM for testing when Ollama is not available."""
    
    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name
    
    def is_ollama_running(self) -> bool:
        return True
    
    def is_model_available(self) -> bool:
        return True
    
    def generate_response(self, prompt: str, context: str = "", max_tokens: int = 200) -> str:
        """Generate a mock response."""
        return f"""Based on the provided financial document context, I can see information related to your query about: "{prompt[:100]}..."

This is a mock response since Ollama is not available. In a real scenario, this would be a detailed analysis based on the retrieved context from the financial documents.

Key points that would typically be covered:
- Relevant financial metrics and data
- Contextual analysis from the documents
- Specific answers to your question

Please ensure Ollama is installed and running for actual LLM responses."""


# Convenience function
def create_llm(model_name: str = "mistral", use_mock: bool = False) -> OllamaLLM:
    """
    Create LLM interface.
    
    Args:
        model_name (str): Model name
        use_mock (bool): Whether to use mock LLM
        
    Returns:
        LLM interface
    """
    if use_mock:
        return MockLLM(model_name)
    else:
        return OllamaLLM(model_name)