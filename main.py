"""
Financial RAG Assistant - Command Line Interface
"""

import os
import sys
from src.rag_system import FinancialRAGSystem


def main():
    """Main function for command line usage."""
    print("🏦 Financial RAG Assistant")
    print("=" * 50)
    
    # Initialize RAG system
    rag = FinancialRAGSystem(
        embedding_model="all-MiniLM-L6-v2",
        llm_model="mistral",
        chunk_size=500,
        overlap=100
    )
    
    # Check for sample document
    sample_doc = "data/raw/example_report.pdf"
    if not os.path.exists(sample_doc):
        print(f"⚠️  Sample document not found: {sample_doc}")
        print("Please place a PDF financial document in the data/raw/ directory")
        
        # Create directory if it doesn't exist
        os.makedirs("data/raw", exist_ok=True)
        return
    
    try:
        # Load document
        doc_id = rag.load_document(sample_doc)
        
        # Build index
        index_id = rag.build_index([doc_id])
        
        # Interactive query loop
        print("\n💬 Ask questions about the financial document (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                # Process query
                result = rag.query(question, k=3)
                
                print(f"\n💡 Answer:")
                print("-" * 20)
                print(result['answer'])
                
                print(f"\n📚 Sources (Confidence: {result['confidence']:.2f}):")
                print("-" * 30)
                for i, source in enumerate(result['sources'], 1):
                    print(f"{i}. {source['text']}")
                    print(f"   Similarity: {source['similarity']:.3f}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()