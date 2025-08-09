"""
Financial RAG Assistant - Streamlit Web Interface
"""

import streamlit as st
import os
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.rag_system import FinancialRAGSystem


# Page configuration
st.set_page_config(
    page_title="Financial RAG Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f4e79;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #2c5aa0;
    margin-bottom: 1rem;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.25rem;
    padding: 1rem;
    margin: 1rem 0;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 0.25rem;
    padding: 1rem;
    margin: 1rem 0;
}
.info-box {
    background-color: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 0.25rem;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'current_index' not in st.session_state:
        st.session_state.current_index = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False


def initialize_rag_system():
    """Initialize the RAG system."""
    if not st.session_state.system_initialized:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = FinancialRAGSystem(
                embedding_model="all-MiniLM-L6-v2",
                llm_model="mistral",
                chunk_size=500,
                overlap=100
            )
            st.session_state.system_initialized = True
        st.success("✅ RAG System initialized successfully!")


def display_system_status():
    """Display system status in sidebar."""
    if st.session_state.rag_system:
        status = st.session_state.rag_system.get_system_status()
        
        st.sidebar.markdown("### 📊 System Status")
        
        # Create status indicators
        if status['llm_status'] == "Available":
            st.sidebar.success(f"🤖 LLM: {status['llm_model']} (Ready)")
        elif "Mock" in status['llm_status']:
            st.sidebar.warning(f"🤖 LLM: Mock Mode")
        else:
            st.sidebar.error(f"🤖 LLM: Not Available")
        
        st.sidebar.info(f"🧠 Embedding: {status['embedding_model']}")
        st.sidebar.info(f"📄 Documents: {status['documents_loaded']}")
        
        if status['index_info']['index_exists']:
            st.sidebar.success(f"🔍 Index: {status['index_info']['total_vectors']} vectors")
        else:
            st.sidebar.warning("🔍 Index: Not built")


def file_upload_section():
    """Handle file upload and document processing."""
    st.markdown('<h2 class="sub-header">📁 Document Upload</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload a financial PDF document",
        type=['pdf'],
        help="Upload 10-K filings, earnings reports, or other financial documents"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                # Load document
                doc_id = st.session_state.rag_system.load_document(temp_path, uploaded_file.name.split('.')[0])
                
                # Update session state
                st.session_state.documents[doc_id] = {
                    'name': uploaded_file.name,
                    'size': len(uploaded_file.getbuffer()),
                    'processed': True
                }
                
                # Build index
                index_id = st.session_state.rag_system.build_index([doc_id])
                st.session_state.current_index = index_id
            
            st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
            
            # Display document info
            status = st.session_state.rag_system.get_system_status()
            if doc_id in status['documents']:
                doc_info = status['documents'][doc_id]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pages", doc_info['pages'])
                with col2:
                    st.metric("Text Chunks", doc_info['chunks'])
                with col3:
                    st.metric("File Size", f"{len(uploaded_file.getbuffer()) / 1024:.1f} KB")
            
        except Exception as e:
            st.error(f"❌ Error processing document: {str(e)}")
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)


def document_management_section():
    """Display loaded documents and management options."""
    if st.session_state.rag_system:
        status = st.session_state.rag_system.get_system_status()
        
        if status['documents']:
            st.markdown('<h2 class="sub-header">📚 Loaded Documents</h2>', unsafe_allow_html=True)
            
            # Create a DataFrame for better display
            doc_data = []
            for doc_id, doc_info in status['documents'].items():
                doc_data.append({
                    'Document': doc_info.get('title', doc_id),
                    'Pages': doc_info['pages'],
                    'Chunks': doc_info['chunks'],
                    'Path': os.path.basename(doc_info['file_path'])
                })
            
            df = pd.DataFrame(doc_data)
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            if len(doc_data) > 1:
                fig = px.bar(
                    df, 
                    x='Document', 
                    y='Chunks', 
                    title='Text Chunks per Document',
                    color='Pages',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)


def chat_interface():
    """Main chat interface for querying documents."""
    st.markdown('<h2 class="sub-header">💬 Ask Questions</h2>', unsafe_allow_html=True)
    
    if not st.session_state.current_index:
        st.warning("⚠️ Please upload and process a document first.")
        return
    
    # Query input
    question = st.text_input(
        "Ask a question about your financial document:",
        placeholder="e.g., What is the company's revenue growth?",
        key="question_input"
    )
    
    # Query parameters
    col1, col2 = st.columns([3, 1])
    with col2:
        k_chunks = st.selectbox("Sources to retrieve:", [3, 5, 7, 10], index=1)
        max_tokens = st.selectbox("Max response length:", [250, 500, 750, 1000], index=1)
    
    if st.button("🔍 Ask Question", type="primary") and question:
        try:
            with st.spinner("Searching document and generating answer..."):
                # Query the RAG system
                result = st.session_state.rag_system.query(
                    question, 
                    k=k_chunks, 
                    max_tokens=max_tokens
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': question,
                    'answer': result['answer'],
                    'confidence': result['confidence'],
                    'sources': result['sources'],
                    'timestamp': time.strftime("%H:%M:%S")
                })
            
            # Display result
            st.markdown("### 💡 Answer")
            st.write(result['answer'])
            
            # Display confidence and sources
            col1, col2 = st.columns([1, 3])
            with col1:
                confidence_color = "green" if result['confidence'] > 0.7 else "orange" if result['confidence'] > 0.5 else "red"
                st.markdown(f"**Confidence:** <span style='color: {confidence_color}'>{result['confidence']:.2f}</span>", 
                           unsafe_allow_html=True)
            
            # Sources
            with st.expander("📚 View Sources", expanded=False):
                for i, source in enumerate(result['sources'], 1):
                    st.markdown(f"**Source {i}** (Similarity: {source['similarity']:.3f})")
                    st.text_area(
                        f"Source {i}",
                        source['full_text'],
                        height=100,
                        key=f"source_{i}_{len(st.session_state.chat_history)}"
                    )
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


def chat_history_section():
    """Display chat history."""
    if st.session_state.chat_history:
        st.markdown('<h2 class="sub-header">📝 Chat History</h2>', unsafe_allow_html=True)
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
            with st.expander(f"{chat['timestamp']} - {chat['question'][:50]}...", expanded=i==1):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                st.markdown(f"**Confidence:** {chat['confidence']:.2f}")


def analytics_section():
    """Display analytics and insights."""
    if st.session_state.rag_system and st.session_state.chat_history:
        st.markdown('<h2 class="sub-header">📈 Analytics</h2>', unsafe_allow_html=True)
        
        # Query analytics
        confidences = [chat['confidence'] for chat in st.session_state.chat_history]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", len(st.session_state.chat_history))
        with col2:
            st.metric("Avg Confidence", f"{sum(confidences) / len(confidences):.2f}")
        with col3:
            high_conf = sum(1 for c in confidences if c > 0.7)
            st.metric("High Confidence", f"{high_conf}/{len(confidences)}")
        
        # Confidence over time
        if len(confidences) > 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=confidences,
                mode='lines+markers',
                name='Confidence Score',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title='Query Confidence Over Time',
                xaxis_title='Query Number',
                yaxis_title='Confidence Score',
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)


def main():
    """Main Streamlit application."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Financial RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize RAG system
    if not st.session_state.system_initialized:
        initialize_rag_system()
    
    # Sidebar
    with st.sidebar:
        st.title("🎛️ Control Panel")
        display_system_status()
        
        st.markdown("---")
        
        # Reset options
        if st.button("🔄 Reset System"):
            if st.session_state.rag_system:
                st.session_state.rag_system.reset_system()
            st.session_state.documents = {}
            st.session_state.current_index = None
            st.session_state.chat_history = []
            st.success("System reset successfully!")
        
        if st.button("🗑️ Clear Cache"):
            if st.session_state.rag_system:
                st.session_state.rag_system.clear_cache()
            st.success("Cache cleared successfully!")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "💬 Chat", "📝 History", "📈 Analytics"])
    
    with tab1:
        file_upload_section()
        document_management_section()
    
    with tab2:
        chat_interface()
    
    with tab3:
        chat_history_section()
    
    with tab4:
        analytics_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Financial RAG Assistant - Built with Streamlit, SentenceTransformers, FAISS, and Ollama"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()