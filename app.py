import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.rag.doc_proc.processor import DocumentProcessor
from src.rag.vector_store.factory import VectorStoreFactory
from src.rag.retrieval.retriever import HybridRetriever
from src.rag.generation.generator import RAGGenerator
from src.rag.generation.prompts import GroundingPrompts

load_dotenv()
client = genai.Client()

st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

## Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_session():
    """Initialize Streamlit session state"""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "generator" not in st.session_state:
        st.session_state.generator = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "query_history" not in st.session_state:
        st.session_state.query_history = []


initialize_session()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def setup_embedding_fn():
    """Setup Gemini embedding function with caching"""
    def embed_text(text: str):
        try:
            result = client.models.embed_content(
                model='gemini-embedding-001',
                contents=text,
            )
            return result.embeddings[0].values #type: ignore
        except Exception as e:
            st.warning(f"Embedding API limit reached. Using fallback embeddings.")
            import random
            random.seed(hash(text) % (2**32))
            return [random.uniform(-1, 1) for _ in range(768)]
    return embed_text

@st.cache_resource
def setup_llm_fn(prompt: str):
    """Setup LLM function using Gemini"""
    try:
        system_instruction = GroundingPrompts.system_prompt()
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
        )
        return response.text 
    except Exception as e:
        print(f"Error While generating {e}")


def process_uploaded_documents(uploaded_files):
    """Process uploaded files and create RAG pipeline"""
    
    if not uploaded_files:
        st.error("Please upload at least one document")
        return False
    
    try:
        # Create temporary directory for uploads
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded files
            for uploaded_file in uploaded_files:
                file_path = temp_path / uploaded_file.name
                file_path.write_bytes(uploaded_file.getvalue())
            
            # Step 1: Load and process documents
            st.info("📄 Processing documents...")
            processor = DocumentProcessor(chunk_size=400, chunk_overlap=100)
            docs = processor.load_documents(str(temp_path))
            
            if not docs:
                st.error("No valid documents found. Please upload CSV, TXT, MD, or JSON files.")
                return False
            
            st.session_state.documents = docs
            chunks = processor.process()
            st.session_state.chunks = chunks    
            st.success(f"✓ Loaded {len(docs)} documents, created {len(chunks)} chunks")

            # Step 2: Setup vector store
            st.info("🔍 Setting up vector store...")
            vector_store = VectorStoreFactory.create("in_memory")
            embedding_fn = setup_embedding_fn()

            # Embed chunks with progress bar
            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                embedding = embedding_fn(chunk.content)
                chunk.embedding = embedding
                progress_bar.progress((i + 1) / len(chunks))

            vector_store.add_chunks(chunks)
            st.session_state.vector_store = vector_store
            st.success(f"✓ Vector store ready: {vector_store.get_stats()}") #type:ignore
            
            # Step 3: Setup retrieval
            st.info("🎯 Configuring hybrid retrieval...")
            retriever = HybridRetriever(
                vector_store=vector_store,
                embedding_fn=embedding_fn,#type: ignore
                dense_weight=0.6,
                sparse_weight=0.4,
            )
            st.session_state.retriever = retriever
            st.success("✓ Hybrid retriever configured (60% dense, 40% sparse)")

            # Step 4: Setup generation
            st.info("🚀 Initializing LLM...")
            generator = RAGGenerator(
                retriever=retriever,
                llm_client=client,
                model_name="gemini-2.5-flash",
                min_context_score=0.2,
                top_k=5,
            )
            st.session_state.generator = generator
            st.success("✓ RAG system ready!")
            st.session_state.processed = True
            return True
    except Exception as e:
        st.error(f"❌ Error processing documents: {str(e)}")
        return False
    

# ============================================================================
# MAIN LAYOUT
# ============================================================================

# Header
st.markdown("# 🤖 Intelligent Document Q&A System Using RAG")
st.markdown("Upload documents and ask questions powered by RAG + Gemini AI")

with st.sidebar:
    st.markdown("### Retrieval Settings")
    dense_weight = st.slider(
        "Dense Search Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1,
        help="Weight for semantic vector similarity (cosine)"
    )
    sparse_weight = 1.0 - dense_weight
    st.caption(f"Sparse Search Weight: {sparse_weight:.1f}")

    st.markdown("---")

    # Stats
    if st.session_state.processed:
        st.markdown("### 📊 Pipeline Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.documents))
        with col2:
            st.metric("Chunks", len(st.session_state.chunks))
        
        total_tokens = sum(c.token_count for c in st.session_state.chunks)
        st.metric("Total Tokens", total_tokens)

tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "❓ Ask Questions", "📝 History"])
# ========================================================================
# TAB 1: DOCUMENT UPLOAD & PROCESSING
# ========================================================================
    
with tab1:
    st.header("Upload Documents")
    st.markdown("""
        Upload product manuals, FAQs, documentation, or any text files. 
        Supported formats: **TXT**, **MD**, **JSON**, **CSV**.
    """)
        
    col1, col2 = st.columns([2, 1])
        
    with col1:
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["txt", "md", "json","csv"],
            accept_multiple_files=True,
            help="Upload multiple documents at once"
        )
    with col2:
        st.markdown("#### File Upload")
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
        
    st.markdown("---")
        
        # Process button
    if uploaded_files:
        if st.button("🔄 Process Documents", use_container_width=True):
            with st.spinner("Processing... This may take a moment"):
                success = process_uploaded_documents(uploaded_files)
                if success:
                    st.balloons()
    else:
        st.info("👆 Upload documents to get started")
        
        # Display current state
    if st.session_state.processed:
        st.markdown("---")
        st.markdown("## ✅ System Status: Ready")
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Loaded", len(st.session_state.documents))
        with col2:
            st.metric("Text Chunks", len(st.session_state.chunks))
        with col3:
            total_tokens = sum(c.token_count for c in st.session_state.chunks)
            st.metric("Tokens", total_tokens)
        with st.expander("📄 Document Details"):
            for doc in st.session_state.documents:
                st.markdown(f"**{doc.filename}**")
                st.caption(f"Type: {doc.doc_type} | Size: {len(doc.content)} chars")
# ========================================================================
# TAB 2: QUERY INTERFACE
# ========================================================================
    
with tab2:
        st.header("Ask Questions")
        
        if not st.session_state.processed:
            st.warning("### 📤 Please upload and process documents first")
        else:
            st.markdown("Ask any question about your documents. The system will search for relevant context and provide grounded answers.")
            
            # Query input
            query = st.text_input(
                "Your question:",
                placeholder="e.g., How do I troubleshoot WiFi issues?",
                help="Ask anything about your documents"
            )
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                search_button = st.button("🔍 Search", use_container_width=True)
            with col2:
                top_k = st.number_input("Top Results", min_value=1, max_value=10, value=5)
            with col3:
                st.write("")  # Spacing
            
            # Execute query
            if search_button and query:
                with st.spinner("Searching and generating response..."):
                    try:
                        # Update retriever weights
                        st.session_state.retriever.dense_weight = dense_weight
                        st.session_state.retriever.sparse_weight = sparse_weight
                        
                        # Generate response
                        response = st.session_state.generator.generate(
                            query,
                            use_verification=False
                        )
                        
                        # Add to history
                        st.session_state.query_history.append({
                            "query": query,
                            "response": response
                        })
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("### 💡 Answer")
                        st.markdown(response['answer'])
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{response['confidence']:.1%}")
                        with col2:
                            st.metric("Context Chunks", response['num_context_chunks'])
                        with col3:
                            st.metric("Sources", len(response['sources']))
                        
                        # Sources
                        st.markdown("### 📚 Sources")
                        for source in response['sources']:
                            st.caption(f"📄 {source}")
                        
                        # Chunk scores
                        if response['chunk_scores']:
                            with st.expander("📊 Retrieval Scores"):
                                for score in response['chunk_scores']:
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.caption(score['chunk_id'])
                                    with col2:
                                        st.metric("Score", f"{score['score']:.3f}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            elif search_button:
                st.warning("Please enter a question")

# ========================================================================
# TAB 3: QUERY HISTORY
# ========================================================================

with tab3:
        st.header("Query History")
        
        if not st.session_state.query_history:
            st.info("No queries yet. Ask a question in the 'Ask Questions' tab!")
        else:
            st.markdown(f"**Total Queries: {len(st.session_state.query_history)}**")
            st.markdown("---")
            
            for i, item in enumerate(reversed(st.session_state.query_history), 1):
                with st.expander(f"Query {len(st.session_state.query_history) - i + 1}: {item['query'][:50]}..."):
                    st.markdown("**Question:**")
                    st.write(item['query'])
                    st.markdown("**Answer:**")
                    st.write(item['response']['answer'][:500] + "...")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"Confidence: {item['response']['confidence']:.1%}")
                    with col2:
                        st.caption(f"Chunks: {item['response']['num_context_chunks']}")
                    with col3:
                        st.caption(f"Sources: {len(item['response']['sources'])}")
            if st.button("📥 Export History as Text"):
                history_text = ""
                for i, item in enumerate(st.session_state.query_history, 1):
                    history_text += f"\n{'='*80}\n"
                    history_text += f"Query {i}: {item['query']}\n"
                    history_text += f"{'-'*80}\n"
                    history_text += f"Answer: {item['response']['answer']}\n"
                    history_text += f"Confidence: {item['response']['confidence']:.1%}\n"
                    history_text += f"Sources: {', '.join(item['response']['sources'])}\n"
                
                st.download_button(
                    label="Download History",
                    data=history_text,
                    file_name="query_history.txt",
                    mime="text/plain"
                )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>RAG v1.0 | Powered by Gemini AI + RAG | Built with Streamlit</p>
    <p>🚀 <strong>Intelligent Document Q&A System</strong></p>
</div>
""", unsafe_allow_html=True)