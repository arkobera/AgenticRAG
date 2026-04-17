import os
from dotenv import load_dotenv

from src.rag.doc_proc.processor import DocumentProcessor
from src.rag.vector_store.factory import VectorStoreFactory
from src.rag.retrieval.retriever import HybridRetriever
from src.rag.generation.generator import RAGGenerator
from src.rag.generation.prompts import GroundingPrompts
from src.rag.generation.langchain_setup import setup_embedding_fn, setup_llm
from src.config import get_config, Config
from src.logger import setup_logging, get_logger

load_dotenv()

# Setup logging for the main module
logger = get_logger(__name__)


def resolve_document_directory() -> str:
    """Pick the first available input directory for local runs."""
    for candidate in ("data", "raw", "benchmark"):
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError("No document directory found. Expected one of: data/, raw/, benchmark/")


def main():
    """Main RAG pipeline demonstration"""
    # Setup logging first
    setup_logging()
    logger.info("Starting RAG pipeline")
    
    # Load configuration
    config = Config()
    logger.info("Configuration loaded from config.yaml")
    
    print("=" * 80)
    print("RAG System for Product Documentation QA")
    print("Using HuggingFace Models + FAISS Vector Store + LangChain")
    print("=" * 80)

    # Verify HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.warning("HF_TOKEN not set in .env file")
        print("Warning: HF_TOKEN not set in .env file")
    else:
        logger.info("HF_TOKEN configured")
        print(f"✓ HF_TOKEN configured")

    ## Document Processing Step
    print("\n[1] DOCUMENT PROCESSING")
    print("-" * 80)
    logger.info("Starting document processing")
    processor = DocumentProcessor()  # Parameters loaded from config
    document_dir = resolve_document_directory()
    logger.debug(f"Document directory: {document_dir}")
    docs = processor.load_documents(document_dir)
    logger.info(f"Loaded {len(docs)} documents")
    print(f"✓ Loaded {len(docs)} documents")
    for doc in docs:
        logger.debug(f"Document: {doc.filename} ({len(doc.content)} chars)")
        print(f"  - {doc.filename} ({len(doc.content)} chars)")
    
    chunks = processor.process()
    logger.info(f"Created {len(chunks)} chunks")
    total_tokens = sum(c.token_count for c in chunks)
    logger.debug(f"Total tokens: {total_tokens}")
    print(f"✓ Created {len(chunks)} chunks")
    print(f"  Total tokens: {total_tokens}")

    ## Setup Vector Store
    print("\n[2] VECTOR STORE SETUP")
    print("-" * 80)
    logger.info("Setting up vector store")
    # Using FAISS with embedding dimension from config
    embedding_dim = get_config("embeddings.embedding_dim")
    vector_store = VectorStoreFactory.create("faiss", embedding_dim=embedding_dim)
    logger.info(f"Created FAISS vector store with {embedding_dim}-dim embeddings")
    print(f"✓ Created FAISS vector store ({embedding_dim}-dim embeddings)")

    ## Setup embedding function using HuggingFace
    print("\n[3] EMBEDDING SETUP")
    print("-" * 80)
    logger.info("Setting up embedding function")
    embedding_fn = setup_embedding_fn()

    ## Embed and store chunks
    print("Embedding and storing chunks...")
    logger.info(f"Embedding {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        if i % max(1, len(chunks) // 4) == 0:
            logger.debug(f"Embedding progress: {i}/{len(chunks)}")
            print(f"  Progress: {i}/{len(chunks)}")
        try:
            embedding = embedding_fn(chunk.content)
            chunk.embedding = embedding
        except Exception as e:
            logger.error(f"Embedding failed for chunk {chunk.chunk_id}: {e}", exc_info=True)
            print(f"Embedding failed for chunk {chunk.chunk_id}: {e}")

    vector_store.add_chunks(chunks)
    logger.info("All chunks added to vector store")
    stats = vector_store.get_stats()
    logger.debug(f"Vector store stats: {stats}")
    print(f"✓ Vector store ready: {stats}")

    ## Setup Retriever
    print("\n[4] RETRIEVER SETUP")
    print("-" * 80)
    logger.info("Setting up retriever")
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_fn=embedding_fn,
    )  # Weights loaded from config
    dense_weight = get_config("retriever.dense_weight")
    sparse_weight = get_config("retriever.sparse_weight")
    logger.info(f"Hybrid retriever initialized: {dense_weight*100:.0f}% dense, {sparse_weight*100:.0f}% sparse")
    print(f"✓ Hybrid retriever initialized ({dense_weight*100:.0f}% dense, {sparse_weight*100:.0f}% sparse)")

    ## Setup LLM
    print("\n[5] LLM SETUP")
    print("-" * 80)
    logger.info("Setting up LLM")
    llm_fn = setup_llm()

    ## Setup RAG Generator
    print("\n[6] RAG GENERATOR SETUP")
    print("-" * 80)
    logger.info("Setting up RAG generator")
    generator = RAGGenerator(
        retriever=retriever,
        llm_fn=llm_fn,
    )  # Parameters loaded from config
    min_context_score = get_config("rag_generator.min_context_score")
    top_k = get_config("rag_generator.top_k")
    logger.info(f"RAG generator initialized: top_k={top_k}, min_score={min_context_score}")
    print(f"✓ RAG generator initialized (top_k={top_k}, min_score={min_context_score})")

    ## Test the pipeline with sample queries
    print("\n[7] TESTING RAG PIPELINE")
    print("=" * 80)
    logger.info("Testing RAG pipeline with sample queries")

    test_queries = [
        "What is the main topic of this document?",
    ]

    for i, query in enumerate(test_queries, 1):
        logger.info(f"Processing test query {i}: {query}")
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        try:
            response = generator.generate(query)
            logger.info(f"Generated response for query {i}")
            logger.debug(f"Response confidence: {response['confidence']:.2f}, chunks: {response['num_context_chunks']}")
            print(f"Answer: {response['answer']}")
            print(f"Sources: {', '.join(response['sources'])}")
            print(f"Confidence: {response['confidence']:.2f}")
            print(f"Context chunks used: {response['num_context_chunks']}")
        except Exception as e:
            logger.error(f"Error generating response for query {i}: {e}", exc_info=True)
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("RAG Pipeline Test Complete")
    print("Configuration loaded from: config.yaml")
    print("=" * 80)
    logger.info("RAG pipeline test complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        
