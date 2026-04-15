import os
from dotenv import load_dotenv

from src.rag.doc_proc.processor import DocumentProcessor
from src.rag.vector_store.factory import VectorStoreFactory
from src.rag.retrieval.retriever import HybridRetriever
from src.rag.generation.generator import RAGGenerator
from src.rag.generation.prompts import GroundingPrompts
from src.rag.generation.langchain_setup import setup_embedding_fn, setup_llm
from src.config import get_config, Config

load_dotenv()


def resolve_document_directory() -> str:
    """Pick the first available input directory for local runs."""
    for candidate in ("data", "raw", "benchmark"):
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError("No document directory found. Expected one of: data/, raw/, benchmark/")


def main():
    """Main RAG pipeline demonstration"""
    # Load configuration
    config = Config()
    
    print("=" * 80)
    print("RAG System for Product Documentation QA")
    print("Using HuggingFace Models + FAISS Vector Store + LangChain")
    print("=" * 80)

    # Verify HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not set in .env file")
    else:
        print(f"✓ HF_TOKEN configured")

    ## Document Processing Step
    print("\n[1] DOCUMENT PROCESSING")
    print("-" * 80)
    processor = DocumentProcessor()  # Parameters loaded from config
    document_dir = resolve_document_directory()
    docs = processor.load_documents(document_dir)
    print(f"✓ Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.filename} ({len(doc.content)} chars)")
    
    chunks = processor.process()
    print(f"✓ Created {len(chunks)} chunks")
    print(f"  Total tokens: {sum(c.token_count for c in chunks)}")

    ## Setup Vector Store
    print("\n[2] VECTOR STORE SETUP")
    print("-" * 80)
    # Using FAISS with embedding dimension from config
    embedding_dim = get_config("embeddings.embedding_dim")
    vector_store = VectorStoreFactory.create("faiss", embedding_dim=embedding_dim)
    print(f"✓ Created FAISS vector store ({embedding_dim}-dim embeddings)")

    ## Setup embedding function using HuggingFace
    print("\n[3] EMBEDDING SETUP")
    print("-" * 80)
    embedding_fn = setup_embedding_fn()

    ## Embed and store chunks
    print("Embedding and storing chunks...")
    for i, chunk in enumerate(chunks):
        if i % max(1, len(chunks) // 4) == 0:
            print(f"  Progress: {i}/{len(chunks)}")
        try:
            embedding = embedding_fn(chunk.content)
            chunk.embedding = embedding
        except Exception as e:
            print(f"Embedding failed for chunk {chunk.chunk_id}: {e}")

    vector_store.add_chunks(chunks)
    stats = vector_store.get_stats()
    print(f"✓ Vector store ready: {stats}")

    ## Setup Retriever
    print("\n[4] RETRIEVER SETUP")
    print("-" * 80)
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_fn=embedding_fn,
    )  # Weights loaded from config
    dense_weight = get_config("retriever.dense_weight")
    sparse_weight = get_config("retriever.sparse_weight")
    print(f"✓ Hybrid retriever initialized ({dense_weight*100:.0f}% dense, {sparse_weight*100:.0f}% sparse)")

    ## Setup LLM
    print("\n[5] LLM SETUP")
    print("-" * 80)
    llm_fn = setup_llm()

    ## Setup RAG Generator
    print("\n[6] RAG GENERATOR SETUP")
    print("-" * 80)
    generator = RAGGenerator(
        retriever=retriever,
        llm_fn=llm_fn,
    )  # Parameters loaded from config
    min_context_score = get_config("rag_generator.min_context_score")
    top_k = get_config("rag_generator.top_k")
    print(f"✓ RAG generator initialized (top_k={top_k}, min_score={min_context_score})")

    ## Test the pipeline with sample queries
    print("\n[7] TESTING RAG PIPELINE")
    print("=" * 80)

    test_queries = [
        "What is the main topic of this document?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        try:
            response = generator.generate(query)
            print(f"Answer: {response['answer']}")
            print(f"Sources: {', '.join(response['sources'])}")
            print(f"Confidence: {response['confidence']:.2f}")
            print(f"Context chunks used: {response['num_context_chunks']}")
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("RAG Pipeline Test Complete")
    print("Configuration loaded from: config.yaml")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        
