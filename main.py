import os
from dotenv import load_dotenv

from src.rag.doc_proc.processor import DocumentProcessor
from src.rag.vector_store.factory import VectorStoreFactory
from src.rag.retrieval.retriever import HybridRetriever
from src.rag.generation.generator import RAGGenerator
from src.rag.generation.prompts import GroundingPrompts
from src.rag.generation.langchain_setup import setup_embedding_fn, setup_llm

load_dotenv()


def main():
    """Main RAG pipeline demonstration"""
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
    processor = DocumentProcessor(chunk_size=400, chunk_overlap=100)
    docs = processor.load_documents("data/")
    print(f"✓ Loaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.filename} ({len(doc.content)} chars)")
    
    chunks = processor.process()
    print(f"✓ Created {len(chunks)} chunks")
    print(f"  Total tokens: {sum(c.token_count for c in chunks)}")

    ## Setup Vector Store
    print("\n[2] VECTOR STORE SETUP")
    print("-" * 80)
    # Using FAISS with embedding dimension 384 (for all-MiniLM-L6-v2)
    vector_store = VectorStoreFactory.create("faiss", embedding_dim=384)
    print("✓ Created FAISS vector store (384-dim embeddings)")

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
        dense_weight=0.7,
        sparse_weight=0.3,
    )
    print("✓ Hybrid retriever initialized (70% dense, 30% sparse)")

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
        min_context_score=0.3,
        top_k=5,
    )
    print("✓ RAG generator initialized")

    ## Test the pipeline with sample queries
    print("\n[7] TESTING RAG PIPELINE")
    print("=" * 80)

    test_queries = [
        "What is the main topic of this document?",
        "Can you summarize the content?",
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
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        
