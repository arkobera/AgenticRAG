import os
from dotenv import load_dotenv
# import google.generativeai as genai
from google import genai
from google.genai import types

from src.rag.doc_proc.processor import DocumentProcessor
from src.rag.vector_store.factory import VectorStoreFactory
from src.rag.retrieval.retriever import HybridRetriever
from src.rag.generation.generator import RAGGenerator
from src.rag.generation.prompts import GroundingPrompts

load_dotenv()
client = genai.Client()

def setup_embedding_fn():
    """Setup embedding function using Gemini"""
    def embed_text(text: str):
        """Embed text using Gemini's embedding model"""
        try:
            result = client.models.embed_content(
                model='gemini-embedding-001',
                contents=text,
            )
            return result.embeddings[0].values #type: ignore
        except Exception as e:
            print(f"Embedding failed: {e}")
            return [0.0] * 768
    
    return embed_text

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


def main():
    """Main RAG pipeline demonstration"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Please configure .env file.")
    print("=" * 80)
    print("RAG System for Product Documentation QA")


    ## Document Processing Step
    print("=" * 80)
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
    vector_store = VectorStoreFactory.create("in_memory")
    print("✓ Created in-memory vector store")


    ## SetUp embedding function
    embedding_fn = setup_embedding_fn()
    print("✓ Setup Gemini embedding function")



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
    stats = vector_store.get_stats() #type:ignore
    print(f"✓ Vector store ready: {stats}")


    ## Retrieval SetUp
    print("\n[3] RETRIEVAL SETUP")
    print("-" * 80)
    retriever = HybridRetriever(
        vector_store=vector_store,
        embedding_fn=embedding_fn, #type:ignore
        dense_weight=0.6,
        sparse_weight=0.4,
    )
    print("✓ Hybrid retriever configured (60% dense, 40% sparse)")


    ## setup Generation

    generator = RAGGenerator(
    retriever=retriever,
    llm_client=client,
    model_name="gemini-2.5-flash",
    min_context_score=0.2,
    top_k=5,
    )

    print("\n[4] GENERATION SETUP")
    print("-" * 80)
    # llm_fn = setup_llm_fn()
    print("✓ Gemini LLM configured")
    


    ## Run Example
    print("\n[5] EXAMPLE QUERIES")
    print("=" * 80)
    
    queries = [
        # "How do I troubleshoot WiFi connection issues?",
        # "What is the document about",
        # "Can multiple phones control the same coffee maker?",
        # "How often should I clean the machine?",
        "What is the capacity of the machine?"
    ]
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 80)
        response = generator.generate(query,use_verification=False)
        print(f"Answer: {response['answer'][:500]}...")
        print(f"Sources: {', '.join(response['sources'])}")
        print(f"Confidence: {response['confidence']:.2f}")
        print(f"Retrieved {response['num_context_chunks']} context chunks")
        
        if 'chunk_scores' in response.keys():
            print("Chunk scores:")
            for score in response['chunk_scores'][:3]:
                print(f"  - {score['chunk_id']}: {score['score']:.3f}")
    print("\n" + "=" * 80)
    print("RAG Pipeline demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
        
