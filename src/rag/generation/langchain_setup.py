"""
LangChain-based embeddings and LLM setup for the RAG system
"""
from typing import List, Callable
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch


load_dotenv()


def setup_embedding_fn() -> Callable[[str], List[float]]:
    """
    Setup HuggingFace embedding function using LangChain.
    Uses 'sentence-transformers/all-MiniLM-L6-v2' for fast embeddings.
    
    Returns:
        Callable that takes text and returns embedding vector
    """
    try:
        # Initialize HuggingFace embeddings through LangChain
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},  # Use CPU, change to "cuda" if GPU available
            show_progress=False,
        )
        
        def embed_text(text: str) -> List[float]:
            """Embed text using HuggingFace embeddings"""
            try:
                # LangChain's embed_query returns a list of floats
                embedding = embeddings.embed_query(text)
                return embedding
            except Exception as e:
                print(f"Embedding failed: {e}")
                return [0.0] * 384  # Return zero vector on error
        
        print("✓ HuggingFace embeddings initialized (all-MiniLM-L6-v2)")
        return embed_text
    
    except Exception as e:
        print(f"Failed to initialize embeddings: {e}")
        raise


def setup_llm() -> Callable[[str], str]:
    """
    Setup HuggingFace LLM using LangChain.
    Uses 'distilgpt2' for fast inference.
    
    Returns:
        Callable that takes prompt and returns generated text
    """
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN not set in .env, using without authentication")
        
        # Model ID - using a smaller model for faster inference
        model_id = "distilgpt2"
        
        # Create pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,  # GPU if available
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        # Wrap in LangChain LLM
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        
        def generate_response(prompt: str) -> str:
            """Generate response using HuggingFace LLM"""
            try:
                response = llm.invoke(prompt)
                return response.strip()
            except Exception as e:
                print(f"LLM generation failed: {e}")
                return f"Sorry, I couldn't generate a response. Error: {e}"
        
        print(f"✓ HuggingFace LLM initialized ({model_id})")
        return generate_response
    
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        raise
