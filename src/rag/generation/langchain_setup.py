"""
LangChain-based embeddings and LLM setup for the RAG system
Uses configuration from config.yaml
"""
from typing import List, Callable
import os
import re
from dotenv import load_dotenv
from sklearn.feature_extraction.text import HashingVectorizer

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import torch
from src.config import get_config
from src.logger import get_logger

logger = get_logger(__name__)

load_dotenv()


def _build_local_embedding_fn() -> Callable[[str], List[float]]:
    """Fallback embedder that works fully offline."""
    embedding_dim = get_config("embeddings.embedding_dim")
    vectorizer = HashingVectorizer(
        n_features=embedding_dim,
        alternate_sign=False,
        norm="l2",
    )

    def embed_text(text: str) -> List[float]:
        vector = vectorizer.transform([text]).toarray()[0]
        return vector.astype(float).tolist()

    print("✓ Using local hashing embeddings fallback")
    return embed_text


def _build_local_llm_fallback() -> Callable[[str], str]:
    """Fallback answerer that extracts the most relevant context snippets."""
    def generate_response(prompt: str) -> str:
        source_pattern = re.compile(
            r"\[Source:\s*(?P<source>[^\]]+)\]\n(?P<content>.*?)(?=\n\n\[Source:|\n\nQUESTION:)",
            re.DOTALL,
        )
        matches = source_pattern.findall(prompt)
        if not matches:
            return "I don't have information about that in the documentation."

        snippets = []
        for source, content in matches[:3]:
            cleaned = " ".join(content.split())
            if cleaned:
                snippets.append(f"{cleaned[:240]} [Source: {source}]")

        return " ".join(snippets) if snippets else "I don't have information about that in the documentation."

    print("✓ Using local extractive LLM fallback")
    return generate_response


def setup_embedding_fn() -> Callable[[str], List[float]]:
    """
    Setup HuggingFace embedding function using LangChain.
    Uses model and parameters from config.yaml
    
    Returns:
        Callable that takes text and returns embedding vector
    """
    logger.info("Setting up embedding function")
    local_fallback = _build_local_embedding_fn()
    try:
        # Load configuration
        model_name = get_config("embeddings.model_name")
        device = get_config("embeddings.device")
        show_progress = get_config("embeddings.show_progress")
        
        logger.debug(f"Initializing HuggingFace embeddings: model={model_name}, device={device}")
        # Initialize HuggingFace embeddings through LangChain
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            show_progress=show_progress,
        )
        
        def embed_text(text: str) -> List[float]:
            """Embed text using HuggingFace embeddings"""
            try:
                # LangChain's embed_query returns a list of floats
                embedding = embeddings.embed_query(text)
                return embedding
            except Exception as e:
                logger.error(f"Embedding failed: {e}, using fallback", exc_info=True)
                return local_fallback(text)
        
        logger.info(f"HuggingFace embeddings initialized successfully ({model_name})")
        return embed_text
    
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings: {e}, using fallback", exc_info=True)
        return local_fallback


def setup_llm() -> Callable[[str], str]:
    """
    Setup HuggingFace LLM using LangChain.
    Uses model and parameters from config.yaml
    
    Returns:
        Callable that takes prompt and returns generated text
    """
    logger.info("Setting up LLM")
    local_fallback = _build_local_llm_fallback()
    try:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN not set in .env, continuing without authentication")
        
        # Load configuration
        model_id = get_config("llm.model_id")
        max_new_tokens = get_config("llm.max_new_tokens")
        do_sample = get_config("llm.do_sample")
        temperature = get_config("llm.temperature")
        top_p = get_config("llm.top_p")
        device = get_config("llm.device")
        
        logger.debug(f"LLM config: model={model_id}, device={device}, max_tokens={max_new_tokens}")
        # Determine device
        if device == "auto":
            device_id = 0 if torch.cuda.is_available() else -1
        elif device == "cuda":
            device_id = 0
        else:  # cpu
            device_id = -1
        
        logger.debug(f"Using device_id: {device_id}")
        # Create pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model_id,
            device=device_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Wrap in LangChain LLM
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        
        def generate_response(prompt: str) -> str:
            """Generate response using HuggingFace LLM"""
            try:
                logger.debug(f"Generating response for prompt (length: {len(prompt)} chars)")
                response = llm.invoke(prompt)
                logger.debug(f"Response generated (length: {len(response)} chars)")
                return response.strip()
            except Exception as e:
                logger.error(f"LLM generation failed: {e}, using fallback", exc_info=True)
                return local_fallback(prompt)
        
        logger.info(f"HuggingFace LLM initialized successfully ({model_id})")
        return generate_response
    
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace LLM: {e}, using fallback", exc_info=True)
        return local_fallback
