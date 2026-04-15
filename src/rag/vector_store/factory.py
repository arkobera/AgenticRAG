from typing import Optional
from src.rag.vector_store.base import VectorStore
from src.rag.vector_store.in_memory import InMemoryVectorStore
from src.rag.vector_store.faiss_store import FAISSVectorStore

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    @staticmethod
    def create(store_type='faiss', embedding_dim: int = 384, **kwargs) -> VectorStore:
        """Create a vector store instance

        Args:
            store_type: Type of store ("faiss", "in_memory")
            embedding_dim: Embedding dimension (for FAISS)
            **kwargs: store-specific configuration
        """
        if store_type == "faiss":
            return FAISSVectorStore(embedding_dim=embedding_dim)
        elif store_type == "in_memory":
            return InMemoryVectorStore()
        else:
            raise ValueError(f"Unknown store type: {store_type}")