from typing import Optional
from src.rag.vector_store.base import VectorStore
from src.rag.vector_store.in_memory import InMemoryVectorStore

class VectorStoreFactory:
    """Factory for creating vector store instances"""
    @staticmethod
    def create(store_type='in_memory',**kwargs)->VectorStore:
        """Create a vector store instance

        Args:
        store_type: Type of store ("in_memory","pinecone","milvus")
        **kwargs: store-specific configuration
        """
        if store_type== "in_memory":
            return InMemoryVectorStore()
        else:
            raise ValueError(f"Unkown store type: {store_type}")