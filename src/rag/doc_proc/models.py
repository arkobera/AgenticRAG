from pydantic import BaseModel, Field
from typing import Optional, List

class DocumentChunk(BaseModel):
    """Represents a single document chunk with meta data"""
    chunk_id: str = Field(..., description="Unique Identifier of Chunk")
    content: str = Field(...,description="The actual text content")
    source_doc: str = Field(...,description='Original document source')
    chunk_index: int = Field(...,description="Index of this chunk in the document")
    start_char: int = Field(...,description='Starting charachter position in original document')
    end_char: int = Field(...,description="Ending charachter position in the original document")
    token_count: int = Field(...,description="Number of tokens in the chunk")
    metadata: dict = Field(default_factory=dict, description="Additional Metadata")
    embedding: Optional[List[float]] = Field(None,description="Vector Embedding (Optional)")

class Document(BaseModel):
    """Represent a source document"""
    doc_id: str = Field(...,description="Unique document identifier")
    filename: str = Field(...,description="source filename")
    content: str = Field(...,description="Full document content")
    doc_type: str = Field(default="product manual",description="Type of document")
    metadata: dict = Field(default_factory=dict,description='document level metadata')

class RetrievalResult(BaseModel):
    """result from retrieval"""
    chunk_id: str
    content: str
    source_doc: str
    score: float = Field(...,description="Relevance score (0-1)")
    search_type: str = Field(...,description="Type of seaarch that returned result")