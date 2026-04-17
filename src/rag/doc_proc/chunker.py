import re
from typing import List, Optional
from src.rag.doc_proc.models import DocumentChunk
from src.logger import get_logger

logger = get_logger(__name__)

class SemanticChunker:
    """
    Chunks documents into semantic coherent units.
    Supports both fixed size and semantic aware chunking
    """
    def __init__(self,chunk_size: int = 400, chunk_overlap:int=100,min_chunk_size: int = 50) -> None:
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target tokens per chunk (approximate)
            chunk_overlap: Tokens to overlap between chunks
        """
        logger.info(f"Initializing SemanticChunker with chunk_size={chunk_size}, overlap={chunk_overlap}, min_size={min_chunk_size}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    def _count_tokens_approx(self,text:str)->int:
        """
        Approximate token count
        :param self: Description
        :param text: Description
        :type text: str
        :return: Description
        :rtype: int
        """
        return len(text.split())
    
    def _split_on_delimiters(self,text:str)->List[str]:
        """
        Split text on semantic boundaries
        
        :param self: Description
        :param text: Description
        :type text: str
        :return: Description
        :rtype: List[str]
        """
        paragraphs = text.split('\n\n')
        segments = []
        for para in paragraphs:
            if not para.strip():
                continue
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())
            segments.extend(sentences)
        return [s.strip() for s in segments if s.strip()]
    
    def chunk(self,text:str,doc_id: str, source_doc:str,metadata: Optional[dict]=None)->List[DocumentChunk]:
        """
        Chunk a document into semantic units
        
        :param self: Description
        :param text: Description
        :type text: str
        :param doc_id: Description
        :type doc_id: str
        :param source_doc: Description
        :type source_doc: str
        :param metadata: Description
        :type metadata: Optional[dict]
        :return: Description
        :rtype: List[DocumentChunk]
        """
        logger.info(f"Chunking document: {doc_id} from {source_doc}")
        if metadata is None:
            metadata = {}
        segments = self._split_on_delimiters(text)
        logger.debug(f"Split text into {len(segments)} segments")

        chunks = []
        current_chunk = []
        current_chunk_pos = 0
        chunk_index = 0

        for segment in segments:
            current_chunk.append(segment)
            current_tokens = self._count_tokens_approx(''.join(current_chunk))
            if current_tokens >= self.chunk_size or segment == segments[-1]:
                chunk_text = ' '.join(current_chunk)
                if self._count_tokens_approx(chunk_text) >= self.min_chunk_size:
                    chunk_id = f"{doc_id}_chunk_{chunk_index}"
                    start_char = text.find(chunk_text)
                    end_char = start_char + len(chunk_text)
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        source_doc=source_doc,
                        chunk_index=chunk_index,
                        start_char=start_char if start_char>=0 else current_chunk_pos,
                        end_char=end_char if end_char >=0 else current_chunk_pos + len(chunk_text),
                        token_count=self._count_tokens_approx(chunk_text),
                        metadata=metadata.copy(),
                        embedding=None
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    current_chunk_pos += len(chunk_text)+1
                if current_tokens >= self.chunk_size:
                    overlap_segments = []
                    remaining_tokens = 0
                    for seg in reversed(current_chunk):
                        overlap_segments.insert(0, seg)
                        remaining_tokens += self._count_tokens_approx(seg)
                        if remaining_tokens >= self.chunk_overlap:
                            break
                    current_chunk = overlap_segments
                else:
                    current_chunk = []
        
        logger.info(f"Created {len(chunks)} chunks for {doc_id}")
        return chunks