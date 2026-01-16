# AgenticRAG

Upload documents → Process automatically → Ask questions → Get AI answers with sources

**RAG** combines document intelligence with large language models to provide accurate, grounded answers backed by your documents

### Key Features

✅ **Hybrid Search** - 60% semantic (dense vectors) + 40% keyword (BM25) matching  
✅ **Gemini Integration** - State-of-the-art embeddings and LLM generation  
✅ **Grounded Responses** - All answers backed by source documents  
✅ **Production-Ready** - Error handling, logging, type safety, validation  

##  Architecture

```
Documents → Chunking → Embedding → Dense Index + Sparse Index
                                           ↓
Query → Embedding → Dense Search (60%) + Sparse Search (40%)
                           ↓
                    Hybrid Ranking → Top-5 Results
                           ↓
                   Grounding Prompt + Context
                           ↓
                   Gemini LLM Generation
                           ↓
                  Grounded Answer + Sources + Confidence

## 📁 Project Structure

```
AI-Powered Document Q&A System/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Dependencies
├── .env.example             # API key template
├── src/rag/                 # Core RAG system
│   ├── document_processing/ # Load & chunk documents
│   ├── vector_store/        # Embeddings & indexing
│   ├── retrieval/           # Hybrid search
│   └── generation/          # LLM responses
├── tests/                   # Unit tests
├── data/                    # Sample documents
└── docs/                    # Documentation
```

## 📜 License

MIT License - See LICENSE file for details

---

**Built with ❤️ using Python, Streamlit, and Google Gemini AI**
