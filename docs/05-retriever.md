# Step 05: Retriever (RAG Core)

This step adds a **retriever**, which takes a user query, turns it into a vector, and fetches the top‑k most similar chunks from the vector store.

## Why this step matters
The retriever is the bridge between “what the user asked” and “which stored chunks are relevant.” It standardizes this process so higher layers (API, RAG chain) stay clean.

## Files created in this step
- `src/rag_core/retrievers/base.py`
- `src/rag_core/retrievers/vector.py`
- `src/rag_core/retrievers/__init__.py`

## What each file does

### `src/rag_core/retrievers/base.py`
Defines the base interface:
- `retrieve(query, k)`: returns a list of `SearchResult` objects.

### `src/rag_core/retrievers/vector.py`
Implements `VectorRetriever`:
1. Uses the embedder to convert the query into a vector.
2. Calls the vector store’s `similarity_search(...)` to fetch top‑k matches.

### `src/rag_core/retrievers/__init__.py`
Exports `BaseRetriever` and `VectorRetriever` for clean imports.

## Example usage
```python
from rag_core.embeddings import ChromaEmbedder
from rag_core.vectorstores import ChromaVectorStore
from rag_core.retrievers import VectorRetriever

embedder = ChromaEmbedder()
store = ChromaVectorStore(collection_name="rag_chunks", persist_directory="./chroma_db")
retriever = VectorRetriever(embedder=embedder, vector_store=store)

results = retriever.retrieve("What is RAG?", k=5)
for result in results:
    print(result.score, result.document.metadata.get("source"))
```

## Next step ideas
- RAG prompt builder (combine query + retrieved chunks)
- LLM response generation
- API endpoints (FastAPI)
