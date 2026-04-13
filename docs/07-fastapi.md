# Step 07: FastAPI Wrapper (Phase 2)

This step exposes the RAG pipeline as a **web API** using FastAPI. The API lets you ingest documents and query for the top‑k most similar chunks.

## Goals of this phase
- Provide HTTP endpoints for ingestion and retrieval.
- Keep the RAG core modular and reusable.
- Make it easy to swap vector backends (Chroma vs FAISS).

## Files created in this step
- `src/api/main.py`
- `src/api/models.py`
- `src/api/deps.py`
- `src/api/services/rag_service.py`
- `src/api/routers/ingest.py`
- `src/api/routers/query.py`
- `src/api/routers/health.py`
- `src/api/routers/__init__.py`
- `src/api/__init__.py`

## How it works

### `src/api/services/rag_service.py`
Defines `RAGService`, the orchestration layer used by the API.

Responsibilities:
- `ingest(...)`: chunk → embed → store
- `retrieve(...)`: embed query → vector search
- `generate(...)`: optional LLM response (currently disabled by default)

A `build_service(...)` helper builds the service using:
- `RecursiveCharacterChunker`
- `ChromaEmbedder`
- Either `ChromaVectorStore` or `FaissVectorStore`

### `src/api/deps.py`
Creates a `RAGService` using environment variables so configuration is centralized.

Supported env vars:
- `RAG_VECTOR_BACKEND` = `chroma` or `faiss`
- `RAG_CHROMA_PERSIST` = persistent directory for Chroma
- `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP`

### `src/api/models.py`
Defines request/response schemas:
- `IngestRequest`: list of documents
- `QueryRequest`: query + k + generation params
- `QueryResponse`: answer + optional context

### Routers
- `/health` for basic health checks
- `/ingest` to add documents
- `/query` to retrieve similar chunks

## Example API usage

### Ingest
```json
POST /ingest
{
  "documents": [
    {"text": "RAG stands for Retrieval-Augmented Generation.", "metadata": {"source": "note"}}
  ]
}
```

### Query
```json
POST /query
{
  "query": "What is RAG?",
  "k": 3,
  "include_context": true
}
```

## Next step ideas
- Add a real LLM provider (OpenAI/local) to enable `generate()`.
- Add `/ingest/files` for uploads (PDF/TXT).
- Add authentication + rate limits.
