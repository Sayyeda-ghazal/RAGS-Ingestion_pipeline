# Step 04: Vector Store (RAG Core)

This step adds the **vector store layer**, which holds embeddings and lets us run similarity search. You asked for both options:
- ChromaDB (easy, persistent, production‑friendly)
- FAISS (local, in‑memory)

We provide a shared interface and two concrete implementations.

## Files created in this step
- `src/rag_core/vectorstores/base.py`
- `src/rag_core/vectorstores/chroma.py`
- `src/rag_core/vectorstores/faiss.py`
- `src/rag_core/vectorstores/__init__.py`

## What each file does

### `src/rag_core/vectorstores/base.py`
Defines the base interface and result object:
- `SearchResult`: holds a `Document` and its similarity `score`.
- `BaseVectorStore.add(...)`: add documents + vectors.
- `BaseVectorStore.similarity_search(...)`: get the top‑k closest documents.

### `src/rag_core/vectorstores/chroma.py`
Implements a ChromaDB‑backed store.

How it works:
- Creates a Chroma client (persistent if `persist_directory` is set).
- Creates/loads a collection with an embedding function.
- `add(...)` inserts ids, texts, metadatas, and embeddings.
- `similarity_search(...)` queries by vector and returns `SearchResult` objects.

Dependencies:
- `chromadb`

### `src/rag_core/vectorstores/faiss.py`
Implements a local in‑memory FAISS store.

How it works:
- Builds a FAISS index the first time you add vectors.
- Keeps a parallel list of `Document` objects for lookup.
- Uses cosine similarity if `normalize=True` (default), otherwise L2 distance.

Dependencies:
- `faiss-cpu`
- `numpy`

### `src/rag_core/vectorstores/__init__.py`
Exports the vector store API for clean imports.

## Example usage

### ChromaDB
```python
from rag_core.vectorstores import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="rag_chunks",
    persist_directory="./chroma_db",
)

store.add(documents, vectors)
results = store.similarity_search(query_vector, k=5)
```

### FAISS (local)
```python
from rag_core.vectorstores import FaissVectorStore

store = FaissVectorStore(normalize=True)
store.add(documents, vectors)
results = store.similarity_search(query_vector, k=5)
```

## Notes on scores
- Chroma returns distances; lower is closer for most distance metrics.
- FAISS returns higher scores for closer matches when using cosine similarity.

We’ll normalize this later if you want consistent scoring across stores.

## Next step ideas
- Retrieval pipeline (query → top‑k chunks)
- Persistence strategy (save/load FAISS index)
- API endpoints in FastAPI
