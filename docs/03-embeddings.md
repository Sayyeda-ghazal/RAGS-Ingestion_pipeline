# Step 03: Embeddings (RAG Core)

This step converts chunked text into **vectors** so we can store and search them in a vector database. Since you chose **ChromaDB**, we build an embedding adapter that plugs into Chroma’s built‑in embedding functions.

## Why embeddings matter
Vector embeddings let us measure semantic similarity. Instead of matching exact keywords, we compare vectors to find the most relevant chunks during retrieval.

## Files created in this step
- `src/rag_core/embeddings/base.py`
- `src/rag_core/embeddings/chroma.py`
- `src/rag_core/embeddings/__init__.py`

## What each file does

### `src/rag_core/embeddings/base.py`
Defines the base interface for embedders:

- `embed_texts(texts)`: converts a list of strings into vectors.
- `embed_query(text)`: convenience method for a single query (uses `embed_texts`).

### `src/rag_core/embeddings/chroma.py`
Implements `ChromaEmbedder`, which wraps ChromaDB’s embedding utilities.

How it works:
- If you pass a custom `embedding_function`, it uses that directly.
- Otherwise it creates a built‑in Chroma embedding function based on `provider`.

Supported providers:
- `sentence-transformers` (default)
- `openai`

Default model choices:
- Sentence‑Transformers: `all-MiniLM-L6-v2`
- OpenAI: `text-embedding-3-small`

If `chromadb` is not installed, it raises a clear error explaining what to install.

### `src/rag_core/embeddings/__init__.py`
Exports `BaseEmbedder` and `ChromaEmbedder` for clean imports.

## Example usage
```python
from rag_core.embeddings import ChromaEmbedder

# Option 1: Local sentence-transformers (default)
embedder = ChromaEmbedder()

# Option 2: OpenAI embeddings
# embedder = ChromaEmbedder(provider="openai", api_key="YOUR_KEY")

vectors = embedder.embed_texts(["hello world", "rag pipeline"])
print(len(vectors), len(vectors[0]))
```

## How this connects to ChromaDB
Chroma collections accept an embedding function. We’ll later pass the same embedder/embedding function into the Chroma collection so:
- The same model is used for indexing and querying.
- You avoid inconsistent vector dimensions.

## Next step ideas
- ChromaDB vector store integration (add/store chunks)
- Persistence config (disk vs memory)
- Retrieval pipeline (query → top‑k chunks)
