# Step 02: Chunking (RAG Core)

This step introduces **chunking**, which splits large documents into smaller pieces. Chunking is crucial for retrieval because embedding models and vector search work best on smaller, focused segments.

## Why chunking matters
- Long documents dilute relevance in retrieval.
- Chunking improves semantic focus and recall.
- It lets us store and search at a finer granularity.

## Files created in this step
- `src/rag_core/chunkers/base.py`
- `src/rag_core/chunkers/recursive.py`
- `src/rag_core/chunkers/__init__.py`

## What each file does

### `src/rag_core/chunkers/base.py`
Defines the `BaseChunker` interface and a helper method to split entire `Document` objects.

Key ideas:
- `split_text(text)`: abstract method every chunker must implement.
- `split_documents(documents)`: loops through `Document` objects, splits each one, and returns new `Document` chunks.

When we split documents, each chunk keeps the original metadata and adds:
- `parent_id`: the original document’s ID
- `chunk_index`: position of the chunk within the document

### `src/rag_core/chunkers/recursive.py`
Implements `RecursiveCharacterChunker`, a simple but effective splitter:

- Uses separators in this order: paragraph → line → word → character.
- If a segment is still too large, it falls back to a smaller separator.
- Merges segments into chunks up to `chunk_size`.
- Optionally adds `chunk_overlap` to preserve context between chunks.

Defaults:
- `chunk_size=1000`
- `chunk_overlap=200`
- `separators=["\n\n", "\n", " ", ""]`

### `src/rag_core/chunkers/__init__.py`
Exports the chunker API so you can import cleanly from `rag_core.chunkers`.

## Example usage
```python
from rag_core.chunkers import RecursiveCharacterChunker
from rag_core.loaders import TextLoader

loader = TextLoader("data/readme.txt")
documents = loader.load()

chunker = RecursiveCharacterChunker(chunk_size=800, chunk_overlap=100)
chunks = chunker.split_documents(documents)

print(len(chunks))
print(chunks[0].metadata)
```

## How overlap works
If `chunk_overlap` is non-zero, each chunk (after the first) repeats a short tail from the previous chunk to preserve continuity. This helps embeddings capture context that might otherwise be split across boundaries.

## Next step ideas
- Embeddings (convert chunks into vectors)
- Vector store (FAISS indexing + persistence)
- Retrieval pipeline (search + rerank)
