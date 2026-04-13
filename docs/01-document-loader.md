# Step 01: Document Loader (RAG Core)

This step creates the foundation for ingesting raw files into a RAG pipeline. We defined a **Document** object plus a set of **loaders** that can read files and return a list of `Document` instances.

## Why this step exists
A RAG system needs a consistent “document” format so later steps (chunking, embedding, indexing) can treat all inputs the same way. Loaders turn files into that format.

## Files created in this step
- `src/rag_core/documents.py`
- `src/rag_core/__init__.py`
- `src/rag_core/loaders/base.py`
- `src/rag_core/loaders/text.py`
- `src/rag_core/loaders/pdf.py`
- `src/rag_core/loaders/directory.py`
- `src/rag_core/loaders/__init__.py`

## What each file does

### `src/rag_core/documents.py`
Defines the **Document** dataclass used across the pipeline.

Key fields:
- `text`: the raw text content.
- `metadata`: a dictionary with useful info (file name, source path, etc.).
- `doc_id`: a unique ID (UUID string) so each document is traceable.

It also includes `with_metadata(...)` so we can add/override metadata without mutating the original document.

### `src/rag_core/loaders/base.py`
Defines the base interface for all loaders.

- `BaseLoader.load()`: must return a list of `Document` objects.
- `BaseLoader.lazy_load()`: yields documents lazily; by default it just wraps `load()`.
- `LoaderError`: a custom error type so ingestion failures are clear and consistent.

### `src/rag_core/loaders/text.py`
Loads plain text files (`.txt`, `.md`, `.rst`):

1. Verifies the file exists and is a regular file.
2. Reads the file using a specified encoding (default `utf-8`).
3. Returns a single `Document` with metadata:
   - `source` (full path)
   - `file_name`
   - `file_ext`

### `src/rag_core/loaders/pdf.py`
Loads PDF files using the optional `pypdf` library.

1. Checks file exists.
2. Tries to import `pypdf.PdfReader`.
   - If missing, raises `LoaderError` with a clear message.
3. Extracts text from all pages and joins them with newlines.
4. Returns a `Document` with metadata plus `page_count`.

### `src/rag_core/loaders/directory.py`
Loads an entire directory of files and delegates to the correct loader based on file extension.

- Default supported extensions: `.txt`, `.md`, `.rst`, `.pdf`.
- `recursive=True` by default, so it walks subfolders.
- `allowed_extensions` can be used to restrict ingestion.
- `loader_map` lets you override/extend supported file types.

Behavior:
1. Walk the directory and collect files (skipping hidden files).
2. For each file, choose the right loader by extension.
3. Aggregate all `Document` objects from each loader.

### `src/rag_core/loaders/__init__.py`
Exports the public loader API so you can import directly from `rag_core.loaders`.

## Example usage
```python
from rag_core.loaders import DirectoryLoader, TextLoader

# Load all supported files in a folder
all_docs = DirectoryLoader("data", recursive=True).load()

# Load a single text file
single = TextLoader("data/readme.txt").load()
```

## Why this design is modular
- Each file type has its own loader, so adding new formats (e.g., `.docx`) is easy.
- All loaders implement the same interface, so the rest of the pipeline doesn’t care about file type.
- `DirectoryLoader` acts as an orchestration layer, not a parser itself.

## Next step ideas
- Chunking (split documents into smaller pieces)
- Embeddings (create vectors)
- FAISS index + persistence
