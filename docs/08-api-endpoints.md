# Step 08: API Endpoints (Phase 2)

This step adds explicit endpoints requested for Phase 2:
- `POST /upload-doc`
- `POST /ask`
- `GET /health`

## Files touched in this step
- `src/api/routers/upload.py`
- `src/api/routers/query.py`
- `src/api/routers/__init__.py`
- `src/api/main.py`

## What changed

### `POST /upload-doc`
Accepts file uploads (TXT/MD/RST/PDF) and ingests them into the vector store.

Behavior:
- Reads each file from the request.
- Text files are decoded directly.
- PDF files are temporarily saved and parsed with `PdfLoader`.
- All extracted text is chunked, embedded, and stored.

### `POST /ask`
A convenience alias for `/query` that returns the top‑k relevant chunks (and generation output if enabled later).

### `GET /health`
Basic health check endpoint.

## Example usage

### Upload
```bash
curl -X POST "http://localhost:8000/upload-doc" \
  -F "files=@./data/readme.txt" \
  -F "files=@./data/guide.pdf"
```

### Ask
```json
POST /ask
{
  "query": "What is RAG?",
  "k": 5,
  "include_context": true
}
```
