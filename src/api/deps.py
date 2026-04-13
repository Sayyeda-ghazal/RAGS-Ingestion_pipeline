from __future__ import annotations

import os
from functools import lru_cache

from .services.rag_service import RAGService, build_service


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    vector_backend = os.getenv("RAG_VECTOR_BACKEND", "chroma")
    persist_directory = os.getenv("RAG_CHROMA_PERSIST", "./chroma_db")
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

    return build_service(
        vector_backend=vector_backend,
        persist_directory=persist_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
