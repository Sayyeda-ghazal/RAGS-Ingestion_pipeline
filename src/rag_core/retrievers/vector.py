from __future__ import annotations

from typing import List

from rag_core.embeddings import BaseEmbedder
from rag_core.vectorstores import BaseVectorStore, SearchResult

from .base import BaseRetriever


class VectorRetriever(BaseRetriever):
    """Retriever that embeds the query and does vector similarity search."""

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 4) -> List[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        return self.vector_store.similarity_search(query_vector, k=k)
