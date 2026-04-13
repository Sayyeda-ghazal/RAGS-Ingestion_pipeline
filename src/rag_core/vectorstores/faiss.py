from __future__ import annotations

from typing import Iterable, List, Optional

from rag_core.documents import Document

from .base import BaseVectorStore, SearchResult


class FaissVectorStore(BaseVectorStore):
    """In-memory FAISS vector store (local)."""

    def __init__(self, normalize: bool = True) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'faiss-cpu'. Install it to use FaissVectorStore."
            ) from exc

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'numpy'. Install it to use FaissVectorStore."
            ) from exc

        self.faiss = faiss
        self.np = np
        self.normalize = normalize
        self.index: Optional[faiss.Index] = None
        self.documents: List[Document] = []

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.index = self.faiss.IndexFlatIP(dim) if self.normalize else self.faiss.IndexFlatL2(dim)

    def _to_array(self, vectors: List[List[float]]) -> "np.ndarray":
        array = self.np.array(vectors, dtype="float32")
        if self.normalize:
            self.faiss.normalize_L2(array)
        return array

    def add(self, documents: Iterable[Document], vectors: List[List[float]]) -> None:
        docs = list(documents)
        if len(docs) != len(vectors):
            raise ValueError("documents and vectors must be the same length")
        if not vectors:
            return

        array = self._to_array(vectors)
        self._ensure_index(array.shape[1])
        assert self.index is not None

        self.index.add(array)
        self.documents.extend(docs)

    def similarity_search(self, query_vector: List[float], k: int = 4) -> List[SearchResult]:
        if self.index is None or not self.documents:
            return []

        query = self._to_array([query_vector])
        assert self.index is not None
        distances, indices = self.index.search(query, k)

        results: List[SearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append(SearchResult(document=self.documents[idx], score=float(score)))
        return results
