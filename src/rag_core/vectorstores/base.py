from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List

from rag_core.documents import Document


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float


class BaseVectorStore(ABC):
    """Base interface for vector stores."""

    @abstractmethod
    def add(self, documents: Iterable[Document], vectors: List[List[float]]) -> None:
        """Add documents and their vectors to the store."""

    @abstractmethod
    def similarity_search(self, query_vector: List[float], k: int = 4) -> List[SearchResult]:
        """Return top-k most similar documents for a query vector."""
