from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rag_core.vectorstores import SearchResult


class BaseRetriever(ABC):
    """Base interface for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[SearchResult]:
        """Return top-k relevant documents for a query string."""
