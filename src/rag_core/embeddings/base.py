from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Base interface for embedding text into vectors."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into vectors."""

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text into a vector."""
        return self.embed_texts([text])[0]
