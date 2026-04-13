from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from rag_core.documents import Document


class LoaderError(RuntimeError):
    """Raised when a loader fails to parse a document."""


class BaseLoader(ABC):
    """Base class for document loaders."""

    @abstractmethod
    def load(self) -> List[Document]:
        """Load all documents into memory."""

    def lazy_load(self) -> Iterable[Document]:
        """Lazily yield documents, defaulting to eager loading."""
        return iter(self.load())
