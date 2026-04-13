from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from rag_core.documents import Document


class BaseChunker(ABC):
    """Base class for text chunkers."""

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split raw text into string chunks."""

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents into chunked documents with inherited metadata."""
        chunked: List[Document] = []
        for doc in documents:
            for index, chunk in enumerate(self.split_text(doc.text)):
                chunked.append(
                    Document(
                        text=chunk,
                        metadata={
                            **doc.metadata,
                            "parent_id": doc.doc_id,
                            "chunk_index": index,
                        },
                    )
                )
        return chunked
