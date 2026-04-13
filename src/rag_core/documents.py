from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
from uuid import uuid4


@dataclass(frozen=True)
class Document:
    """Simple document container for RAG pipelines."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid4()))

    def with_metadata(self, **updates: Any) -> "Document":
        merged = dict(self.metadata)
        merged.update(updates)
        return Document(text=self.text, metadata=merged, doc_id=self.doc_id)
