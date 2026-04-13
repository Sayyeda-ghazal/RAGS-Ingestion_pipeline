from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from rag_core.documents import Document
from rag_core.vectorstores import SearchResult


@dataclass(frozen=True)
class GenerationRequest:
    query: str
    contexts: Sequence[SearchResult | Document]
    system_prompt: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    model: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    prompt_messages: List[dict]


class BaseGenerator(ABC):
    """Base interface for LLM-based generators."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate an answer from a request."""
