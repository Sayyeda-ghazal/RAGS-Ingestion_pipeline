from __future__ import annotations

from typing import Callable, List, Optional

from .base import BaseEmbedder


class ChromaEmbedder(BaseEmbedder):
    """Adapter around ChromaDB embedding functions."""

    def __init__(
        self,
        embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None,
        provider: str = "sentence-transformers",
        **provider_kwargs,
    ) -> None:
        """
        If embedding_function is provided, it is used directly.

        Otherwise, a built-in Chroma embedding function is created using `provider`.
        Supported providers (via chromadb.utils.embedding_functions):
        - "sentence-transformers"
        - "openai"
        """
        if embedding_function is not None:
            self.embedding_function = embedding_function
            return

        try:
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'chromadb'. Install it to use ChromaEmbedder."
            ) from exc

        provider = provider.lower()
        if provider in {"sentence-transformers", "sentence_transformers", "st"}:
            model_name = provider_kwargs.get("model_name", "all-MiniLM-L6-v2")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        elif provider in {"openai", "openai-embeddings"}:
            api_key = provider_kwargs.get("api_key")
            model_name = provider_kwargs.get("model_name", "text-embedding-3-small")
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=model_name,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embedding_function(texts)
