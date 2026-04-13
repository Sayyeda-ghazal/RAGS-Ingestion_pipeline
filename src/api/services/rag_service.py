from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from rag_core.chunkers import BaseChunker, RecursiveCharacterChunker
from rag_core.documents import Document
from rag_core.embeddings import BaseEmbedder, ChromaEmbedder
from rag_core.generators import BaseGenerator, GenerationRequest
from rag_core.retrievers import VectorRetriever
from rag_core.vectorstores import BaseVectorStore, ChromaVectorStore, FaissVectorStore, SearchResult


class RAGService:
    """Orchestrates ingest + retrieval + optional generation."""

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        generator: Optional[BaseGenerator] = None,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = VectorRetriever(embedder=embedder, vector_store=vector_store)
        self.generator = generator

    def ingest(self, documents: Iterable[Document]) -> int:
        docs = list(documents)
        chunks = self.chunker.split_documents(docs)
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.add(chunks, vectors)
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> List[SearchResult]:
        return self.retriever.retrieve(query, k=k)

    def generate(
        self,
        query: str,
        contexts: Sequence[SearchResult],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Optional[str]:
        if not self.generator:
            return None

        request = GenerationRequest(
            query=query,
            contexts=contexts,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        result = self.generator.generate(request)
        return result.answer


def build_service(
    vector_backend: str = "chroma",
    persist_directory: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> RAGService:
    chunker = RecursiveCharacterChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embedder = ChromaEmbedder()

    if vector_backend.lower() == "faiss":
        vector_store: BaseVectorStore = FaissVectorStore()
    else:
        vector_store = ChromaVectorStore(
            collection_name="rag_chunks",
            persist_directory=persist_directory,
            embedder=embedder,
        )

    return RAGService(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        generator=None,
    )
