from __future__ import annotations

from typing import Iterable, List, Optional

from rag_core.documents import Document
from rag_core.embeddings import ChromaEmbedder

from .base import BaseVectorStore, SearchResult


class ChromaVectorStore(BaseVectorStore):
    """Chroma-backed vector store."""

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        persist_directory: Optional[str] = None,
        embedder: Optional[ChromaEmbedder] = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency 'chromadb'. Install it to use ChromaVectorStore."
            ) from exc

        self.embedder = embedder or ChromaEmbedder()

        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder.embedding_function,
        )

    def add(self, documents: Iterable[Document], vectors: List[List[float]]) -> None:
        docs = list(documents)
        if len(docs) != len(vectors):
            raise ValueError("documents and vectors must be the same length")

        ids = [doc.doc_id for doc in docs]
        texts = [doc.text for doc in docs]
        metadatas = [doc.metadata for doc in docs]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=vectors,
        )

    def similarity_search(self, query_vector: List[float], k: int = 4) -> List[SearchResult]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        output: List[SearchResult] = []
        for doc_text, metadata, doc_id, distance in zip(docs, metadatas, ids, distances):
            document = Document(text=doc_text, metadata=metadata or {}, doc_id=doc_id)
            score = float(distance) if distance is not None else 0.0
            output.append(SearchResult(document=document, score=score))

        return output
