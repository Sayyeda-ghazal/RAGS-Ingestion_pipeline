from __future__ import annotations

from typing import List, Sequence

from rag_core.documents import Document
from rag_core.vectorstores import SearchResult

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the user. "
    "If the answer is not in the context, say you don't know."
)


def _extract_texts(contexts: Sequence[SearchResult | Document]) -> List[str]:
    texts: List[str] = []
    for item in contexts:
        if isinstance(item, SearchResult):
            texts.append(item.document.text)
        else:
            texts.append(item.text)
    return texts


def build_messages(
    query: str,
    contexts: Sequence[SearchResult | Document],
    system_prompt: str | None = None,
) -> List[dict]:
    """Build chat-style messages for an LLM."""
    system = system_prompt or DEFAULT_SYSTEM_PROMPT
    context_text = "\n\n---\n\n".join(_extract_texts(contexts))

    user_message = (
        "Answer the question using the context below.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
