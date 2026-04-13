from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_rag_service
from ..models import ContextItem, QueryRequest, QueryResponse
from ..services.rag_service import RAGService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query_rag(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    results = service.retrieve(request.query, k=request.k)

    answer = service.generate(
        query=request.query,
        contexts=results,
        system_prompt=request.system_prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        model=request.model,
    )

    contexts = []
    if request.include_context:
        for item in results:
            contexts.append(
                ContextItem(
                    text=item.document.text,
                    metadata=item.document.metadata,
                    score=item.score,
                )
            )

    return QueryResponse(query=request.query, answer=answer, contexts=contexts)


@router.post("/ask", response_model=QueryResponse)
def ask_rag(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    return query_rag(request=request, service=service)
