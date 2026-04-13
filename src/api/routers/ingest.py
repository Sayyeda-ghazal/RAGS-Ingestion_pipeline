from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_core.documents import Document

from ..deps import get_rag_service
from ..models import IngestRequest, IngestResponse
from ..services.rag_service import RAGService

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def ingest_documents(
    request: IngestRequest,
    service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    documents = [Document(text=item.text, metadata=item.metadata) for item in request.documents]
    chunk_count = service.ingest(documents)
    return IngestResponse(
        documents_received=len(documents),
        chunks_created=chunk_count,
    )
