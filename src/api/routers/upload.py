from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from rag_core.documents import Document
from rag_core.loaders import LoaderError, PdfLoader

from ..deps import get_rag_service
from ..models import IngestResponse
from ..services.rag_service import RAGService

router = APIRouter(prefix="/upload-doc", tags=["ingest"])

TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS


def _load_file(upload: UploadFile) -> List[Document]:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    content = upload.file.read()
    if not content:
        return []

    if suffix in TEXT_EXTENSIONS:
        text = content.decode("utf-8", errors="ignore")
        return [
            Document(
                text=text,
                metadata={"source": upload.filename or "upload", "file_ext": suffix},
            )
        ]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        loader = PdfLoader(tmp_path)
        return loader.load()
    except LoaderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


@router.post("", response_model=IngestResponse)
def upload_documents(
    files: List[UploadFile] = File(...),
    service: RAGService = Depends(get_rag_service),
) -> IngestResponse:
    documents: List[Document] = []
    for upload in files:
        documents.extend(_load_file(upload))

    if not documents:
        return IngestResponse(documents_received=0, chunks_created=0)

    chunk_count = service.ingest(documents)
    return IngestResponse(
        documents_received=len(documents),
        chunks_created=chunk_count,
    )
