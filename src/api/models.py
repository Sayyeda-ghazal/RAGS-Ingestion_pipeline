from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: List[DocumentIn]


class IngestResponse(BaseModel):
    documents_received: int
    chunks_created: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(4, ge=1, le=50)
    include_context: bool = True
    system_prompt: Optional[str] = None
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1)
    model: Optional[str] = None


class ContextItem(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    contexts: List[ContextItem] = Field(default_factory=list)
