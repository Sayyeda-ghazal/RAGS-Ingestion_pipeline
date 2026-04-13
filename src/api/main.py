from __future__ import annotations

from fastapi import FastAPI

from .routers import health, ingest, query, upload


def create_app() -> FastAPI:
    app = FastAPI(title="RAG API", version="0.1.0")

    app.include_router(health)
    app.include_router(ingest)
    app.include_router(upload)
    app.include_router(query)

    return app


app = create_app()
