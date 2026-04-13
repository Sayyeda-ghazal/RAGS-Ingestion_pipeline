from .health import router as health
from .ingest import router as ingest
from .query import router as query
from .upload import router as upload

__all__ = ["health", "ingest", "query", "upload"]
