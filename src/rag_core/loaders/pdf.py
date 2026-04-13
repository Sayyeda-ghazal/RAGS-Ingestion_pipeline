from __future__ import annotations

from pathlib import Path
from typing import List

from rag_core.documents import Document

from .base import BaseLoader, LoaderError


class PdfLoader(BaseLoader):
    """Load PDFs using the optional pypdf dependency."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> List[Document]:
        if not self.path.exists():
            raise LoaderError(f"File not found: {self.path}")
        if not self.path.is_file():
            raise LoaderError(f"Not a file: {self.path}")

        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise LoaderError(
                "Missing optional dependency 'pypdf'. "
                "Install it to enable PDF loading."
            ) from exc

        try:
            reader = PdfReader(str(self.path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise LoaderError(f"Failed to parse PDF: {self.path}") from exc

        return [
            Document(
                text=text,
                metadata={
                    "source": str(self.path),
                    "file_name": self.path.name,
                    "file_ext": self.path.suffix.lower(),
                    "page_count": len(reader.pages),
                },
            )
        ]
