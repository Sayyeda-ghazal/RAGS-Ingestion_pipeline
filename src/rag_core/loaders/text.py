from __future__ import annotations

from pathlib import Path
from typing import List

from rag_core.documents import Document

from .base import BaseLoader, LoaderError


class TextLoader(BaseLoader):
    """Load plain text-like files (txt, md, rst)."""

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def load(self) -> List[Document]:
        if not self.path.exists():
            raise LoaderError(f"File not found: {self.path}")
        if not self.path.is_file():
            raise LoaderError(f"Not a file: {self.path}")
        try:
            text = self.path.read_text(encoding=self.encoding)
        except UnicodeDecodeError as exc:
            raise LoaderError(
                f"Failed to decode {self.path} with {self.encoding}"
            ) from exc

        return [
            Document(
                text=text,
                metadata={
                    "source": str(self.path),
                    "file_name": self.path.name,
                    "file_ext": self.path.suffix.lower(),
                },
            )
        ]
