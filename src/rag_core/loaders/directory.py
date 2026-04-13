from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Type

from rag_core.documents import Document

from .base import BaseLoader, LoaderError
from .pdf import PdfLoader
from .text import TextLoader


class DirectoryLoader(BaseLoader):
    """Load a directory of documents using extension-based loaders."""

    DEFAULT_LOADERS: Dict[str, Type[BaseLoader]] = {
        ".txt": TextLoader,
        ".md": TextLoader,
        ".rst": TextLoader,
        ".pdf": PdfLoader,
    }

    def __init__(
        self,
        path: str | Path,
        recursive: bool = True,
        allowed_extensions: Sequence[str] | None = None,
        loader_map: Dict[str, Type[BaseLoader]] | None = None,
    ) -> None:
        self.path = Path(path)
        self.recursive = recursive
        self.allowed_extensions = (
            [ext.lower() for ext in allowed_extensions]
            if allowed_extensions
            else None
        )
        self.loader_map = loader_map or dict(self.DEFAULT_LOADERS)

    def _iter_files(self) -> Iterable[Path]:
        if not self.path.exists():
            raise LoaderError(f"Directory not found: {self.path}")
        if not self.path.is_dir():
            raise LoaderError(f"Not a directory: {self.path}")

        pattern = "**/*" if self.recursive else "*"
        for path in self.path.glob(pattern):
            if path.is_file() and not path.name.startswith("."):
                yield path

    def load(self) -> List[Document]:
        documents: List[Document] = []
        for path in self._iter_files():
            suffix = path.suffix.lower()
            if self.allowed_extensions and suffix not in self.allowed_extensions:
                continue

            loader_cls = self.loader_map.get(suffix)
            if not loader_cls:
                continue

            loader = loader_cls(path)
            try:
                documents.extend(loader.load())
            except LoaderError:
                raise
            except Exception as exc:
                raise LoaderError(f"Failed to load {path}") from exc

        return documents
