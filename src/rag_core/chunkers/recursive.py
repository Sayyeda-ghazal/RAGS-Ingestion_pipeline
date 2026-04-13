from __future__ import annotations

from typing import Iterable, List

from .base import BaseChunker


class RecursiveCharacterChunker(BaseChunker):
    """Split text into chunks using a separator fall-through strategy."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Iterable[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = list(separators or ["\n\n", "\n", " ", ""])

    def split_text(self, text: str) -> List[str]:
        cleaned = text.strip()
        if not cleaned:
            return []

        segments = self._split_by_separators(cleaned, self.separators)
        return self._merge_segments(segments)

    def _split_by_separators(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return [text]

        sep = separators[0]
        if sep == "":
            return list(text)

        parts = text.split(sep)
        if len(parts) == 1:
            return self._split_by_separators(text, separators[1:])

        results: List[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                results.append(part)
            else:
                results.extend(self._split_by_separators(part, separators[1:]))
        return results

    def _merge_segments(self, segments: List[str]) -> List[str]:
        if not segments:
            return []

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for segment in segments:
            seg_len = len(segment)
            if current and current_len + seg_len + 1 > self.chunk_size:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0

            current.append(segment)
            current_len += seg_len + 1

        if current:
            chunks.append(" ".join(current).strip())

        if self.chunk_overlap > 0 and len(chunks) > 1:
            return self._apply_overlap(chunks)

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        overlapped: List[str] = []
        for index, chunk in enumerate(chunks):
            if index == 0:
                overlapped.append(chunk)
                continue

            prev_tail = chunks[index - 1][-self.chunk_overlap :]
            combined = f"{prev_tail} {chunk}".strip()
            overlapped.append(combined)

        return overlapped
