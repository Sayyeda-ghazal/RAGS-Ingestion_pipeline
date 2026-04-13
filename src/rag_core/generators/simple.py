from __future__ import annotations

from typing import Callable, Dict, List

from .base import BaseGenerator, GenerationRequest, GenerationResult
from .prompt import build_messages


class CallableGenerator(BaseGenerator):
    """Generator that delegates to a callable LLM function."""

    def __init__(self, llm_callable: Callable[[List[dict], Dict], str]) -> None:
        self.llm_callable = llm_callable

    def generate(self, request: GenerationRequest) -> GenerationResult:
        messages = build_messages(
            query=request.query,
            contexts=request.contexts,
            system_prompt=request.system_prompt,
        )

        params: Dict = {}
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens
        if request.model is not None:
            params["model"] = request.model

        answer = self.llm_callable(messages, params)
        return GenerationResult(answer=answer, prompt_messages=messages)
