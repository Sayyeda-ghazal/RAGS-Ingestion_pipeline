# Step 06: Generator (RAG Core)

This step adds the **generator**, which takes the user query plus retrieved chunks and asks an LLM to produce a final answer. The generator layer keeps prompt construction and model parameters centralized and reusable.

## What the LLM takes (all important inputs)
- `query`: the user’s question.
- `contexts`: the retrieved chunks (documents or search results).
- `system_prompt`: optional instructions that control tone/behavior.
- `temperature`: creativity vs determinism.
- `max_tokens`: response length cap.
- `model`: model name, if your LLM provider needs one.

## Files created in this step
- `src/rag_core/generators/base.py`
- `src/rag_core/generators/prompt.py`
- `src/rag_core/generators/simple.py`
- `src/rag_core/generators/__init__.py`

## What each file does

### `src/rag_core/generators/base.py`
Defines the generator interface and data structures:
- `GenerationRequest`: holds all inputs (query, contexts, prompts, params).
- `GenerationResult`: holds the answer plus the prompt messages used.
- `BaseGenerator`: interface with `generate(request)`.

### `src/rag_core/generators/prompt.py`
Builds chat-style messages for an LLM:
- Collects context texts.
- Inserts a system prompt.
- Builds a single user message that includes context + question.

### `src/rag_core/generators/simple.py`
Implements `CallableGenerator`, which delegates to any callable LLM client.

This keeps the RAG core provider‑agnostic. Later, we can plug in OpenAI, local models, or any API by writing a small wrapper callable.

### `src/rag_core/generators/__init__.py`
Exports the generator API for clean imports.

## Example usage (provider‑agnostic)
```python
from rag_core.generators import CallableGenerator, GenerationRequest

# Example callable shape
# def llm(messages: list[dict], params: dict) -> str:
#     ... call your provider ...
#     return response_text

generator = CallableGenerator(llm_callable=llm)

request = GenerationRequest(
    query="What is RAG?",
    contexts=results,
    system_prompt="You are a precise technical assistant.",
    temperature=0.2,
    max_tokens=300,
    model="your-model-name",
)

response = generator.generate(request)
print(response.answer)
```

## Next step ideas
- Provider integration (OpenAI or local model wrapper)
- End‑to‑end RAG pipeline (loader → chunker → embedder → store → retriever → generator)
- FastAPI endpoints for `/ingest` and `/query`
