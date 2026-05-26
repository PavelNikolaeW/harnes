"""LLM client — обёртка над LiteLLM.

Chat completions: call / async_call / health_check.
Embeddings: embed (primary через /v1/embeddings, fallback на fastembed).

См. `agent_architecture.html` § 17.
"""
from harnes.llm.client import async_call, call, health_check
from harnes.llm.embeddings import embed

__all__ = ["async_call", "call", "embed", "health_check"]
