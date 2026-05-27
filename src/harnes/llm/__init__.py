"""LLM client — обёртка над LiteLLM.

Chat completions: call / async_call / health_check / is_router_reachable.
Embeddings: embed (primary через /v1/embeddings, fallback на fastembed).

См. `agent_architecture.html` § 17, `docs/router_roadmap.md`.
"""
from harnes.llm.client import (
    async_call,
    call,
    get_router_load_status,
    health_check,
    is_router_reachable,
    warn_if_models_on_cpu,
)
from harnes.llm.embeddings import embed, reset_server_state

__all__ = [
    "async_call",
    "call",
    "embed",
    "get_router_load_status",
    "health_check",
    "is_router_reachable",
    "reset_server_state",
    "warn_if_models_on_cpu",
]
