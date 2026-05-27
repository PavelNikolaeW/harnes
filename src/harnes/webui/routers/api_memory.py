"""Memory explorer: episodic / semantic / world. См. § 13 архитектуры."""
from __future__ import annotations

import json as jsonlib

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from harnes.memory.episodic import EpisodicStore, extract_terms
from harnes.memory.schema import MemoryType
from harnes.memory.semantic import SemanticStore
from harnes.memory.world import WorldModelStore
from harnes.webui.deps import (
    get_episodic,
    get_semantic,
    get_world,
)
from harnes.webui.templating import templates

router = APIRouter()


@router.get("", response_class=HTMLResponse)
def memory_explorer(
    request: Request,
    q: str = "",
    type_: str = "episodic",  # episodic | semantic | world
    k: int = 20,
    episodic: EpisodicStore = Depends(get_episodic),
    semantic: SemanticStore | None = Depends(get_semantic),
    world: WorldModelStore | None = Depends(get_world),
) -> HTMLResponse:
    """Поиск по выбранному backend'у. Пустой query → recent / placeholder."""
    k = max(1, min(k, 200))
    if type_ not in ("episodic", "semantic", "world"):
        raise HTTPException(400, "type must be one of: episodic, semantic, world")

    results: list = []
    error: str | None = None
    terms: list[str] = []

    if type_ == "episodic":
        terms = extract_terms(q) if q else []
        if terms:
            raw = episodic.search_steps_by_terms(terms=terms, limit=k)
        else:
            raw = episodic.recent_steps(limit=k)
        for r in raw:
            try:
                parsed = jsonlib.loads(r.get("content_json", "{}"))
            except jsonlib.JSONDecodeError:
                parsed = {"_raw": r.get("content_json", "")}
            results.append({**r, "content": parsed})

    elif type_ == "semantic":
        if semantic is None:
            error = "Qdrant недоступен (semantic store не подключён)"
        elif not q.strip():
            results = []
        else:
            try:
                from harnes.llm.embeddings import embed

                vec = embed([q])
                if not vec:
                    error = "embeddings backend не вернул вектор"
                else:
                    results = semantic.search(query_vector=vec[0], k=k)
            except Exception as exc:
                error = f"semantic search failed: {exc}"

    elif type_ == "world":
        if world is None:
            error = "Neo4j/Graphiti недоступен (world model не подключён)"
        elif not q.strip():
            results = []
        else:
            try:
                results = world.search(query=q, k=k)
            except Exception as exc:
                error = f"world search failed: {exc}"

    return templates.TemplateResponse(
        request,
        "memory/explorer.html",
        {
            "q": q,
            "type_": type_,
            "k": k,
            "results": results,
            "terms": terms,
            "error": error,
            "semantic_available": semantic is not None,
            "world_available": world is not None,
            "memory_types": [t.value for t in MemoryType],
        },
    )
