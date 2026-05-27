"""Memory explorer: episodic / semantic / world. См. § 13 архитектуры."""
from __future__ import annotations

import json as jsonlib

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from harnes.config import Settings
from harnes.memory.episodic import EpisodicStore, extract_terms
from harnes.memory.schema import MemoryType
from harnes.memory.semantic import SemanticStore
from harnes.memory.world import WorldModelStore
from harnes.webui.deps import (
    get_agent_settings,
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


# ---------- World KG graph visualization (Cytoscape) ----------


def _neo4j_driver(settings: Settings):
    """Read-only neo4j driver — отдельный от Graphiti, чтобы не таскать
    его lazy-init Pipeline. Закрывается caller'ом."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        settings.memory.neo4j_uri,
        auth=(settings.memory.neo4j_user, settings.memory.neo4j_password),
    )


def _cytoscape_payload(driver, limit: int = 200) -> dict:
    """Cypher → Cytoscape.js graph format.

    LIMIT — на каждый side (узлы и рёбра) — браузер не тянет графы >1000 без
    сильной оптимизации. Возвращает {"nodes": [...], "edges": [...]}.
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    with driver.session() as s:
        result = s.run(
            "MATCH (n) RETURN n, elementId(n) AS eid, labels(n) AS lbls LIMIT $lim",
            lim=limit,
        )
        for rec in result:
            n = rec["n"]
            nid = str(rec["eid"])
            props = dict(n)
            label = (rec["lbls"] or ["Node"])[0]
            display = (
                props.get("name") or props.get("fact") or
                props.get("summary") or label
            )
            nodes[nid] = {
                "data": {
                    "id": nid,
                    "label": label,
                    "display": str(display)[:80],
                    "properties": {k: str(v)[:200] for k, v in props.items()},
                }
            }
        result = s.run(
            "MATCH (a)-[r]->(b) "
            "RETURN elementId(a) AS sid, elementId(b) AS tid, type(r) AS rel, "
            "       elementId(r) AS eid, properties(r) AS rprops "
            "LIMIT $lim",
            lim=limit,
        )
        for rec in result:
            sid = str(rec["sid"])
            tid = str(rec["tid"])
            # Если узлы за пределами node LIMIT, добавим placeholder.
            for nid in (sid, tid):
                if nid not in nodes:
                    nodes[nid] = {
                        "data": {
                            "id": nid, "label": "Node", "display": "…",
                            "properties": {},
                        }
                    }
            edges.append({
                "data": {
                    "id": str(rec["eid"]),
                    "source": sid,
                    "target": tid,
                    "label": rec["rel"],
                    "properties": {k: str(v)[:200] for k, v in (rec["rprops"] or {}).items()},
                }
            })
    return {"nodes": list(nodes.values()), "edges": edges}


@router.get("/world/graph", response_class=HTMLResponse)
def world_graph_page(
    request: Request,
    world: WorldModelStore | None = Depends(get_world),
) -> HTMLResponse:
    """Cytoscape-страница для KG. Данные тянутся из /memory/world/cytoscape.json."""
    return templates.TemplateResponse(
        request,
        "memory/world_graph.html",
        {
            "world_available": world is not None,
        },
    )


@router.get("/world/cytoscape.json")
def world_cytoscape_json(
    limit: int = 200,
    settings: Settings = Depends(get_agent_settings),
    world: WorldModelStore | None = Depends(get_world),
) -> JSONResponse:
    """JSON-endpoint для Cytoscape.js. Direct Cypher через neo4j-driver."""
    if world is None:
        raise HTTPException(503, "Neo4j/Graphiti недоступен")
    limit = max(1, min(limit, 1000))
    try:
        driver = _neo4j_driver(settings)
    except Exception as exc:
        raise HTTPException(503, f"neo4j driver init failed: {exc}")
    try:
        payload = _cytoscape_payload(driver, limit=limit)
    except Exception as exc:
        raise HTTPException(500, f"cypher query failed: {exc}")
    finally:
        try:
            driver.close()
        except Exception:
            pass
    return JSONResponse(payload)
