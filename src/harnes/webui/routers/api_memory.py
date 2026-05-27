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


_DESTRUCTIVE_CYPHER = (
    "create", "delete", "remove", "set ", "merge", "drop", "load csv",
    "call dbms", "call db.create", "call apoc.create",
)


def _is_safe_read_cypher(snippet: str) -> bool:
    """Грубая защита от destructive операций в operator-вводе.

    Эвристика, не полная sandbox — operator localhost тулом, не публичный API.
    Достаточно отлавливать "ой, случайно вставил MERGE".
    """
    s = snippet.lower()
    return not any(kw in s for kw in _DESTRUCTIVE_CYPHER)


def _cytoscape_payload(
    driver,
    limit: int = 200,
    labels: list[str] | None = None,
    min_degree: int = 0,
) -> dict:
    """Cypher → Cytoscape.js graph format.

    LIMIT — на каждый side (узлы и рёбра); браузер не тянет графы >1k без
    сильной оптимизации. `labels` — фильтр узлов; `min_degree` — отбросить
    орбитальные узлы.
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    label_filter = ""
    if labels:
        # `WHERE ANY(lbl IN labels(n) WHERE lbl IN $labels)`
        label_filter = " WHERE ANY(lbl IN labels(n) WHERE lbl IN $labels) "
    with driver.session() as s:
        node_q = (
            f"MATCH (n) {label_filter}"
            "RETURN n, elementId(n) AS eid, labels(n) AS lbls, "
            "       size([(n)-[]-() | 1]) AS deg "
            "LIMIT $lim"
        )
        result = s.run(node_q, lim=limit, labels=labels or [])
        for rec in result:
            if min_degree > 0 and int(rec["deg"] or 0) < min_degree:
                continue
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
                    "degree": int(rec["deg"] or 0),
                    "properties": {k: str(v)[:200] for k, v in props.items()},
                }
            }
        # Edges — только между узлами, прошедшими фильтр.
        edge_q = (
            "MATCH (a)-[r]->(b) "
            "RETURN elementId(a) AS sid, elementId(b) AS tid, type(r) AS rel, "
            "       elementId(r) AS eid, properties(r) AS rprops "
            "LIMIT $lim"
        )
        result = s.run(edge_q, lim=limit)
        for rec in result:
            sid = str(rec["sid"])
            tid = str(rec["tid"])
            if sid not in nodes or tid not in nodes:
                # Узел отфильтрован — пропускаем ребро.
                continue
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


def _list_neo4j_labels(driver) -> list[str]:
    """Все используемые labels для фильтра в UI."""
    out: list[str] = []
    with driver.session() as s:
        try:
            for rec in s.run("CALL db.labels()"):
                out.append(rec["label"])
        except Exception:
            pass
    return sorted(out)


def _custom_cypher_payload(driver, snippet: str, limit: int = 200) -> dict:
    """Custom Cypher (read-only) → Cytoscape.

    Ожидается snippet который возвращает path / (a,r,b) / nodes/edges.
    Применяется guardrail (без destructive keywords).
    """
    if not _is_safe_read_cypher(snippet):
        raise HTTPException(
            400,
            "Cypher snippet содержит destructive keyword (CREATE/DELETE/SET/MERGE/REMOVE/DROP). "
            "Webui разрешает только read-only запросы.",
        )
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    with driver.session() as s:
        # Оборачиваем snippet в lazy CALL { ... } и берём первые $lim строк.
        wrapped = f"CALL {{ {snippet} }} RETURN * LIMIT $lim"
        result = s.run(wrapped, lim=limit)
        for rec in result:
            for value in rec.values():
                _walk_value(value, nodes, edges)
    return {"nodes": list(nodes.values()), "edges": edges}


def _walk_value(value, nodes: dict, edges: list) -> None:
    """Достать ноды/рёбра из любого Neo4j значения (Node/Relationship/Path/list)."""
    from neo4j.graph import Node, Path, Relationship

    if value is None:
        return
    if isinstance(value, Node):
        nid = str(value.element_id)
        if nid not in nodes:
            props = dict(value)
            label = (list(value.labels) or ["Node"])[0]
            display = (props.get("name") or props.get("fact") or
                       props.get("summary") or label)
            nodes[nid] = {
                "data": {
                    "id": nid, "label": label,
                    "display": str(display)[:80],
                    "properties": {k: str(v)[:200] for k, v in props.items()},
                }
            }
    elif isinstance(value, Relationship):
        sid = str(value.start_node.element_id)
        tid = str(value.end_node.element_id)
        for n in (value.start_node, value.end_node):
            _walk_value(n, nodes, edges)
        edges.append({
            "data": {
                "id": str(value.element_id),
                "source": sid, "target": tid,
                "label": value.type,
                "properties": {k: str(v)[:200] for k, v in dict(value).items()},
            }
        })
    elif isinstance(value, Path):
        for n in value.nodes:
            _walk_value(n, nodes, edges)
        for r in value.relationships:
            _walk_value(r, nodes, edges)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _walk_value(v, nodes, edges)


@router.get("/world/graph", response_class=HTMLResponse)
def world_graph_page(
    request: Request,
    settings: Settings = Depends(get_agent_settings),
    world: WorldModelStore | None = Depends(get_world),
) -> HTMLResponse:
    """Cytoscape-страница для KG. Данные тянутся из /memory/world/cytoscape.json."""
    available_labels: list[str] = []
    if world is not None:
        try:
            driver = _neo4j_driver(settings)
            available_labels = _list_neo4j_labels(driver)
            driver.close()
        except Exception:
            available_labels = []
    return templates.TemplateResponse(
        request,
        "memory/world_graph.html",
        {
            "world_available": world is not None,
            "available_labels": available_labels,
        },
    )


@router.get("/world/cytoscape.json")
def world_cytoscape_json(
    limit: int = 200,
    labels: str = "",
    min_degree: int = 0,
    cypher: str = "",
    settings: Settings = Depends(get_agent_settings),
    world: WorldModelStore | None = Depends(get_world),
) -> JSONResponse:
    """JSON-endpoint для Cytoscape.js.

    Параметры:
    - `labels` — CSV-список Neo4j-labels для фильтра ("Episodic,Entity").
    - `min_degree` — отбросить узлы с degree < min_degree.
    - `cypher` — read-only snippet, перевешивает остальные фильтры.
    """
    if world is None:
        raise HTTPException(503, "Neo4j/Graphiti недоступен")
    limit = max(1, min(limit, 1000))
    label_list = [s.strip() for s in labels.split(",") if s.strip()] or None
    min_degree = max(0, min_degree)
    try:
        driver = _neo4j_driver(settings)
    except Exception as exc:
        raise HTTPException(503, f"neo4j driver init failed: {exc}")
    try:
        if cypher.strip():
            payload = _custom_cypher_payload(driver, cypher.strip(), limit=limit)
        else:
            payload = _cytoscape_payload(
                driver, limit=limit, labels=label_list, min_degree=min_degree,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"cypher query failed: {exc}")
    finally:
        try:
            driver.close()
        except Exception:
            pass
    return JSONResponse(payload)
