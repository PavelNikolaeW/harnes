"""Webui smoke tests: страницы рендерятся, статус-коды разумные.

Цель — поймать regressions уровня "lifespan не стартует", "template не
найден", "router.include_router забыл prefix". Содержание страниц
почти не проверяем.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_dashboard_renders(client: TestClient) -> None:
    """GET /dashboard → 200, тело упоминает имя агента или 'dashboard'."""
    r = client.get("/dashboard")
    assert r.status_code == 200
    body = r.text.lower()
    # base.html имеет 'harnes · Irida', dashboard.html — title 'dashboard'
    assert any(token in body for token in ("harnes", "irida", "dashboard"))


def test_root_redirects(client: TestClient) -> None:
    """GET / → 307 → /dashboard. follow_redirects=False обязательно."""
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"] == "/dashboard"


@pytest.mark.parametrize(
    "path",
    [
        "/goals",
        "/trajectories",
        "/journal",
        "/memory",
        "/skills",
        "/eval",
        "/llm",
        "/cost",
        "/self-gen",
        "/reflect",
        "/standing",
        "/memory/volumes",
        "/commands",
        "/config",
        "/health",
    ],
)
def test_all_pages_respond_2xx_or_5xx(client: TestClient, path: str) -> None:
    """Каждая страница навбара должна отвечать.

    200 — нормально. 503 — store недоступен (Neo4j/Qdrant down), и роутер
    корректно бросил HTTPException(503). Любой 5xx-без-503 — провал.
    """
    r = client.get(path)
    assert r.status_code in (200, 503), (
        f"{path} → {r.status_code}; body[:300]={r.text[:300]}"
    )


def test_404_on_invalid_goal_id(client: TestClient) -> None:
    """GET /goals/<bad-uuid> → 404 (NOT 500)."""
    r = client.get("/goals/not-a-uuid")
    assert r.status_code == 404


def test_health_page(client: TestClient) -> None:
    """GET /health → 200, имеет 'backend' (заголовок 'Backend health')."""
    r = client.get("/health")
    assert r.status_code == 200
    assert "backend" in r.text.lower()
