"""Tests for is_router_reachable — дешёвый precheck доступности роутера.

См. docs/router_roadmap.md R2.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from harnes.llm import is_router_reachable
from harnes.llm import client as llm_client


# ---------- httpx.Client.get моки ----------


class _MockResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _patch_client_get(monkeypatch: pytest.MonkeyPatch, get_impl) -> None:
    """Подменяет httpx.Client.get на наш callable."""
    monkeypatch.setattr(
        "httpx.Client.get",
        get_impl,
    )


# ---------- Health endpoint ВЕРНУЛ 200 ----------


def test_reachable_when_health_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """GET /health → 200 — immediately True, /models не дёргается."""
    calls = []

    def fake_get(self, url, **kwargs):
        calls.append(url)
        if url.endswith("/health"):
            return _MockResponse(200)
        raise AssertionError(f"должен был остановиться на /health, дёрнул: {url}")

    _patch_client_get(monkeypatch, fake_get)

    assert is_router_reachable("http://router.test:8000/v1") is True
    assert calls == ["http://router.test:8000/health"]


# ---------- Health 404 → fallback на /models 200 ----------


def test_reachable_via_models_when_health_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """Сегодняшняя реальность: /health не реализован → /v1/models 200 → True."""
    calls = []

    def fake_get(self, url, **kwargs):
        calls.append(url)
        if url.endswith("/health"):
            return _MockResponse(404)
        if url.endswith("/models"):
            return _MockResponse(200)
        raise AssertionError(f"unexpected url: {url}")

    _patch_client_get(monkeypatch, fake_get)

    assert is_router_reachable("http://router.test:8000/v1") is True
    assert calls == [
        "http://router.test:8000/health",
        "http://router.test:8000/v1/models",
    ]


# ---------- Оба endpoint'а в ошибке ----------


def test_unreachable_when_both_endpoints_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection refused на обоих → False."""

    def fake_get(self, url, **kwargs):
        raise httpx.ConnectError("Connection refused")

    _patch_client_get(monkeypatch, fake_get)

    assert is_router_reachable("http://router.test:8000/v1") is False


def test_unreachable_when_models_returns_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если /health 404 и /models даёт 503 — False."""
    def fake_get(self, url, **kwargs):
        if url.endswith("/health"):
            return _MockResponse(404)
        return _MockResponse(503)

    _patch_client_get(monkeypatch, fake_get)

    assert is_router_reachable("http://router.test:8000/v1") is False


# ---------- Timeout ----------


def test_unreachable_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """httpx.TimeoutException → False, не raise."""
    def fake_get(self, url, **kwargs):
        raise httpx.ReadTimeout("timed out")

    _patch_client_get(monkeypatch, fake_get)

    assert is_router_reachable("http://router.test:8000/v1", timeout_s=0.1) is False


# ---------- api_base из settings ----------


def test_uses_settings_api_base_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Без аргумента — берёт api_base из settings."""
    captured_urls = []

    def fake_get(self, url, **kwargs):
        captured_urls.append(url)
        return _MockResponse(200)

    _patch_client_get(monkeypatch, fake_get)

    is_router_reachable()
    settings = llm_client.get_settings()
    # Хотя бы первый URL должен быть на корне settings.llm.api_base
    assert captured_urls[0].startswith(settings.llm.api_base.split("/v1")[0])


# ---------- URL-конструирование: суффикс /v1 ----------


def test_health_url_strips_v1_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_base заканчивается на /v1 → /health на корне, не /v1/health."""
    urls = []
    def fake_get(self, url, **kwargs):
        urls.append(url)
        return _MockResponse(200 if url.endswith("/health") else 404)

    _patch_client_get(monkeypatch, fake_get)

    is_router_reachable("http://192.168.0.111:8000/v1")
    assert urls[0] == "http://192.168.0.111:8000/health"


def test_models_url_keeps_v1_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    """/v1/models не теряет суффикс при fallback."""
    urls = []
    def fake_get(self, url, **kwargs):
        urls.append(url)
        return _MockResponse(404 if "/health" in url else 200)

    _patch_client_get(monkeypatch, fake_get)

    is_router_reachable("http://192.168.0.111:8000/v1")
    assert any(u == "http://192.168.0.111:8000/v1/models" for u in urls)
