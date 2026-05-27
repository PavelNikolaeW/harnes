"""Tests for harnes.llm.embeddings.

Оба пути (server и fastembed) мокаются — внешние зависимости не дёргаются.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from harnes.llm import embeddings


def test_embed_empty_returns_empty() -> None:
    assert embeddings.embed([]) == []


def test_embed_via_fastembed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """По умолчанию (use_server=False) идём через fastembed."""
    fake_model = MagicMock()
    fake_model.embed.return_value = iter(
        [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
    )
    monkeypatch.setattr(embeddings, "_fastembed_model", lambda: fake_model)

    settings = embeddings.get_settings()
    settings.embeddings.use_server = False

    result = embeddings.embed(["hello", "world"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2, 0.3]
    assert result[1] == [0.4, 0.5, 0.6]
    fake_model.embed.assert_called_once_with(["hello", "world"])


def test_embed_via_server_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """use_server=True → идём через LiteLLM /v1/embeddings."""
    captured: dict[str, Any] = {}

    def fake_embedding(**kwargs: Any) -> Any:
        captured.update(kwargs)
        response = MagicMock()
        response.data = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
        return response

    monkeypatch.setattr(embeddings, "embedding", fake_embedding)

    settings = embeddings.get_settings()
    settings.embeddings.use_server = True
    embeddings.reset_server_state()

    result = embeddings.embed(["a", "b"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.3, 0.4]
    assert captured["input"] == ["a", "b"]
    assert captured["model"].startswith("openai/")
    assert captured["model"].endswith(settings.embeddings.model)
    assert captured["api_base"] == settings.llm.api_base


# ---------- Graceful fallback при server-exception (router_roadmap R1) ----------


def test_embed_server_failure_falls_back_to_fastembed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """use_server=True + server бросает → warning + fastembed fallback, не raise."""
    # 1) Server раздаёт ошибку (404, timeout, что угодно)
    def broken_embedding(**kwargs: Any) -> Any:
        raise RuntimeError("404 Not Found: /v1/embeddings")

    monkeypatch.setattr(embeddings, "embedding", broken_embedding)

    # 2) Fastembed возвращает валидные вектора
    fake_model = MagicMock()
    fake_model.embed.return_value = iter([np.array([0.7, 0.8])])
    monkeypatch.setattr(embeddings, "_fastembed_model", lambda: fake_model)

    settings = embeddings.get_settings()
    settings.embeddings.use_server = True
    embeddings.reset_server_state()

    # Не должно raise. Возвращает fastembed-вектор.
    result = embeddings.embed(["query"])
    assert result == [[0.7, 0.8]]


def test_embed_broken_state_short_circuits_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """После 1 server-fail → следующий embed идёт сразу в fastembed (не дёргаем сервер)."""
    call_count = {"server": 0, "fastembed": 0}

    def broken_embedding(**kwargs: Any) -> Any:
        call_count["server"] += 1
        raise RuntimeError("server down")

    fake_model = MagicMock()
    def fake_embed(texts):
        call_count["fastembed"] += 1
        return iter([np.array([0.0]) for _ in texts])
    fake_model.embed.side_effect = fake_embed

    monkeypatch.setattr(embeddings, "embedding", broken_embedding)
    monkeypatch.setattr(embeddings, "_fastembed_model", lambda: fake_model)

    settings = embeddings.get_settings()
    settings.embeddings.use_server = True
    embeddings.reset_server_state()

    # 1-й вызов — server fail + fallback fastembed
    embeddings.embed(["a"])
    assert call_count["server"] == 1
    assert call_count["fastembed"] == 1

    # 2-й вызов — broken cache активен, server НЕ дёргается
    embeddings.embed(["b"])
    assert call_count["server"] == 1  # без изменений!
    assert call_count["fastembed"] == 2


def test_embed_reset_server_state_re_enables_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset_server_state() убирает broken-cache → следующий embed дёрнет сервер снова."""
    server_calls = {"n": 0}

    def server_then_ok(**kwargs: Any) -> Any:
        server_calls["n"] += 1
        if server_calls["n"] == 1:
            raise RuntimeError("first call fails")
        # Второй и далее — успешен
        response = MagicMock()
        response.data = [{"embedding": [0.9, 0.9]}]
        return response

    fake_model = MagicMock()
    fake_model.embed.return_value = iter([np.array([0.0])])

    monkeypatch.setattr(embeddings, "embedding", server_then_ok)
    monkeypatch.setattr(embeddings, "_fastembed_model", lambda: fake_model)

    settings = embeddings.get_settings()
    settings.embeddings.use_server = True
    embeddings.reset_server_state()

    # 1-й вызов — server fail
    r1 = embeddings.embed(["x"])
    assert server_calls["n"] == 1
    assert r1 == [[0.0]]  # fastembed fallback

    # Reset → broken cache очищен
    embeddings.reset_server_state()

    # 2-й вызов — сервер дёргается снова, теперь успешно
    r2 = embeddings.embed(["y"])
    assert server_calls["n"] == 2
    assert r2 == [[0.9, 0.9]]


def test_embed_server_disabled_never_touches_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """use_server=False → embedding() не вызывается ни при каких обстоятельствах."""
    def must_not_be_called(**kwargs: Any) -> Any:
        raise AssertionError("server endpoint must NOT be called when use_server=False")

    fake_model = MagicMock()
    fake_model.embed.return_value = iter([np.array([0.5])])

    monkeypatch.setattr(embeddings, "embedding", must_not_be_called)
    monkeypatch.setattr(embeddings, "_fastembed_model", lambda: fake_model)

    settings = embeddings.get_settings()
    settings.embeddings.use_server = False

    result = embeddings.embed(["text"])
    assert result == [[0.5]]
