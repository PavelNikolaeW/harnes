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

    result = embeddings.embed(["a", "b"])

    assert len(result) == 2
    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.3, 0.4]
    assert captured["input"] == ["a", "b"]
    assert captured["model"].startswith("openai/")
    assert captured["model"].endswith(settings.embeddings.model)
    assert captured["api_base"] == settings.llm.api_base
