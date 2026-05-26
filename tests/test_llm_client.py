"""Тесты для harnes.llm.client.

Мокаем litellm.completion — реальный LLM-endpoint не дёргается.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from harnes.llm import client


def _fake_response(content: str = "ok") -> MagicMock:
    """Минимальный объект, похожий на ответ LiteLLM."""
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    return response


def test_model_id_adds_openai_prefix() -> None:
    """LiteLLM требует префикс openai/ для OpenAI-compatible endpoint'ов."""
    assert client._model_id().startswith("openai/")


# ---------- Tier dispatch (v0.1) ----------


def test_resolve_model_default() -> None:
    """Без tier → settings.llm.model (legacy)."""
    settings = client.get_settings()
    assert client._resolve_model(None) == settings.llm.model


def test_resolve_model_known_tier() -> None:
    """tier='light' → settings.llm.tiers['light']."""
    settings = client.get_settings()
    assert client._resolve_model("light") == settings.llm.tiers["light"]


def test_resolve_model_unknown_tier_falls_back() -> None:
    """Неизвестный tier → fallback на model, без исключения."""
    settings = client.get_settings()
    assert client._resolve_model("nonsense") == settings.llm.model


def test_call_forwards_tier_to_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """tier-параметр должен влиять на model в payload."""
    settings = client.get_settings()
    # Подменим один тир, чтобы отличить от дефолтного model
    settings.llm.tiers["heavy"] = "alt-model-id"

    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr(client, "completion", fake_completion)

    client.call([{"role": "user", "content": "x"}], tier="heavy")

    assert captured["model"] == "openai/alt-model-id"


def test_call_without_tier_uses_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tier=None → settings.llm.model (а не tiers[default_tier])."""
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr(client, "completion", fake_completion)

    settings = client.get_settings()
    client.call([{"role": "user", "content": "x"}])

    assert captured["model"] == f"openai/{settings.llm.model}"


def test_call_forwards_to_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """call() должен передавать настройки и сообщения в litellm.completion."""
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return _fake_response("hello")

    monkeypatch.setattr(client, "completion", fake_completion)

    response = client.call(
        [{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=42,
    )

    assert captured["model"].startswith("openai/")
    assert captured["temperature"] == 0.5
    assert captured["max_tokens"] == 42
    assert captured["api_base"].endswith("/v1")
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert response.choices[0].message.content == "hello"


def test_call_passes_extra_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Доп. kwargs должны пробрасываться (нужно для request_overrides и т.п.)."""
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return _fake_response()

    monkeypatch.setattr(client, "completion", fake_completion)

    client.call(
        [{"role": "user", "content": "x"}],
        stop=["END"],
        seed=42,
    )

    assert captured["stop"] == ["END"]
    assert captured["seed"] == 42


def test_health_check_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client, "completion", lambda **kw: _fake_response("ok"))
    assert client.health_check() is True


def test_health_check_handles_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ошибки коннекта НЕ должны выкидывать исключение из health_check."""

    def boom(**kw: Any) -> MagicMock:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(client, "completion", boom)
    assert client.health_check() is False
