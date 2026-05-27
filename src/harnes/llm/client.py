"""LiteLLM-based chat completions client.

Tier-абстракция:
- light  — для attend / critic / verify (быстрый, маленький промпт)
- main   — для thought_call / action_call / plan_call в ReAct
- heavy  — для reflect (большой контекст, дорогая консолидация)

В dev все тиры мапятся на `gemma-26b-a4b` (см. LLMConfig.tiers). В проде —
gemma-31b-mtp на main, qwen-35b на heavy.

Public API:
- call(messages, *, tier=None, ...) -> response
- async_call(messages, *, tier=None, ...) -> awaitable
- health_check() -> bool

См. `agent_architecture.html` § 17.
"""
from __future__ import annotations

from typing import Any

import structlog
from litellm import acompletion, completion

from harnes.config import get_settings

log = structlog.get_logger()


def _resolve_model(tier: str | None) -> str:
    """Разрешает tier-имя в id модели.

    - tier=None → settings.llm.model (legacy / default)
    - tier='light'|'main'|'heavy' → settings.llm.tiers[tier]
    - неизвестный tier → settings.llm.model + warning
    """
    settings = get_settings()
    if tier is None:
        return settings.llm.model
    if tier in settings.llm.tiers:
        return settings.llm.tiers[tier]
    log.warning(
        "llm.tier.unknown_fallback_to_default",
        requested_tier=tier,
        known_tiers=list(settings.llm.tiers.keys()),
    )
    return settings.llm.model


def _model_id(tier: str | None = None) -> str:
    """LiteLLM требует префикс `openai/` для OpenAI-совместимых endpoint'ов."""
    model = _resolve_model(tier)
    if not model.startswith("openai/"):
        model = f"openai/{model}"
    return model


def _common_kwargs(tier: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    return {
        "model": _model_id(tier),
        "api_base": settings.llm.api_base,
        "api_key": settings.llm.api_key,
        "timeout": settings.llm.timeout,
        "num_retries": settings.llm.max_retries,
    }


def call(
    messages: list[dict[str, Any]],
    *,
    tier: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """Synchronous chat completion.

    `tier` опционально маршрутизирует на конкретную модель по конфигу.
    None — модель по умолчанию (settings.llm.model).
    """
    model_id = _resolve_model(tier)
    log.debug(
        "llm.call.start",
        model=model_id,
        tier=tier,
        message_count=len(messages),
        temperature=temperature,
    )
    response = completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **_common_kwargs(tier),
        **kwargs,
    )
    usage = getattr(response, "usage", None)
    log.debug(
        "llm.call.done",
        model=model_id,
        tier=tier,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )
    return response


async def async_call(
    messages: list[dict[str, Any]],
    *,
    tier: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """Asynchronous chat completion."""
    return await acompletion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **_common_kwargs(tier),
        **kwargs,
    )


def health_check() -> bool:
    """Smoke-тест endpoint'а — короткий запрос на дефолтную модель.

    ТЯЖЁЛЫЙ check: делает реальный chat-completion. Используется при boot'е
    агента (см. scripts/run_agent.py). Если нужен только дешёвый precheck —
    используй `is_router_reachable()`.
    """
    settings = get_settings()
    try:
        response = call(
            [{"role": "user", "content": "say ok"}],
            max_tokens=8,
        )
        content = response.choices[0].message.content
        log.info("llm.health_check.ok", endpoint=settings.llm.api_base, reply=content)
        return True
    except Exception as exc:  # noqa: BLE001 — boot-time check, не должно валиться
        log.error(
            "llm.health_check.failed",
            endpoint=settings.llm.api_base,
            model=settings.llm.model,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return False


def is_router_reachable(
    api_base: str | None = None,
    timeout_s: float = 2.0,
) -> bool:
    """Дешёвый precheck: жив ли роутер.

    Аналог `_bolt_reachable` из memory/world.py — TCP/HTTP precheck с коротким
    таймаутом, чтобы не висеть на 60s LiteLLM-таймауте если роутер упал.

    Использует:
    1. `GET {api_base}/../health` — когда роутер реализует endpoint (см.
       docs/router_roadmap.md R2)
    2. Fallback на `GET {api_base}/models` — это уже существует.

    Любая ошибка (ConnectError, timeout, 5xx) → False. 200 → True. 404 на
    health → пробуем models. Логи только при изменении состояния — не флудим.

    Args:
        api_base: например "http://192.168.0.111:8000/v1". None = из settings.
        timeout_s: общий timeout по запросу.

    Returns:
        True если роутер отвечает 200 хотя бы на один endpoint, иначе False.
    """
    import httpx

    if api_base is None:
        api_base = get_settings().llm.api_base

    # /v1/embeddings → /v1/.., /health НЕ /v1/health. Срезаем суффикс /v1.
    root = api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    health_url = f"{root}/health"
    models_url = f"{api_base.rstrip('/')}/models"

    with httpx.Client(timeout=timeout_s) as client:
        # 1) GET /health
        try:
            r = client.get(health_url)
            if r.status_code == 200:
                return True
            # 404 — endpoint ещё не реализован, проверяем /models
        except httpx.RequestError:
            # connection refused / timeout — пробуем второй вариант перед False
            pass
        except Exception:  # noqa: BLE001
            return False

        # 2) GET /v1/models (fallback)
        try:
            r = client.get(models_url)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False
