"""LiteLLM-based chat completions client.

В v0 — единственная модель `gemma-26b-a4b` через ll-router на
http://192.168.0.111:8000/v1.

Public API:
- call(messages, **kwargs) -> response from LiteLLM
- async_call(messages, **kwargs) -> awaitable
- health_check() -> bool (тонкий smoke-тест endpoint'а)

См. `agent_architecture.html` § 17.
"""
from __future__ import annotations

from typing import Any

import structlog
from litellm import acompletion, completion

from harnes.config import get_settings

log = structlog.get_logger()


def _model_id() -> str:
    """LiteLLM использует префикс `openai/` для OpenAI-совместимых endpoint'ов."""
    settings = get_settings()
    model = settings.llm.model
    if not model.startswith("openai/"):
        model = f"openai/{model}"
    return model


def _common_kwargs() -> dict[str, Any]:
    settings = get_settings()
    return {
        "model": _model_id(),
        "api_base": settings.llm.api_base,
        "api_key": settings.llm.api_key,
        "timeout": settings.llm.timeout,
        "num_retries": settings.llm.max_retries,
    }


def call(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """Synchronous chat completion."""
    settings = get_settings()
    log.debug(
        "llm.call.start",
        model=settings.llm.model,
        message_count=len(messages),
        temperature=temperature,
    )
    response = completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **_common_kwargs(),
        **kwargs,
    )
    usage = getattr(response, "usage", None)
    log.debug(
        "llm.call.done",
        model=settings.llm.model,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )
    return response


async def async_call(
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> Any:
    """Asynchronous chat completion."""
    return await acompletion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **_common_kwargs(),
        **kwargs,
    )


def health_check() -> bool:
    """Smoke-тест endpoint'а: послать один маленький запрос, убедиться, что ответ валидный.

    Логирует ошибку и возвращает False вместо бросания исключения —
    используется при boot'е агента (см. scripts/run_agent.py).
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
