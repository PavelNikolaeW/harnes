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


def _thinking_extra_body(enable_thinking: bool) -> dict[str, Any]:
    """Возвращает extra_body для LiteLLM с включённым/выключенным native thinking.

    ll-router принимает chat_template_kwargs через extra_body. По умолчанию
    backend имеет enable_thinking=false (см. /v1/models request_overrides),
    мы переопределяем здесь.

    Empirical test: gemma-26b-a4b и qwen-35b возвращают `reasoning_content`
    как отдельное поле в `message`, а `content` остаётся чистым (JSON для
    action_call, plain text для thought_call). Это значит наш regex-парсинг
    JSON не ловит мусор из reasoning.
    """
    return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}


def call(
    messages: list[dict[str, Any]],
    *,
    tier: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    enable_thinking: bool = True,
    **kwargs: Any,
) -> Any:
    """Synchronous chat completion.

    `tier` опционально маршрутизирует на конкретную модель по конфигу.
    None — модель по умолчанию (settings.llm.model).

    `enable_thinking` (default True) — включает native CoT через chat-template
    kwarg. Модель вернёт `reasoning_content` отдельным полем, `content` чистым.
    Caller сам решает использовать reasoning_content (для логов) или нет.
    Латентность ↑ ×2-3, качество reasoning ↑ заметно.
    """
    model_id = _resolve_model(tier)
    log.debug(
        "llm.call.start",
        model=model_id,
        tier=tier,
        message_count=len(messages),
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
    # Merge extra_body: caller может передать свой extra_body через kwargs.
    extra_body = dict(kwargs.pop("extra_body", None) or {})
    extra_body.update(_thinking_extra_body(enable_thinking))

    response = completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
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
    enable_thinking: bool = True,
    **kwargs: Any,
) -> Any:
    """Asynchronous chat completion. См. call() для enable_thinking."""
    extra_body = dict(kwargs.pop("extra_body", None) or {})
    extra_body.update(_thinking_extra_body(enable_thinking))
    return await acompletion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
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


def get_router_load_status(
    api_base: str | None = None,
    timeout_s: float = 3.0,
) -> list[dict[str, Any]]:
    """Per-model снимок состояния роутера (см. docs/router_roadmap.md R3).

    GET /v1/models/load_status → list of {id, backend_type (gpu|cpu), kind,
    health_ok, ...}. Полезно при boot run-loop'а — поймать ситуацию когда
    модель fallback'нулась на CPU (case 2026-05-26: gemma-26b-a4b-cpu вместо
    gpu варианта, latency упала с 146 t/s до ~12 t/s — мы это не сразу заметили).

    Возвращает [] если endpoint недоступен или вернул ошибку. Не raise.

    Args:
        api_base: например "http://192.168.0.111:8000/v1". None = из settings.
        timeout_s: HTTP timeout.

    Returns:
        list of dicts с per-model информацией, или [] на ошибке.
    """
    import httpx

    if api_base is None:
        api_base = get_settings().llm.api_base
    url = f"{api_base.rstrip('/')}/models/load_status"

    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.get(url)
            if r.status_code != 200:
                return []
            data = r.json()
            entries = data.get("data") if isinstance(data, dict) else None
            return list(entries) if isinstance(entries, list) else []
    except Exception as exc:  # noqa: BLE001
        log.debug("llm.load_status.failed", error=str(exc))
        return []


def warn_if_models_on_cpu(
    expected_tier_models: list[str] | None = None,
    api_base: str | None = None,
) -> list[str]:
    """Warns если какие-то из tier-моделей крутятся на CPU а не GPU.

    Используется при старте run-loop'а — поймать случай 2026-05-26 (gemma fallback
    на CPU после того как админ забрал одну карту → агент тормозил в 12 раз и
    мы это не сразу заметили).

    Args:
        expected_tier_models: список model_id которые ожидаются на GPU. None =
          settings.llm.tiers values.
        api_base: см. get_router_load_status.

    Returns:
        список model_id которые реально не на GPU (для информирования оператора).
        Пустой список = всё ОК или endpoint недоступен.
    """
    if expected_tier_models is None:
        expected_tier_models = list(set(get_settings().llm.tiers.values()))

    entries = get_router_load_status(api_base=api_base)
    if not entries:
        return []

    on_cpu: list[str] = []
    for entry in entries:
        mid = entry.get("id")
        if mid not in expected_tier_models:
            continue
        backend_type = entry.get("backend_type")
        if backend_type and backend_type != "gpu":
            on_cpu.append(str(mid))
            log.warning(
                "llm.model.not_on_gpu",
                model_id=mid,
                backend_type=backend_type,
                hint="latency будет ×10-30 хуже; проверь nvidia-smi на роутере",
            )
    return on_cpu


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
