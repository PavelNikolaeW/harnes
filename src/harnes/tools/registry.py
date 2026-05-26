"""Tool registry — глобальный реестр тулов + tool-layer pipeline.

См. `agent_architecture.html` § 10.

Pipeline:
  raw_args
    ↓ validate against tool.input_schema (Pydantic)
    ↓ resolve_irreversibility(tool, skill, args)  [только метка — критик отдельно]
    ↓ invoke(impl, args) with timeout
    ↓ classify outcome
    ↓ retry if retryable
    ↓ validate output against tool.output_schema
  → ObservationStep

В v0 не реализованы: настоящий timeout-enforcement (полагаемся на короткие
операции), отдельный critic-вызов на irreversible (он живёт в ReAct).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import structlog
from pydantic import BaseModel, ValidationError

from harnes.react.schema import ObservationOutcome, ObservationStep
from harnes.skills.schema import Skill
from harnes.tools.schema import BaseIrreversibility, BackoffStrategy, Tool

log = structlog.get_logger()


# ---------- Registration record ----------


@dataclass
class _ToolRecord:
    """Внутренняя запись реестра: spec + impl + типы args/result."""

    tool: Tool
    impl: Callable[[BaseModel], BaseModel]
    args_model: type[BaseModel]
    result_model: type[BaseModel]


# ---------- Conditional irreversibility predicates ----------


# Реестр предикатов: имя из tool.conditional_predicate → callable(args) → bool.
# Регистрируется через @register_predicate декоратор в builtin-тулах.
CONDITIONAL_PREDICATES: dict[str, Callable[[BaseModel], bool]] = {}


def register_predicate(name: str) -> Callable[[Callable[..., bool]], Callable[..., bool]]:
    def decorator(fn: Callable[..., bool]) -> Callable[..., bool]:
        CONDITIONAL_PREDICATES[name] = fn
        return fn
    return decorator


# ---------- Registry ----------


class ToolRegistry:
    """Глобальный реестр тулов. Один инстанс на процесс."""

    def __init__(self) -> None:
        self._records: dict[str, _ToolRecord] = {}

    def register(
        self,
        tool: Tool,
        impl: Callable[[Any], Any],
        args_model: type[BaseModel],
        result_model: type[BaseModel],
    ) -> None:
        if tool.id in self._records:
            raise ValueError(f"Tool {tool.id!r} already registered")
        self._records[tool.id] = _ToolRecord(
            tool=tool, impl=impl, args_model=args_model, result_model=result_model
        )
        log.debug("tool.registered", tool_id=tool.id, category=tool.category)

    def get(self, tool_id: str) -> Tool | None:
        rec = self._records.get(tool_id)
        return rec.tool if rec else None

    def list_ids(self) -> list[str]:
        return list(self._records.keys())

    # ---------- Irreversibility ----------

    def resolve_irreversibility(
        self,
        tool_id: str,
        raw_args: dict[str, Any],
        skill: Skill | None = None,
    ) -> bool:
        """Источник: tool (база) → skill (override) → action (resolved)."""
        rec = self._records.get(tool_id)
        if rec is None:
            return False

        # Skill override
        if skill is not None and tool_id in skill.irreversibility_overrides:
            override = skill.irreversibility_overrides[tool_id]
            if isinstance(override, bool):
                return override
            # Если override — строка, считаем именем callable. v0: только bool.

        tool = rec.tool
        if tool.base_irreversibility == BaseIrreversibility.NEVER:
            return False
        if tool.base_irreversibility == BaseIrreversibility.ALWAYS:
            return True

        # CONDITIONAL
        if tool.conditional_predicate is None:
            return False
        predicate = CONDITIONAL_PREDICATES.get(tool.conditional_predicate)
        if predicate is None:
            log.warning(
                "tool.conditional_predicate.missing",
                tool_id=tool_id,
                name=tool.conditional_predicate,
            )
            return False
        try:
            args = rec.args_model.model_validate(raw_args)
            return bool(predicate(args))
        except ValidationError:
            # Args не валидны — определять irreversibility преждевременно;
            # дальше pipeline вернёт SCHEMA_ERROR.
            return False

    # ---------- Invocation ----------

    def invoke(
        self,
        tool_id: str,
        raw_args: dict[str, Any],
        skill: Skill | None = None,
    ) -> ObservationStep:
        rec = self._records.get(tool_id)
        if rec is None:
            return ObservationStep(
                outcome=ObservationOutcome.SCHEMA_ERROR,
                error_detail=f"Unknown tool: {tool_id}",
            )

        retry_policy = rec.tool.retry_policy
        attempt = 0
        while True:
            obs = self._invoke_once(rec, raw_args)
            if (
                obs.outcome.value in retry_policy.retryable_outcomes
                and attempt < retry_policy.max_retries
            ):
                delay = self._backoff(retry_policy.backoff, retry_policy.initial_delay_seconds, attempt)
                log.debug(
                    "tool.retry",
                    tool_id=tool_id,
                    attempt=attempt,
                    outcome=obs.outcome,
                    delay_seconds=delay,
                )
                time.sleep(delay)
                attempt += 1
                continue
            return obs

    def _invoke_once(
        self,
        rec: _ToolRecord,
        raw_args: dict[str, Any],
    ) -> ObservationStep:
        # 1. Validate args
        try:
            args = rec.args_model.model_validate(raw_args)
        except ValidationError as e:
            return ObservationStep(
                outcome=ObservationOutcome.SCHEMA_ERROR,
                error_detail=str(e),
            )

        # 2. Invoke + classify exceptions
        try:
            result = rec.impl(args)
        except PermissionError as e:
            return ObservationStep(
                outcome=ObservationOutcome.PERMISSION_DENIED,
                error_detail=str(e),
            )
        except TimeoutError as e:
            return ObservationStep(
                outcome=ObservationOutcome.TIMEOUT,
                error_detail=str(e),
            )
        except Exception as e:  # noqa: BLE001
            log.error("tool.invocation.failed", tool_id=rec.tool.id, error=str(e))
            return ObservationStep(
                outcome=ObservationOutcome.TOOL_ERROR,
                error_detail=f"{type(e).__name__}: {e}",
            )

        # 3. Validate output
        try:
            if isinstance(result, BaseModel):
                validated = rec.result_model.model_validate(result.model_dump())
            else:
                validated = rec.result_model.model_validate(result)
        except ValidationError as e:
            return ObservationStep(
                outcome=ObservationOutcome.MALFORMED_OUTPUT,
                error_detail=str(e),
            )

        return ObservationStep(
            outcome=ObservationOutcome.SUCCESS,
            payload=validated.model_dump(),
        )

    @staticmethod
    def _backoff(strategy: BackoffStrategy, initial: float, attempt: int) -> float:
        if strategy == BackoffStrategy.LINEAR:
            return initial * (attempt + 1)
        # EXPONENTIAL
        return initial * (2 ** attempt)


# ---------- Module-level singleton ----------

_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Глобальный singleton-реестр. Lazy-init + auto-register builtin тулов."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        # Ленивая регистрация builtin (избегаем circular imports).
        from harnes.tools.builtin import register_builtins

        register_builtins(_registry)
    return _registry


def reset_registry() -> None:
    """Для тестов: сбросить singleton, чтобы следующий get_registry создал свежий."""
    global _registry
    _registry = None
