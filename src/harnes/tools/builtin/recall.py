"""Global recall_memory tool, поднятый поверх MemoryRouter.

v1.0 #34: recall становится first-class шагом. До этого Graphiti / Qdrant /
LanceDB подключены, но ReAct-цикл их активно не использовал — за исключением
multi-turn task-scoped recall в `harnes.eval.multi_turn`.

В отличие от `multi_turn.make_recall_tool_for(InMemoryChunkStore)` —
который привязывался к per-task chunk-store с эмбеддингами в RAM, —
этот тул работает поверх постоянной памяти агента:
- episodic (LanceDB; recent_steps),
- semantic (Qdrant; embedding search),
- world (Graphiti; temporal KG).

Любой backend может быть None — тогда соответствующее поле RecallResult
пустое. Falls open: ошибки бэкендов логируются, тул возвращает пустой ответ.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import structlog
from pydantic import BaseModel, Field

from harnes.memory.router import MemoryRouter
from harnes.memory.schema import MemoryType
from harnes.tools.registry import ToolRegistry
from harnes.tools.schema import BaseIrreversibility, RetryPolicy, Tool, ToolCategory

log = structlog.get_logger()


# ---------- Schemas ----------


_KNOWN_TYPES = {"episodic", "semantic", "world"}


class RecallArgs(BaseModel):
    query: str = Field(
        description=(
            "Natural-language query. Examples: 'previous attempts at this task',"
            " 'what is X', 'recent observations about Y'."
        )
    )
    k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return per memory type.",
    )
    types: list[str] | None = Field(
        default=None,
        description=(
            'Optional subset of ["episodic","semantic","world"]. '
            "None = search all available backends."
        ),
    )


class RecallHit(BaseModel):
    """Унифицированный hit. Backend-specific поля идут в metadata."""

    text: str = ""
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResult(BaseModel):
    episodic: list[RecallHit] = Field(default_factory=list)
    semantic: list[RecallHit] = Field(default_factory=list)
    world: list[RecallHit] = Field(default_factory=list)


# ---------- Factory ----------


def make_recall_tool_for_router(
    router: MemoryRouter,
) -> tuple[Tool, Callable[[RecallArgs], RecallResult]]:
    """Создаёт (Tool spec, impl) recall_memory привязанный к данному router'у.

    Возвращается фабрикой, потому что registry.register знает только impl(args)->result,
    а нам нужно прокинуть router без глобалок.
    """

    def recall_impl(args: RecallArgs) -> RecallResult:
        types: list[MemoryType] | None = None
        if args.types is not None:
            mapped: list[MemoryType] = []
            for t in args.types:
                if t in _KNOWN_TYPES:
                    mapped.append(MemoryType(t))
            types = mapped or None
        try:
            bundle = router.recall(query=args.query, types=types, k=args.k)
        except Exception as exc:  # noqa: BLE001
            log.warning("recall_memory.router.failed", error=str(exc), error_type=type(exc).__name__)
            return RecallResult()

        result = RecallResult()
        for r in bundle.episodic:
            content = r.content if isinstance(r.content, str) else str(r.content)
            result.episodic.append(
                RecallHit(
                    text=content[:500],
                    metadata={
                        "step_type": r.step_type,
                        "timestamp": str(r.timestamp),
                        "trajectory_id": str(r.trajectory_id),
                        "goal_id": str(r.goal_id),
                    },
                )
            )
        for r in bundle.semantic:
            result.semantic.append(
                RecallHit(
                    text=r.text,
                    metadata=dict(r.metadata),
                )
            )
        for r in bundle.world:
            label_str = ", ".join(r.labels) if r.labels else "(node)"
            # World nodes часто не имеют единого 'text' — берём label + краткий dump props.
            props = dict(r.properties)
            summary = props.get("name") or props.get("summary") or ""
            result.world.append(
                RecallHit(
                    text=f"{label_str}: {summary}" if summary else label_str,
                    metadata={"id": r.id, "labels": r.labels, "properties": props},
                )
            )
        log.debug(
            "recall_memory.done",
            query_len=len(args.query),
            episodic=len(result.episodic),
            semantic=len(result.semantic),
            world=len(result.world),
        )
        return result

    tool = Tool(
        id="recall_memory",
        name="recall_memory",
        description=(
            "Search the agent's persistent memory across three backends: "
            "episodic (past trajectory steps), semantic (facts), and world "
            "(temporal knowledge graph). Use to ground answers in remembered "
            "information before resorting to guesses. Returns up to k hits per "
            "backend."
        ),
        input_schema=RecallArgs.model_json_schema(),
        output_schema=RecallResult.model_json_schema(),
        base_irreversibility=BaseIrreversibility.NEVER,
        side_effects="None — reads only.",
        category=ToolCategory.INFO,
        retry_policy=RetryPolicy(retryable_outcomes=["timeout"], max_retries=1),
        timeout_seconds=10.0,
        implementation_ref="harnes.tools.builtin.recall.recall_impl_router_bound",
    )
    return tool, recall_impl


def build_runtime_registry(
    router: MemoryRouter | None,
    *,
    include_io: bool = True,
) -> ToolRegistry:
    """Возвращает ToolRegistry с read_file + write_file + (если router передан) recall_memory.

    v1.0 #34: используется в run-loop и run-eval --real вместо глобального
    get_registry() — чтобы recall_memory был доступен в agent'ской трассе.

    include_io=False позволяет в тестах собрать registry только с recall.
    """
    from harnes.tools.builtin.io import (
        READ_FILE_TOOL,
        ReadFileArgs,
        ReadFileResult,
        WRITE_FILE_TOOL,
        WriteFileArgs,
        WriteFileResult,
        read_file_impl,
        write_file_impl,
    )

    registry = ToolRegistry()
    if include_io:
        registry.register(READ_FILE_TOOL, read_file_impl, ReadFileArgs, ReadFileResult)
        registry.register(
            WRITE_FILE_TOOL, write_file_impl, WriteFileArgs, WriteFileResult
        )

    if router is not None:
        tool, impl = make_recall_tool_for_router(router)
        registry.register(tool, impl, RecallArgs, RecallResult)

    log.debug(
        "runtime_registry.built",
        has_router=router is not None,
        include_io=include_io,
        tools=registry.list_ids(),
    )
    return registry
