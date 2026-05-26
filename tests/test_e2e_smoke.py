"""End-to-end smoke test для v0.

Реально дёргает LLM-endpoint и tool-layer. Сlautless если endpoint не достижим.

Два уровня проверок:
1. **Integrity** — пайплайн не падает, trajectory создаётся и сохраняется.
2. **Capability** — модель действительно выполняет задачу (file written с правильным
   содержимым). Это reliability-индикатор; в v0 может flakeить — это валидный
   research-сигнал, не bug пайплайна.
"""
from __future__ import annotations

import socket
from pathlib import Path

import pytest

from harnes.goals.schema import (
    Goal,
    GoalClass,
    GoalStatus,
    JudgePredicate,
    Origin,
    StructuralPredicate,
)
from harnes.goals.store import GoalRepository
from harnes.memory.episodic import EpisodicStore
from harnes.memory.router import MemoryRouter
from harnes.metacycle.tick import run_tick
from harnes.react.loop import run_react
from harnes.react.schema import TrajectoryStatus
from harnes.skills.store import SkillRegistry
from harnes.tools.registry import get_registry, reset_registry


# ---------- LLM reachability gate ----------


def _llm_reachable(host: str = "192.168.0.111", port: int = 8000, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _llm_reachable(),
    reason="LLM endpoint at 192.168.0.111:8000 not reachable",
)


# ---------- Helpers ----------


@pytest.fixture
def repo_path(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def tool_registry():
    reset_registry()
    return get_registry()


@pytest.fixture
def general_skill():
    """Load the real `general` skill from the repo's skills/ dir."""
    repo_skills_dir = Path(__file__).resolve().parent.parent / "skills"
    reg = SkillRegistry(bundles_dir=repo_skills_dir, metrics_db=":memory:")
    skill = reg.get("general")
    assert skill is not None, "general.yaml should exist in skills/"
    return skill


# ---------- Tests ----------


def test_react_loop_runs_without_crash(
    tool_registry, general_skill, tmp_path: Path
) -> None:
    """Integrity check: ReAct запускается, дёргает реальный LLM, не падает.

    Не требуем успешного выполнения задачи — только что pipeline не упал и
    trajectory оформлена.
    """
    target_file = tmp_path / "smoke.txt"
    target_text = "hello from harnes smoke test"

    goal = Goal(
        description=(
            f"Write the exact text {target_text!r} (without surrounding quotes) "
            f"to the file at {target_file}. After writing, use tool_id=finish."
        ),
        goal_class=GoalClass.TASK,
        predicate_of_success=StructuralPredicate(expected_schema={"type": "object"}),
        origin=Origin.OPERATOR,
        originator="smoke",
    )

    traj = run_react(
        active_goal=goal,
        skill=general_skill,
        tool_registry=tool_registry,
        max_steps=10,
        budget_tokens=50_000,
    )

    # Integrity-инварианты.
    assert traj.id is not None
    assert traj.goal_id == goal.id
    assert traj.status in (
        TrajectoryStatus.SUCCESS,
        TrajectoryStatus.FAILURE,
        TrajectoryStatus.BUDGET_EXCEEDED,
    )
    assert traj.started_at is not None
    assert traj.ended_at is not None
    assert len(traj.steps) > 0

    # Логируем capability-результат как research-сигнал.
    print(
        f"\n[smoke] status={traj.status}, steps={len(traj.steps)}, "
        f"tokens={traj.total_cost.tokens}, "
        f"file_written={target_file.exists()}"
    )


def test_full_tick_through_metacycle(
    tool_registry, general_skill, tmp_path: Path
) -> None:
    """Полный путь через метацикл с реальным ReAct.

    Создаём цель → run_tick с real react_fn → trajectory сохраняется в LanceDB
    → goal-статус обновляется.
    """
    target_file = tmp_path / "tick_smoke.txt"
    target_text = "harnes ok"

    goal_repo = GoalRepository(":memory:")
    episodic = EpisodicStore(tmp_path / "lancedb")
    router = MemoryRouter(episodic=episodic)

    goal = Goal(
        description=(
            f"Write the text {target_text!r} to {target_file}. "
            f"Then use tool_id=finish with args.final_state={{'done': true}}."
        ),
        goal_class=GoalClass.TASK,
        predicate_of_success=StructuralPredicate(expected_schema={"type": "object"}),
        priority=1,
        origin=Origin.OPERATOR,
        originator="smoke",
    )
    goal_repo.create(goal)

    def react_fn(active_goal, focus, memory):
        return run_react(
            active_goal=active_goal,
            skill=general_skill,
            tool_registry=tool_registry,
            focus=focus,
            memory=memory,
            max_steps=10,
            budget_tokens=50_000,
        )

    state = run_tick(
        tick_id=0,
        event_queue=[],
        goal_repo=goal_repo,
        memory_router=router,
        episodic=episodic,
        react_fn=react_fn,
    )

    # Integrity checks
    assert state.idle is False
    assert state.trajectory is not None
    assert state.verdict is not None

    # Persistence
    meta = episodic.get_trajectory_meta(state.trajectory.id)
    assert meta is not None
    steps_stored = episodic.get_steps(state.trajectory.id)
    assert len(steps_stored) == len(state.trajectory.steps)

    # Goal status updated
    refetched = goal_repo.get(goal.id)
    assert refetched is not None
    assert refetched.status in (GoalStatus.DONE, GoalStatus.FAILED, GoalStatus.ACTIVE)

    print(
        f"\n[tick smoke] tick_id={state.tick_id}, "
        f"verdict={state.verdict.status}, "
        f"goal_status={refetched.status}, "
        f"trajectory_steps={len(state.trajectory.steps)}, "
        f"file_written={target_file.exists()}"
    )
