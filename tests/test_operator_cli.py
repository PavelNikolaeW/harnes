"""Tests for harnes.operator.cli — через Click CliRunner.

Переопределяем пути storage'а на tmp_path через env-переменные.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from harnes.operator.cli import cli


@pytest.fixture
def isolated_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> CliRunner:
    """Per-test изоляция: redirect goal/memory paths в tmp_path."""
    monkeypatch.setenv("GOAL_STORE__SQLITE_PATH", str(tmp_path / "goals.db"))
    monkeypatch.setenv("MEMORY__LANCEDB_PATH", str(tmp_path / "lancedb"))
    monkeypatch.setenv(
        "PROCEDURAL_STORE__SQLITE_PATH", str(tmp_path / "skill_metrics.db")
    )
    monkeypatch.setenv("PROCEDURAL_STORE__BUNDLES_DIR", str(tmp_path / "skills"))
    # CLI tests не должны лезть в реальный Neo4j — поинтим на unreachable.
    # WorldModelStore swallows connection errors, тест проверяет только
    # CLI-механику, не side-channel в KG.
    monkeypatch.setenv("MEMORY__NEO4J_URI", "bolt://nowhere.invalid:7687")
    return CliRunner()


def test_list_goals_empty(isolated_cli: CliRunner) -> None:
    result = isolated_cli.invoke(cli, ["list-goals"])
    assert result.exit_code == 0
    assert "no goals" in result.output


def test_enter_goal_creates(isolated_cli: CliRunner) -> None:
    result = isolated_cli.invoke(cli, ["enter-goal", "write hello"])
    assert result.exit_code == 0
    assert "Created goal" in result.output
    assert "status: pending" in result.output


def test_enter_then_list(isolated_cli: CliRunner) -> None:
    isolated_cli.invoke(cli, ["enter-goal", "first"])
    isolated_cli.invoke(cli, ["enter-goal", "second", "--priority", "5"])
    result = isolated_cli.invoke(cli, ["list-goals"])
    assert result.exit_code == 0
    assert "first" in result.output
    assert "second" in result.output


def test_enter_with_structural_predicate(isolated_cli: CliRunner) -> None:
    result = isolated_cli.invoke(
        cli, ["enter-goal", "x", "--predicate", "structural"]
    )
    assert result.exit_code == 0
    assert "Created goal" in result.output


def test_inspect_unknown(isolated_cli: CliRunner) -> None:
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    result = isolated_cli.invoke(cli, ["inspect", fake_uuid])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_inspect_known(isolated_cli: CliRunner) -> None:
    # Create a goal
    isolated_cli.invoke(cli, ["enter-goal", "to inspect"])

    # Extract goal id from list output
    list_result = isolated_cli.invoke(cli, ["list-goals"])
    first_line = list_result.output.strip().split("\n")[0]
    goal_id = first_line.split()[0]

    result = isolated_cli.invoke(cli, ["inspect", goal_id])
    assert result.exit_code == 0
    assert "to inspect" in result.output
    # JSON parseable
    import json

    json.loads(result.output)


def test_approve_wrong_status(isolated_cli: CliRunner) -> None:
    """Цель в PENDING (не PENDING_APPROVAL) — approve должен упасть."""
    isolated_cli.invoke(cli, ["enter-goal", "x"])
    list_result = isolated_cli.invoke(cli, ["list-goals"])
    goal_id = list_result.output.strip().split("\n")[0].split()[0]

    result = isolated_cli.invoke(cli, ["approve", goal_id])
    assert result.exit_code == 1
    assert "expected pending_approval" in result.output


def test_run_tick_idle(isolated_cli: CliRunner) -> None:
    result = isolated_cli.invoke(cli, ["run-tick"])
    assert result.exit_code == 0
    assert "idle" in result.output


def test_run_tick_processes_goal(isolated_cli: CliRunner) -> None:
    """Полный e2e через CLI: enter → run-tick → goal goes DONE."""
    isolated_cli.invoke(cli, ["enter-goal", "test goal", "--predicate", "structural"])
    result = isolated_cli.invoke(cli, ["run-tick"])
    assert result.exit_code == 0
    assert "Tick processed" in result.output
    assert "verdict" in result.output

    # Проверить, что цель теперь DONE
    list_result = isolated_cli.invoke(cli, ["list-goals", "--status", "done"])
    assert "test goal" in list_result.output


# ---------- v0.1: bootstrap-standing ----------


def test_bootstrap_standing_creates_set(isolated_cli: CliRunner) -> None:
    result = isolated_cli.invoke(cli, ["bootstrap-standing"])
    assert result.exit_code == 0
    assert "Created" in result.output
    assert "on_alert_observation" in result.output or "alerts" in result.output

    # idempotent
    second = isolated_cli.invoke(cli, ["bootstrap-standing"])
    assert "already exist" in second.output


# ---------- v0.1: trace explorer ----------


def test_recent_trajectories_empty(isolated_cli: CliRunner) -> None:
    result = isolated_cli.invoke(cli, ["recent-trajectories"])
    assert result.exit_code == 0
    assert "(no trajectories)" in result.output


def test_recent_steps_empty(isolated_cli: CliRunner) -> None:
    result = isolated_cli.invoke(cli, ["recent-steps"])
    assert result.exit_code == 0
    assert "(no steps)" in result.output


def test_recent_trajectories_after_tick(isolated_cli: CliRunner) -> None:
    """run-tick (stub ReAct) пишет trajectory → recent-trajectories её показывает."""
    isolated_cli.invoke(cli, ["enter-goal", "x", "--predicate", "structural"])
    isolated_cli.invoke(cli, ["run-tick"])

    result = isolated_cli.invoke(cli, ["recent-trajectories"])
    assert result.exit_code == 0
    assert "success" in result.output


def test_inspect_trajectory_unknown(isolated_cli: CliRunner) -> None:
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    result = isolated_cli.invoke(cli, ["inspect-trajectory", fake_uuid])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_inspect_trajectory_after_tick(isolated_cli: CliRunner) -> None:
    """Создаём trajectory через тик, потом inspect её через CLI."""
    isolated_cli.invoke(cli, ["enter-goal", "x", "--predicate", "structural"])
    isolated_cli.invoke(cli, ["run-tick"])

    list_result = isolated_cli.invoke(cli, ["recent-trajectories"])
    traj_id = list_result.output.strip().split()[0]

    result = isolated_cli.invoke(cli, ["inspect-trajectory", traj_id])
    assert result.exit_code == 0
    assert "Trajectory" in result.output
    assert "Steps" in result.output


def test_goal_tree_single(isolated_cli: CliRunner) -> None:
    isolated_cli.invoke(cli, ["enter-goal", "root goal"])
    list_result = isolated_cli.invoke(cli, ["list-goals"])
    goal_id = list_result.output.strip().split()[0]

    result = isolated_cli.invoke(cli, ["goal-tree", goal_id])
    assert result.exit_code == 0
    assert "root goal" in result.output


# ---------- v0.1: run-loop ----------


def test_run_loop_finite_idle(isolated_cli: CliRunner) -> None:
    """Без целей: run-loop --max-ticks 3 --stub → 3 idle тика, exit 0."""
    result = isolated_cli.invoke(
        cli, ["run-loop", "--max-ticks", "3", "--interval", "0.01", "--stub"]
    )
    assert result.exit_code == 0
    assert "Stopped after 3 ticks" in result.output
    assert "3 idle" in result.output


def test_run_loop_processes_pending_goals(isolated_cli: CliRunner) -> None:
    """Создаём цель → run-loop --stub один тик → цель должна стать done."""
    isolated_cli.invoke(cli, ["enter-goal", "a", "--predicate", "structural"])
    result = isolated_cli.invoke(
        cli, ["run-loop", "--max-ticks", "1", "--interval", "0.01", "--stub"]
    )
    assert result.exit_code == 0
    assert "1 processed" in result.output

    # Check goal is done
    list_result = isolated_cli.invoke(cli, ["list-goals", "--status", "done"])
    assert "a" in list_result.output
