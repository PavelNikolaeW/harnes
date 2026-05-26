"""Tests for harnes.tools.registry + builtin tools."""
from __future__ import annotations

from pathlib import Path

import pytest

from harnes.react.schema import ObservationOutcome
from harnes.skills.schema import Skill
from harnes.tools.registry import ToolRegistry, get_registry, reset_registry


@pytest.fixture
def registry() -> ToolRegistry:
    """Fresh global registry (builtin tools auto-registered)."""
    reset_registry()
    return get_registry()


# ---------- Registry basics ----------


def test_builtin_tools_registered(registry: ToolRegistry) -> None:
    assert "read_file" in registry.list_ids()
    assert "write_file" in registry.list_ids()


def test_get_unknown_tool(registry: ToolRegistry) -> None:
    assert registry.get("nonexistent") is None


# ---------- read_file ----------


def test_read_file_success(registry: ToolRegistry, tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello, world!", encoding="utf-8")

    obs = registry.invoke("read_file", {"path": str(f)})
    assert obs.outcome == ObservationOutcome.SUCCESS
    assert obs.payload is not None
    assert obs.payload["content"] == "hello, world!"
    assert obs.payload["bytes_read"] == 13
    assert obs.payload["truncated"] is False


def test_read_file_truncation(registry: ToolRegistry, tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_text("a" * 1000, encoding="utf-8")

    obs = registry.invoke("read_file", {"path": str(f), "max_bytes": 100})
    assert obs.outcome == ObservationOutcome.SUCCESS
    assert obs.payload is not None
    assert obs.payload["bytes_read"] == 100
    assert obs.payload["truncated"] is True


def test_read_file_missing(registry: ToolRegistry, tmp_path: Path) -> None:
    obs = registry.invoke("read_file", {"path": str(tmp_path / "missing.txt")})
    assert obs.outcome == ObservationOutcome.TOOL_ERROR
    assert "does not exist" in (obs.error_detail or "")


def test_read_file_bad_args(registry: ToolRegistry) -> None:
    obs = registry.invoke("read_file", {})  # missing path
    assert obs.outcome == ObservationOutcome.SCHEMA_ERROR


def test_read_file_irreversibility_is_never(registry: ToolRegistry) -> None:
    flag = registry.resolve_irreversibility("read_file", {"path": "/tmp/x"})
    assert flag is False


# ---------- write_file ----------


def test_write_file_creates_new(registry: ToolRegistry, tmp_path: Path) -> None:
    f = tmp_path / "out.txt"
    obs = registry.invoke(
        "write_file", {"path": str(f), "content": "hello"}
    )
    assert obs.outcome == ObservationOutcome.SUCCESS
    assert obs.payload is not None
    assert obs.payload["overwritten"] is False
    assert obs.payload["bytes_written"] == 5
    assert f.read_text(encoding="utf-8") == "hello"


def test_write_file_overwrites(registry: ToolRegistry, tmp_path: Path) -> None:
    f = tmp_path / "out.txt"
    f.write_text("old", encoding="utf-8")

    obs = registry.invoke("write_file", {"path": str(f), "content": "new"})
    assert obs.outcome == ObservationOutcome.SUCCESS
    assert obs.payload["overwritten"] is True
    assert f.read_text(encoding="utf-8") == "new"


def test_write_file_creates_parents(registry: ToolRegistry, tmp_path: Path) -> None:
    f = tmp_path / "a" / "b" / "out.txt"
    obs = registry.invoke("write_file", {"path": str(f), "content": "x"})
    assert obs.outcome == ObservationOutcome.SUCCESS
    assert f.exists()


# ---------- Irreversibility resolution ----------


def test_write_file_irreversibility_when_new(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    f = tmp_path / "fresh.txt"
    flag = registry.resolve_irreversibility(
        "write_file", {"path": str(f), "content": "x"}
    )
    assert flag is False


def test_write_file_irreversibility_when_existing(
    registry: ToolRegistry, tmp_path: Path
) -> None:
    f = tmp_path / "existing.txt"
    f.write_text("old", encoding="utf-8")
    flag = registry.resolve_irreversibility(
        "write_file", {"path": str(f), "content": "x"}
    )
    assert flag is True


def test_skill_override_takes_precedence(registry: ToolRegistry) -> None:
    """Skill может перекрыть базовое значение тула."""
    skill = Skill(
        id="paranoid",
        name="paranoid",
        description="treats every tool as irreversible",
        prompt_template="...",
        irreversibility_overrides={"read_file": True},
    )
    flag = registry.resolve_irreversibility("read_file", {"path": "/tmp/x"}, skill=skill)
    assert flag is True


# ---------- Validation of outputs ----------


def test_invoke_returns_observation_step(registry: ToolRegistry, tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("y", encoding="utf-8")
    obs = registry.invoke("read_file", {"path": str(f)})
    # obs is a real ObservationStep
    assert obs.type == "observation"
    assert obs.outcome == ObservationOutcome.SUCCESS


def test_unknown_tool_returns_schema_error(registry: ToolRegistry) -> None:
    obs = registry.invoke("nonsense", {})
    assert obs.outcome == ObservationOutcome.SCHEMA_ERROR
    assert "Unknown tool" in (obs.error_detail or "")
