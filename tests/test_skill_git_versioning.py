"""Tests for SkillRegistry git auto-commit (v0.3 #30)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from harnes.skills.schema import Skill
from harnes.skills.store import SkillRegistry


def _git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + args, cwd=cwd, capture_output=True, text=True, check=check, timeout=10
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Создаёт изолированный git-репо для теста с уже сделанным initial commit."""
    _git(["init", "-b", "main"], cwd=tmp_path)
    _git(["config", "user.email", "test@harnes.local"], cwd=tmp_path)
    _git(["config", "user.name", "Test"], cwd=tmp_path)
    (tmp_path / "README.md").write_text("test repo")
    _git(["add", "README.md"], cwd=tmp_path)
    _git(["commit", "-m", "initial"], cwd=tmp_path)
    return tmp_path


def _make_skill(skill_id: str = "general", version: str = "0.0.1", **kwargs) -> Skill:
    return Skill(
        id=skill_id,
        name=skill_id,
        description="test skill",
        version=version,
        prompt_template="Goal: {goal_description}",
        **kwargs,
    )


# ---------- default: no auto-commit ----------


def test_save_no_commit_by_default(git_repo: Path) -> None:
    bundles = git_repo / "skills"
    bundles.mkdir()
    reg = SkillRegistry(bundles_dir=bundles, metrics_db=":memory:")

    reg.save(_make_skill())

    # YAML создан
    assert (bundles / "general.yaml").exists()
    # Git status показывает untracked content (с -uall — конкретный файл)
    status = _git(
        ["status", "--porcelain", "--untracked-files=all"],
        cwd=git_repo,
        check=False,
    ).stdout
    assert "skills/general.yaml" in status


# ---------- with auto_commit ----------


def test_save_auto_commits_when_enabled(git_repo: Path) -> None:
    bundles = git_repo / "skills"
    bundles.mkdir()
    reg = SkillRegistry(
        bundles_dir=bundles, metrics_db=":memory:", git_auto_commit=True
    )

    reg.save(_make_skill(version="0.0.1"))

    # YAML создан + закоммичен
    assert (bundles / "general.yaml").exists()
    status = _git(["status", "--porcelain"], cwd=git_repo, check=False).stdout
    assert status.strip() == ""  # чисто

    # Лог содержит наш коммит
    log = _git(["log", "--oneline", "-n", "5"], cwd=git_repo).stdout
    assert "skill general: v0.0.1" in log


def test_save_versioning_creates_multiple_commits(git_repo: Path) -> None:
    bundles = git_repo / "skills"
    bundles.mkdir()
    reg = SkillRegistry(
        bundles_dir=bundles, metrics_db=":memory:", git_auto_commit=True
    )

    reg.save(_make_skill(version="0.0.1"))
    reg.save(_make_skill(version="0.0.2", parent_version_id="0.0.1"))
    reg.save(_make_skill(version="0.0.3", parent_version_id="0.0.2"))

    log = _git(["log", "--oneline"], cwd=git_repo).stdout
    assert "v0.0.1" in log
    assert "v0.0.2 (parent v0.0.1)" in log
    assert "v0.0.3 (parent v0.0.2)" in log

    # Каждая версия в отдельном коммите
    commit_count = len([ln for ln in log.splitlines() if "skill general:" in ln])
    assert commit_count == 3


def test_save_skips_commit_when_no_changes(git_repo: Path) -> None:
    """Повторное сохранение того же скилла не должно создавать пустой коммит."""
    bundles = git_repo / "skills"
    bundles.mkdir()
    reg = SkillRegistry(
        bundles_dir=bundles, metrics_db=":memory:", git_auto_commit=True
    )

    s = _make_skill(version="0.0.1")
    reg.save(s)
    log_before = _git(["log", "--oneline"], cwd=git_repo).stdout

    reg.save(s)  # тот же контент
    log_after = _git(["log", "--oneline"], cwd=git_repo).stdout

    assert log_before == log_after


def test_save_outside_git_repo_safe(tmp_path: Path) -> None:
    """Если bundles_dir не в git-репо — нет краша."""
    bundles = tmp_path / "skills"
    bundles.mkdir()
    reg = SkillRegistry(
        bundles_dir=bundles, metrics_db=":memory:", git_auto_commit=True
    )
    # Должно просто не упасть
    reg.save(_make_skill())
    assert (bundles / "general.yaml").exists()


# ---------- with auto_tag ----------


def test_save_creates_git_tag_when_enabled(git_repo: Path) -> None:
    bundles = git_repo / "skills"
    bundles.mkdir()
    reg = SkillRegistry(
        bundles_dir=bundles,
        metrics_db=":memory:",
        git_auto_commit=True,
        git_auto_tag=True,
    )

    reg.save(_make_skill(version="1.0.0"))

    tags = _git(["tag", "-l"], cwd=git_repo).stdout
    assert "skill/general/v1.0.0" in tags


def test_save_no_tag_without_flag(git_repo: Path) -> None:
    bundles = git_repo / "skills"
    bundles.mkdir()
    reg = SkillRegistry(
        bundles_dir=bundles, metrics_db=":memory:", git_auto_commit=True
    )  # git_auto_tag default False

    reg.save(_make_skill(version="1.0.0"))

    tags = _git(["tag", "-l"], cwd=git_repo).stdout
    assert "skill/general/v1.0.0" not in tags
