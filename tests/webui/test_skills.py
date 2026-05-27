"""Tests для /skills + utility `_bump_patch`.

- list smoke (works with empty bundles).
- edit-bump version (YAML rewrite + parent_version_id).
- _bump_patch — параметризованная таблица.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from harnes.skills.schema import Skill, SkillOrigin, SkillStatus
from harnes.webui.routers.api_skills import _bump_patch

# ---------- _bump_patch unit ----------


@pytest.mark.parametrize(
    "version, expected",
    [
        ("0.0.1", "0.0.2"),
        ("1.4", "1.5"),
        ("0.0", "0.1"),
        ("1.0.0.0", "1.0.0.1"),
        ("alpha", "alpha.1"),
    ],
)
def test_bump_patch(version: str, expected: str) -> None:
    assert _bump_patch(version) == expected


# ---------- /skills list ----------


def test_skills_list_renders(client: TestClient) -> None:
    """GET /skills → 200 даже если в bundles_dir пусто."""
    r = client.get("/skills")
    assert r.status_code == 200


# ---------- POST /skills/{id} с bump_version=on ----------


def test_skill_edit_bump_version(
    client: TestClient, tmp_data_dir: Path, webui_env: Path
) -> None:
    """Создаём YAML-бандл, edit с bump_version=on, перечитываем — новая версия."""
    bundles_dir = tmp_data_dir / "skills"
    bundle_path = bundles_dir / "test_skill.yaml"

    skill = Skill(
        id="test_skill",
        name="test skill",
        description="initial",
        version="0.0.1",
        prompt_template="initial prompt",
        status=SkillStatus.ACTIVE,
        origin=SkillOrigin.OPERATOR,
    )
    data = skill.model_dump(mode="json", exclude_none=True)
    with bundle_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    r = client.post(
        "/skills/test_skill",
        data={
            "prompt_template": "updated prompt",
            "status": "active",
            "description": "updated desc",
            "bump_version": "on",
        },
        follow_redirects=False,
    )
    assert r.status_code == 303, f"body={r.text[:300]}"

    # Перечитываем YAML и проверяем bumped version + parent + prompt.
    with bundle_path.open(encoding="utf-8") as f:
        reloaded = yaml.safe_load(f)
    assert reloaded["version"] == "0.0.2"
    assert reloaded["parent_version_id"] == "0.0.1"
    assert reloaded["prompt_template"] == "updated prompt"
