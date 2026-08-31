"""Regression test for coexistence with an unrelated top-level models package."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_replite_namespace_does_not_collide_with_models(tmp_path) -> None:
    unrelated = tmp_path / "models"
    unrelated.mkdir()
    (unrelated / "__init__.py").write_text("OWNER = 'other-project'\n")
    repository = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(repository)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import models; "
                "from replite import RepLiteConfig, TaskConfig, create_replite_model; "
                "from replite.backbone import create_backbone; "
                "assert models.OWNER == 'other-project'; "
                "assert create_backbone('mobilenetv3_small_050'); "
                "assert create_replite_model(RepLiteConfig(tasks=TaskConfig(depth=True)))"
            ),
        ],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
