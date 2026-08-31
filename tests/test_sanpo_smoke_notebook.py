"""Static contract checks for the SANPO extract/visualize/smoke notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "SANPO_Real_Extract_Visualize_Smoke_Train.ipynb"


def _notebook_source() -> tuple[dict[str, object], str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, source


def test_notebook_is_clean_and_all_code_cells_compile() -> None:
    notebook, _ = _notebook_source()
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_notebook_locks_three_unique_pilot_sessions_and_smoke_range() -> None:
    _, source = _notebook_source()
    for session_id in (
        "eAXv0qwkO9zgffv9lQdSts_uRZCQI3Ro",
        "a7bGB6aD6bcUMhl86R9HNNHUVUUxlS2c",
        "zKsJQMv6IV6seRnaYa_gp6fYkiKFYR3h",
    ):
        assert session_id in source
    assert "SMOKE_EPOCHS = 3" in source
    assert "2 <= SMOKE_EPOCHS <= 5" in source
    assert "trainer.fit(train_loader, val_loader)" in source
    assert "trainer.fit(train_loader, official_test_loader)" not in source
    assert "Official-test không đi vào fit()" in source


def test_notebook_contains_integrity_extraction_and_training_gates() -> None:
    _, source = _notebook_source()
    required = (
        "copy_and_hash",
        "validate_tar_members",
        'line[0] in {"-", "d"}',
        '".." not in path.parts',
        "load_sanpo_joint_manifest",
        "sanpo_joint_collate",
        'segmentation_classes=len(SANPO_SEGMENTATION_CLASS_NAMES)',
        'monitor="val/total"',
        "trainer.amp_skip_count == 0",
        "artifact_manifest.json",
        "sha256_file(copied) == record[\"sha256\"]",
    )
    for marker in required:
        assert marker in source


def test_notebook_visualization_is_explicit_about_depth_and_raw_data() -> None:
    _, source = _notebook_source()
    assert 'plt.get_cmap("cividis")' in source
    assert 'label="Depth (m)"' in source
    assert '"raw_data_modified": False' in source
    assert '"invalid_depth_color": "gray"' in source
    assert "SANPO_DETECTION_CLASS_NAMES[int(label)]" in source

