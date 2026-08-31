"""Static and isolated-function tests for the self-contained Colab cell."""

from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image


CELL_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "download_sanpo_real_pilot.py"
)
CELL_SOURCE = CELL_PATH.read_text(encoding="utf-8")
CELL_TREE = ast.parse(CELL_SOURCE, filename=str(CELL_PATH))
NOTEBOOK_PATH = (
    CELL_PATH.parents[1]
    / "notebooks"
    / "SANPO_Real_Pilot_2train_1test_Detection.ipynb"
)


def _constant_assignments() -> dict[str, object]:
    values: dict[str, object] = {}
    for node in CELL_TREE.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            values[target.id] = node.value.value
    return values


def _load_functions(*names: str, namespace: dict[str, object]) -> dict[str, object]:
    selected = [
        node
        for node in CELL_TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(CELL_PATH), "exec"), namespace)
    return namespace


def test_cell_defaults_to_exact_three_session_pilot() -> None:
    values = _constant_assignments()

    assert values["ACTION"] == "download"
    assert values["ANNOTATION_POLICY"] == "human_only"
    assert values["INCLUDE_OFFICIAL_TEST"] is True
    assert values["MAX_TRAIN_SESSION_CAMERAS"] == 2
    assert values["MAX_TEST_SESSION_CAMERAS"] == 1
    assert values["DETECTION_MIN_COMPONENT_PIXELS"] == 100


def test_cell_selection_never_uses_two_cameras_from_same_session() -> None:
    records = [
        {"split": "train", "session_id": "same", "sensor": "camera_head"},
        {"split": "train", "session_id": "same", "sensor": "camera_chest"},
        {"split": "train", "session_id": "other", "sensor": "camera_head"},
        {"split": "train", "session_id": "third", "sensor": "camera_head"},
    ]
    namespace = _load_functions(
        "seeded_order",
        "select_split",
        namespace={"hashlib": hashlib, "qualified_records": records},
    )

    selected = namespace["select_split"]("train", 2)

    assert len(selected) == 2
    assert len({record["session_id"] for record in selected}) == 2


def test_cell_panoptic_converter_keeps_iid_zero_and_splits_components() -> None:
    namespace = _load_functions(
        "_find_component_root",
        "_union_components",
        "_thing_runs",
        "_extract_panoptic_components",
        namespace={
            "np": np,
            "Counter": __import__("collections").Counter,
            "source_to_detection": {12: 3, 28: 14},
        },
    )
    mask = np.zeros((6, 8, 3), dtype=np.uint8)
    mask[0, 0] = [12, 0, 7]
    mask[1, 1] = [12, 0, 7]
    mask[3:5, 1:3] = [28, 0, 0]
    mask[3:5, 5:7] = [28, 0, 0]

    components, _ = namespace["_extract_panoptic_components"](mask)

    assert components == [
        {
            "area": 2,
            "box": [0, 0, 2, 2],
            "source_semantic_id": 12,
            "instance_id": 7,
            "detection_class_id": 3,
            "component_index": 0,
        },
        {
            "area": 4,
            "box": [1, 3, 3, 5],
            "source_semantic_id": 28,
            "instance_id": 0,
            "detection_class_id": 14,
            "component_index": 0,
        },
        {
            "area": 4,
            "box": [5, 3, 7, 5],
            "source_semantic_id": 28,
            "instance_id": 0,
            "detection_class_id": 14,
            "component_index": 1,
        },
    ]


def test_cell_generates_detection_before_packaging_and_versions_archive() -> None:
    derive_at = CELL_SOURCE.index("detection_target = derive_detection_target(")
    archive_at = CELL_SOURCE.index("local_archive = LOCAL_ARCHIVES / archive_name")

    assert derive_at < archive_at
    assert 'enriched_sample["detection_path"] = detection_relative' in CELL_SOURCE
    assert '"detection_config_sha256": detection_config_sha' in CELL_SOURCE
    assert '"package_sha256": package_sha' in CELL_SOURCE
    assert '"valid_negative": len(positives) == 0' in CELL_SOURCE


def test_cell_writes_direct_replite_target_contract(tmp_path: Path) -> None:
    namespace = _load_functions(
        "_find_component_root",
        "_union_components",
        "_thing_runs",
        "_extract_panoptic_components",
        "derive_detection_target",
        namespace={
            "np": np,
            "Counter": __import__("collections").Counter,
            "Path": Path,
            "Image": Image,
            "io": io,
            "hashlib": hashlib,
            "source_to_detection": {21: 8},
            "DETECTION_MIN_COMPONENT_PIXELS": 1,
            "detection_config": {
                "bbox_format": "XYXY half-open, absolute target-RGB pixels"
            },
            "detection_config_sha": "config-sha",
        },
    )
    rgb_path = tmp_path / "000010.png"
    mask_path = tmp_path / "000010_mask.png"
    Image.new("RGB", (5, 4)).save(rgb_path)
    mask = np.zeros((4, 5, 3), dtype=np.uint8)
    mask[1:3, 2:5] = [21, 0, 0]
    Image.fromarray(mask, mode="RGB").save(mask_path)
    sample = {
        "target_frame": 10,
        "annotation_type": "HUMAN_ANNOTATED",
        "rgb_context_paths": ["8.png", "9.png", "10.png"],
        "panoptic_path": "10_mask.png",
    }

    target = namespace["derive_detection_target"](
        mask_path, rgb_path, sample
    )

    assert target["boxes"] == [[2.0, 1.0, 5.0, 3.0]]
    assert target["labels"] == [8]
    assert target["instance_ids"] == [0]
    assert target["valid_size"] == [4, 5]
    assert target["ignore_boxes"] == []
    assert target["valid_negative"] is False


def test_notebook_contains_the_exact_checked_cell_without_stale_outputs() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]

    assert notebook["nbformat"] == 4
    assert len(code_cells) == 1
    assert "".join(code_cells[0]["source"]) == CELL_SOURCE
    assert code_cells[0]["execution_count"] is None
    assert code_cells[0]["outputs"] == []
