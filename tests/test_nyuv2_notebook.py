"""Static contracts for the checked-in NYUDv2 Colab workflow."""

from __future__ import annotations

import json
from pathlib import Path

from tools import build_nyuv2_train_notebook as builder


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "NYUDv2_SegDepth_Train.ipynb"
GENERATOR = ROOT / "tools" / "build_nyuv2_train_notebook.py"
RUNNER = ROOT / "tools" / "train_nyuv2.py"


def _sources() -> tuple[dict[str, object], str, str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_source = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    all_source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )
    return notebook, code_source, all_source


def test_nyuv2_notebook_is_clean_generated_and_compiles() -> None:
    notebook, _, _ = _sources()
    assert notebook == builder.build_notebook()
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["nbformat"] == 4
    assert NOTEBOOK.name in GENERATOR.read_text(encoding="utf-8")
    compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"nyuv2-notebook-cell-{index}", "exec")


def test_nyuv2_notebook_locks_archive_schema_and_static_input() -> None:
    _, source, all_source = _sources()
    for marker in (
        'REPO_REF = "8811440d23fda2dc6db1c9dddb80c870f6654d06"',
        'ARCHIVE_PATH = Path("/content/drive/MyDrive/datasets/NYUDv2/NYUDv2.tar.gz")',
        'ARCHIVE_DRIVE_FILE_ID = "14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw"',
        "ARCHIVE_EXPECTED_BYTES = 4_215_751_725",
        'ARCHIVE_SHA256 = "33338d895404a9144a2c6892a8b0d6d5c26b02021f945b12e36c431fb369fcb2"',
        'LOCAL_ARCHIVE_STAGE = Path("/content/.replite_nyuv2_cache/NYUDv2.tar.gz")',
        'LOCAL_DATASET_ROOT = Path("/content/nyudv2")',
        '"expected_train_samples": 795',
        '"expected_test_samples": 654',
        '"image_size": [288, 384]',
        '"num_classes": 40',
        '"ignore_index": 255',
        '"source_ignore_labels": [0]',
        '"expected_raw_label_ids": list(range(41))',
        '"depth_unit_scale": 1.0',
        '"google_drive_file_id": ARCHIVE_DRIVE_FILE_ID',
        '"local_stage_path": str(LOCAL_ARCHIVE_STAGE)',
        '"download_retries": 4',
        '"download_timeout_seconds": 900',
        'raw_label_mapping = {str(raw_id): raw_id - 1 for raw_id in range(1, 41)}',
        'SOURCE_PIN = DRIVE_RUN_DIR / "source_pin.json"',
        '["git", "-C", str(REPO_DIR), "checkout", "--detach", pinned_commit]',
        "assert SOURCE_COMMIT == pinned_commit",
        'EXPECTED_PROTOCOL_ID = "replite-nyuv2-segdepth-v2"',
        'str(REPO_DIR / "tools/train_nyuv2.py")',
        '"protocol-info"',
        'source_pin.get("schema_version") == 2',
        'source_pin.get("protocol_id") == EXPECTED_PROTOCOL_ID',
        '"protocol_id": EXPECTED_PROTOCOL_ID',
        "Source pin cũ/không tương thích; dùng RUN_ID mới",
    ):
        assert marker in source
    assert 'not LOCAL_ARCHIVE_STAGE.exists()' in source
    assert '"gdown>=6,<7"' in source
    assert 'gdown --continue' in all_source
    assert "copy2" not in source
    assert "shutil.copy" not in source


def test_nyuv2_notebook_locks_model_conflict_fix_and_light_augmentation() -> None:
    _, source, _ = _sources()
    for marker in (
        '"active_tasks": ["segmentation", "depth"]',
        '"backbone_name": "mobilenetv4_conv_small"',
        '"pretrained_in1k": True',
        '"dense_fusion_direction": "seg_to_depth"',
        '"dense_fusion_detach_source": True',
        '"dense_decoder": "dense_v2_s"',
        '"segmentation_auxiliary": True',
        '"task_weights": {"segmentation": 1.0, "depth": 1.0}',
        '"horizontal_flip_probability": 0.5',
        '"scale_min": 0.75',
        '"scale_max": 1.25',
        '"brightness": 0.10',
        '"contrast": 0.10',
        '"saturation": 0.08',
        '"class_aware_crop_probability": 0.35',
        '"blur_probability": 0.08',
        '"segmentation_lovasz_weight": 0.25',
        '"depth_loss_type": "per_image_silog_log_l1_gradient"',
        '"single_task_anchors": SINGLE_TASK_ANCHORS',
    ):
        assert marker in source
    assert "detection_classes" not in source
    assert "detection_head" not in source
    assert "mAP" not in source


def test_nyuv2_notebook_uses_inner_val_only_for_selection_then_final_test() -> None:
    _, source, all_source = _sources()
    for marker in (
        '"protocol_id": EXPECTED_PROTOCOL_ID',
        '"mode": "multitask"',
        '"stage": "main"',
        '"inner_validation_fraction": 0.10',
        '"split_seed": 42',
        '"monitor": "val/selection/joint"',
        '"monitor_mode": "max"',
        '"early_stopping_patience": 10',
        '"early_stopping_min_delta": 0.0',
        "strict-load `best.pt`",
        "Không chọn checkpoint theo train loss",
        "official test only after strict best.pt",
    ):
        assert marker in (source if marker.startswith(('"', "official")) else all_source)
    assert 'run_cli("test")' not in source
    assert 'run_cli("evaluate-test")' in source
    assert source.index('run_cli("inspect")') < source.index('run_cli("train"')


def test_nyuv2_notebook_has_runner_actions_resume_and_visible_live_log() -> None:
    _, source, _ = _sources()
    assert RUNNER.name in source
    assert 'run_cli("extract")' in source
    assert 'run_cli("inspect")' in source
    assert 'run_cli("train", *resume_arguments)' in source
    assert 'RESUME = False  #@param {type:"boolean"}' in source
    assert 'RUN_ID = "replite_nyuv2_mnv4convs_segdepth_seed42_v2_r2"' in source
    assert 'SEG_ONLY_MIOU_ANCHOR = 0.0' in source
    assert 'DEPTH_ONLY_ABSREL_ANCHOR = 0.0' in source
    assert 'resume_arguments = ["--resume"] if RESUME else []' in source
    assert "stdout=subprocess.PIPE" in source
    assert "stderr=subprocess.STDOUT" in source
    assert 'f"tail -n 80 -F {CONSOLE_LOG}"' in source
    assert 'CONSOLE_LOG.open("w"' in source
    assert 'DRIVE_RUN_DIR / "official_test_metrics.json"' in source
    assert 'DRIVE_RUN_DIR / "inner_split_manifest.json"' in source
    assert source.index("probe_runner_protocol()") < source.index(
        "temporary_pin.write_text"
    )
    # One function definition plus probes before creating a pin and after
    # checking out an existing pin.
    assert source.count("probe_runner_protocol()") == 3


def test_nyuv2_notebook_readme_link_and_runner_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "notebooks/NYUDv2_SegDepth_Train.ipynb" in readme
    assert RUNNER.is_file()
    compile(RUNNER.read_text(encoding="utf-8"), str(RUNNER), "exec")
