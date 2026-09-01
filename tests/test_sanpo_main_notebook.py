"""Static safety and orchestration contracts for SANPO main training."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "SANPO_Real_Main_Train.ipynb"
GENERATOR = ROOT / "tools" / "build_sanpo_main_train_notebook.py"
RUNNER = ROOT / "tools" / "train_sanpo_main.py"


def _source() -> tuple[dict[str, object], str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, source


def test_main_notebook_is_clean_generated_and_compiles() -> None:
    notebook, _ = _source()
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["nbformat"] == 4
    compile(GENERATOR.read_text(encoding="utf-8"), str(GENERATOR), "exec")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")


def test_main_notebook_has_visible_locked_config_and_one_epoch_gate() -> None:
    _, source = _source()
    for marker in (
        'BACKBONE_NAME = "mobilenetv4_conv_small"',
        "PRETRAINED_IN1K = True",
        "EPOCHS = 50",
        "IMAGE_HEIGHT = 288",
        "IMAGE_WIDTH = 512",
        "BATCH_SIZE = 4",
        "VAL_FRACTION = 0.15",
        'DRIVE_BASE_ROOT = Path("/content/drive/MyDrive/nckh1m_data")',
        'RUN_ID = "replite_sanpo_mnv4convs_subset20_seed42_v1"',
        'LOCAL_STAGE_CACHE_ID = "replite_sanpo_mnv4convs_seed42_v4"',
        "OFFICIAL_TRAIN_SESSION_LIMIT = 20",
        '"official_train_session_limit": OFFICIAL_TRAIN_SESSION_LIMIT',
        '"cache_id": LOCAL_STAGE_CACHE_ID',
        "PROGRESS_EVERY_N_STEPS = 10",
        'print("SANPO DATA ROOT:", DRIVE_DATA_ROOT)',
        "LOCAL_STAGE_EXPANSION_FACTOR = 1.03",
        'SOURCE_PIN = DRIVE_RUNS_ROOT / RUN_ID / "source_pin.json"',
        '"checkout", "--detach", pinned_commit',
        'assert SOURCE_COMMIT == pinned_commit',
        'run_live(["stage-train"])',
        '"monitor": "val/total"',
        'run_live(["pilot"])',
        'run_live(["inspect"])',
        "START_MAIN = False",
        "MAIN_APPROVAL_TOKEN",
        'run_live(["train", "--approval-token", MAIN_APPROVAL_TOKEN])',
        'run_live(["stage-test"])',
        "def run_live(arguments)",
        "tail -F",
        "Stage subset 20 session official-train lên SSD",
    ):
        assert marker in source


def test_main_notebook_delegates_audited_streaming_and_never_downloads() -> None:
    _, source = _source()
    assert "train_sanpo_main.py" in source
    assert 'run_live(["inspect"])' in source
    assert "RepLite command failed with exit code" in source
    assert "download_sanpo_real_pilot.py" not in source
    assert "gdown" not in source
    assert "official-test không được mở" in NOTEBOOK.read_text(encoding="utf-8").lower()


def test_main_notebook_keeps_command_logs_isolated_and_visible() -> None:
    _, source = _source()
    assert 'CONSOLE_LOG.open("w"' in source
    assert 'command_log.open("x"' in source
    assert 'CONSOLE_LOG.open("a"' not in source
    assert 'fcntl.LOCK_EX | fcntl.LOCK_NB' in source
    assert 'f"[run_live] START action={action}' in source
    assert 'f"[run_live] WAIT action={action}' in source
    assert "silent={silent}s (process still running)" in source
    assert "now - last_child_output >= 30.0" in source
    assert source.count("last_child_output = time.monotonic()") >= 2
    assert "[run_live] HEARTBEAT" not in source
    assert 'f"[run_live] END action={action}' in source
    assert 'f"[run_live] INTERRUPT action={action}' in source
    assert "start_new_session=True" in source
    assert "os.killpg(process.pid, signal.SIGINT)" in source
    assert '"stage-train", "pilot", "train", "stage-test"' in source


def test_main_runner_exists_and_compiles() -> None:
    assert RUNNER.is_file()
    compile(RUNNER.read_text(encoding="utf-8"), str(RUNNER), "exec")
