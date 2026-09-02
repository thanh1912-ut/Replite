"""Build the checked-in NYUDv2 segmentation/depth Colab notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "NYUDv2_SegDepth_Train.ipynb"


def _lines(source: str) -> list[str]:
    return (textwrap.dedent(source).strip("\n") + "\n").splitlines(keepends=True)


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def build_notebook() -> dict[str, object]:
    cells = [
        markdown(
            r"""
            # RepLite × NYUDv2 — semantic segmentation + metric depth

            Đây là notebook train chính cho hai task **segmentation + depth** ở
            kích thước tĩnh **288×384**. Detection không được dựng, không có loss
            detection và không có synthetic video/clip. MobileNetV4-Conv-S dùng
            pretrained ImageNet-1K; LiteConvLSTM hoạt động như iterative feature
            refinement trên một ảnh tĩnh.

            Protocol chống leakage:

            - archive trên Drive được kiểm đúng byte + SHA-256 rồi giải nén thẳng
              vào SSD `/content/nyudv2`; notebook **không copy file 3.93 GiB** sang SSD;
            - official train (795 ảnh) được chia cố định bằng seed 42 thành fit và
              10% inner-validation;
            - `best.pt` và early stopping chỉ theo `val/total`, patience 10;
            - official held-out split (654 ảnh; bundle gọi là `val`) bị khóa trong
              toàn bộ quá trình chọn model. Runner chỉ được mở split này **sau khi
              strict-load `best.pt`**, rồi test đúng một lần.

            Không chọn checkpoint theo train loss và tuyệt đối không nhìn official
            test để tuning. Làm vậy sẽ biến test thành validation và làm kết quả
            benchmark không còn hợp lệ.

            Log train/validation được stream theo kiểu YOLO. Trong lúc cell train
            đang chạy, mở Colab **Terminal** và dùng lệnh `tail -F` được in ở Cell 2.
            Checkpoint, resolved config, split manifest, history và final test metric
            nằm trên Drive để không mất khi runtime Colab ngắt.
            """
        ),
        code(
            r"""
            #@title 1) Mount Drive, lấy source và kiểm GPU
            import json
            import os
            import shutil
            import subprocess
            import sys
            from pathlib import Path

            import torch
            from google.colab import drive
            drive.mount("/content/drive")

            REPO_URL = "https://github.com/thanh1912-ut/Replite.git"
            REPO_REF = "main"  #@param {type:"string"}
            REPO_DIR = Path("/content/Replite")

            if not (REPO_DIR / ".git").is_dir():
                assert not REPO_DIR.exists(), f"Path tồn tại nhưng không phải Git repo: {REPO_DIR}"
                subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
            else:
                remote = subprocess.check_output(
                    ["git", "-C", str(REPO_DIR), "remote", "get-url", "origin"], text=True
                ).strip()
                assert remote.rstrip("/").removesuffix(".git") == REPO_URL.rstrip("/").removesuffix(".git"), remote
                subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=True)

            immutable_ref = (
                len(REPO_REF) == 40
                and all(character in "0123456789abcdef" for character in REPO_REF.lower())
            )
            subprocess.run(
                ["git", "-C", str(REPO_DIR), "checkout", *( ["--detach"] if immutable_ref else [] ), REPO_REF],
                check=True,
            )
            if not immutable_ref:
                subprocess.run(
                    ["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", REPO_REF],
                    check=True,
                )
            SOURCE_COMMIT = subprocess.check_output(
                ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
            ).strip()

            subprocess.run(
                [
                    sys.executable, "-m", "pip", "install", "-q",
                    "timm==1.0.28", "huggingface_hub>=0.34,<2",
                    "safetensors>=0.5,<1", "Pillow>=10,<13", "tqdm>=4.66,<5",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)],
                check=True,
            )
            if str(REPO_DIR) not in sys.path:
                sys.path.insert(0, str(REPO_DIR))

            assert torch.cuda.is_available(), "Chọn Runtime > Change runtime type > GPU"
            print("SOURCE COMMIT:", SOURCE_COMMIT)
            print("Python:", sys.version.split()[0], "| Torch:", torch.__version__)
            print("GPU:", torch.cuda.get_device_name(0))
            print("SSD free:", f"{shutil.disk_usage('/content').free / 1024**3:.1f} GiB")
            """
        ),
        code(
            r"""
            #@title 2) Khóa protocol, cấu hình model/train và đường dẫn artifact
            ARCHIVE_PATH = Path("/content/drive/MyDrive/datasets/NYUDv2/NYUDv2.tar.gz")
            ARCHIVE_EXPECTED_BYTES = 4_215_751_725
            ARCHIVE_SHA256 = "33338d895404a9144a2c6892a8b0d6d5c26b02021f945b12e36c431fb369fcb2"
            LOCAL_DATASET_ROOT = Path("/content/nyudv2")
            LOCAL_WORK_ROOT = Path("/content/replite_nyuv2")
            DRIVE_RUNS_ROOT = Path("/content/drive/MyDrive/datasets/NYUDv2/replite_runs")

            RUN_ID = "replite_nyuv2_mnv4convs_segdepth_seed42_v1"  #@param {type:"string"}
            BATCH_SIZE = 16  #@param {type:"integer"}
            NUM_WORKERS = 4  #@param {type:"integer"}
            EPOCHS = 100  #@param {type:"integer"}
            RESUME = False  #@param {type:"boolean"}

            assert RUN_ID and "/" not in RUN_ID and "\\" not in RUN_ID and RUN_ID not in {".", ".."}
            assert BATCH_SIZE > 0 and NUM_WORKERS >= 0 and EPOCHS > 0
            assert ARCHIVE_PATH.is_file(), f"Không tìm thấy archive: {ARCHIVE_PATH}"

            DRIVE_RUN_DIR = DRIVE_RUNS_ROOT / RUN_ID
            CONFIG_DIR = LOCAL_WORK_ROOT / "configs"
            CONFIG_PATH = CONFIG_DIR / f"{RUN_ID}.json"
            CONSOLE_LOG = LOCAL_WORK_ROOT / "runs" / RUN_ID / "console.log"
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            CONSOLE_LOG.parent.mkdir(parents=True, exist_ok=True)
            DRIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)

            # Pin the source commit per RUN_ID so a Colab reconnect remains an
            # exact resume even when the repository's main branch advances.
            SOURCE_PIN = DRIVE_RUN_DIR / "source_pin.json"
            if SOURCE_PIN.exists():
                source_pin = json.loads(SOURCE_PIN.read_text(encoding="utf-8"))
                assert source_pin.get("schema_version") == 1, source_pin
                assert source_pin.get("run_id") == RUN_ID, source_pin
                assert source_pin.get("source_repository") == REPO_URL, source_pin
                pinned_commit = source_pin.get("source_commit")
                assert (
                    isinstance(pinned_commit, str)
                    and len(pinned_commit) == 40
                    and all(character in "0123456789abcdef" for character in pinned_commit.lower())
                ), source_pin
                if SOURCE_COMMIT != pinned_commit:
                    available = subprocess.run(
                        ["git", "-C", str(REPO_DIR), "cat-file", "-e", f"{pinned_commit}^{{commit}}"],
                        check=False,
                    )
                    if available.returncode:
                        subprocess.run(
                            ["git", "-C", str(REPO_DIR), "fetch", "origin", pinned_commit],
                            check=True,
                        )
                    subprocess.run(
                        ["git", "-C", str(REPO_DIR), "checkout", "--detach", pinned_commit],
                        check=True,
                    )
                    SOURCE_COMMIT = subprocess.check_output(
                        ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
                    ).strip()
                assert SOURCE_COMMIT == pinned_commit
                print("RESUME SOURCE PIN:", SOURCE_COMMIT)
            else:
                source_pin = {
                    "schema_version": 1,
                    "run_id": RUN_ID,
                    "source_repository": REPO_URL,
                    "source_commit": SOURCE_COMMIT,
                }
                temporary_pin = SOURCE_PIN.with_suffix(".json.tmp")
                temporary_pin.write_text(
                    json.dumps(source_pin, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_pin, SOURCE_PIN)
                print("NEW SOURCE PIN:", SOURCE_COMMIT)

            raw_label_mapping = {str(raw_id): raw_id - 1 for raw_id in range(1, 41)}
            CONFIG = {
                "schema_version": 1,
                "protocol_id": "replite-nyuv2-segdepth-v1",
                "run_id": RUN_ID,
                "source_repository": REPO_URL,
                "source_commit": SOURCE_COMMIT,
                "archive": {
                    "path": str(ARCHIVE_PATH),
                    "expected_bytes": ARCHIVE_EXPECTED_BYTES,
                    "sha256": ARCHIVE_SHA256,
                    "expected_train_samples": 795,
                    "expected_test_samples": 654,
                },
                "paths": {
                    "local_dataset_root": str(LOCAL_DATASET_ROOT),
                    "local_work_root": str(LOCAL_WORK_ROOT),
                    "drive_run_root": str(DRIVE_RUN_DIR),
                },
                "model": {
                    "active_tasks": ["segmentation", "depth"],
                    "backbone_name": "mobilenetv4_conv_small",
                    "pretrained_in1k": True,
                    "recurrence_steps": 3,
                    "recurrent_c4_channels": 48,
                    "recurrent_c5_channels": 64,
                    "neck_channels": 48,
                    "dense_channels": 32,
                    "task_adapter_channels": 32,
                    "use_sppf": False,
                    "dense_fusion_direction": "seg_to_depth",
                    "dense_fusion_detach_source": True,
                },
                "data": {
                    "image_size": [288, 384],
                    "batch_size": BATCH_SIZE,
                    "num_workers": NUM_WORKERS,
                    "prefetch_factor": 2,
                    "num_classes": 40,
                    "ignore_index": 255,
                    "raw_label_mapping": raw_label_mapping,
                    "source_ignore_labels": [0],
                    "expected_raw_label_ids": list(range(41)),
                    "depth_unit_scale": 1.0,
                    "depth_min_metres": 0.1,
                    "depth_max_metres": 10.0,
                    "inner_validation_fraction": 0.10,
                    "split_seed": 42,
                    "augmentation": {
                        "horizontal_flip_probability": 0.5,
                        "scale_min": 1.0,
                        "scale_max": 1.10,
                        "brightness": 0.10,
                        "contrast": 0.10,
                        "saturation": 0.08,
                    },
                },
                "train": {
                    "epochs": EPOCHS,
                    "seed": 42,
                    "base_lr": 3e-4,
                    "backbone_lr_multiplier": 0.1,
                    "weight_decay": 1e-2,
                    "warmup_fraction": 0.05,
                    "min_lr_ratio": 0.05,
                    "grad_accum_steps": 1,
                    "grad_clip_norm": 1.0,
                    "amp": True,
                    "amp_dtype": "float16",
                    "amp_initial_scale": 1024.0,
                    "progress_every_n_steps": 10,
                    "monitor": "val/total",
                    "monitor_mode": "min",
                    "early_stopping_patience": 10,
                    "early_stopping_min_delta": 0.0,
                    "task_weights": {"segmentation": 1.0, "depth": 0.25},
                },
            }

            payload = json.dumps(CONFIG, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            if CONFIG_PATH.exists():
                existing = CONFIG_PATH.read_text(encoding="utf-8")
                assert existing == payload, (
                    "Config của RUN_ID hiện có không khớp. Đổi RUN_ID cho thí nghiệm mới; "
                    "không ghi đè campaign cũ."
                )
            else:
                temporary = CONFIG_PATH.with_suffix(".json.tmp")
                temporary.write_text(payload, encoding="utf-8")
                os.replace(temporary, CONFIG_PATH)

            def run_cli(action, *extra_arguments):
                command = [
                    sys.executable, "-u", str(REPO_DIR / "tools/train_nyuv2.py"),
                    action, "--config", str(CONFIG_PATH), *extra_arguments,
                ]
                print("Running:", " ".join(map(str, command)), flush=True)
                print("Xem cùng log trong Colab Terminal:", flush=True)
                print(f"tail -n 80 -F {CONSOLE_LOG}", flush=True)
                with CONSOLE_LOG.open("w", encoding="utf-8", buffering=1) as log_file:
                    process = subprocess.Popen(
                        command,
                        cwd=REPO_DIR,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    assert process.stdout is not None
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log_file.write(line)
                    return_code = process.wait()
                if return_code:
                    raise RuntimeError(
                        f"NYUDv2 action {action!r} failed with exit code {return_code}; "
                        f"xem {CONSOLE_LOG}"
                    )

            print("CONFIG:", CONFIG_PATH)
            print("DRIVE RUN:", DRIVE_RUN_DIR)
            print("INPUT: B×3×288×384 (static RGB)")
            print("TASKS: segmentation(40) + depth(m); detection=OFF")
            print("SELECTION: inner val/total; patience=10; official test only after strict best.pt")
            print("TERMINAL:", f"tail -n 80 -F {CONSOLE_LOG}")
            """
        ),
        markdown(
            r"""
            ## Giải nén vào SSD Colab

            Cell này đọc archive trực tiếp từ Drive, kiểm kích thước và SHA-256,
            preflight dung lượng, rồi safe-extract vào `/content/nyudv2`. Archive
            không được copy thành `/content/NYUDv2.tar.gz`. Lần chạy lại sẽ dùng
            completion manifest nếu dataset local còn nguyên; `/content` sẽ mất khi
            Colab reset nên lúc đó chỉ cần chạy lại cell này.
            """
        ),
        code(
            r"""
            #@title 3) Verify archive và extract trực tiếp Drive → /content/nyudv2
            run_cli("extract")
            assert LOCAL_DATASET_ROOT.is_dir(), LOCAL_DATASET_ROOT
            assert not Path("/content/NYUDv2.tar.gz").exists(), "Notebook không được copy archive vào SSD"
            print("Dataset local:", LOCAL_DATASET_ROOT)
            print("SSD free:", f"{shutil.disk_usage('/content').free / 1024**3:.1f} GiB")
            """
        ),
        markdown(
            r"""
            ## Audit trước khi train

            `inspect` phải xác nhận 795 official-train + 654 official-held-out,
            khóa inner-validation 10% từ official train và in model/backbone,
            pretrained provenance, parameter, batch/update count, augmentation,
            fusion direction và task weights. Cell này không train và không mở ảnh
            official-held-out để tính metric.
            """
        ),
        code(
            r"""
            #@title 4) Audit data/split và in toàn bộ cấu hình (KHÔNG train)
            run_cli("inspect")
            """
        ),
        markdown(
            r"""
            ## Train, early stop và final test

            Train dùng augmentation nhẹ đồng bộ RGB/segmentation/depth. Gradient và
            loss được chuẩn hóa theo **số mẫu**, còn batch sampler phân phối phần dư
            đều thay vì tạo singleton tail batch. Vì vậy batch cuối không còn có sức
            nặng gấp nhiều lần batch thường.

            `val/total` trên inner-validation là metric duy nhất để lưu `best.pt` và
            early stop. Nếu không giảm trong 10 epoch validation liên tiếp, train dừng.
            Sau đó runner tạo model đánh giá riêng, strict-load `best.pt`, mới mở
            official held-out split và chạy test đúng một lần. Không dùng official-test
            metric để quay lại chỉnh campaign này.
            """
        ),
        code(
            r"""
            #@title 5) Train main với log kiểu YOLO; resume nếu được bật ở Cell 2
            resume_arguments = ["--resume"] if RESUME else []
            run_cli("train", *resume_arguments)
            """
        ),
        code(
            r"""
            #@title 6) Xem artifact và metric final
            print("DRIVE RUN:", DRIVE_RUN_DIR)
            print("BEST:", DRIVE_RUN_DIR / "checkpoints/best.pt")
            print("LAST:", DRIVE_RUN_DIR / "checkpoints/last.pt")
            print("FINAL TEST:", DRIVE_RUN_DIR / "official_test_metrics.json")
            print("SPLIT MANIFEST:", DRIVE_RUN_DIR / "inner_split_manifest.json")
            print("RESOLVED CONFIG:", DRIVE_RUN_DIR / "resolved_config.json")
            print("LOG hiện tại:", CONSOLE_LOG)
            if (DRIVE_RUN_DIR / "official_test_metrics.json").is_file():
                final_metrics = json.loads(
                    (DRIVE_RUN_DIR / "official_test_metrics.json").read_text(encoding="utf-8")
                )
                print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
            else:
                print("Chưa có final test metric; xem console log để biết train đang ở đâu.")
            """
        ),
        markdown(
            r"""
            ## Resume sau khi Colab mất runtime

            Chạy lại Cell 1 → 3 vì SSD `/content` là tạm thời, rồi đặt `RESUME=True`
            ở Cell 2 và chạy Cell 4 → 5. Không đổi `RUN_ID` hay config. Resume phải
            strict-load `last.pt` cùng optimizer, scheduler, scaler, RNG, global step,
            best monitor và early-stop counter. Official test vẫn bị khóa cho đến khi
            selection kết thúc và `best.pt` được strict-load.
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
