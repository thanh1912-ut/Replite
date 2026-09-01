"""Build the checked-in all-archive SANPO-Real main training notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "SANPO_Real_Main_Train.ipynb"


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


cells = [
    markdown(
        r"""
        # RepLite × SANPO-Real — main training trên 234 archive

        Notebook này **không tải lại data**. Nó dùng 234 archive đã có trên Drive,
        audit đủ `186 official-train + 48 official-test`, rồi chỉ chia 186 archive
        official-train thành train/val theo **session_id**. Official-test không được mở
        trong train, validation, early stopping hay chọn checkpoint.

        Luồng chạy:

        1. mount Drive, lấy source và cài runtime;
        2. khóa cấu hình campaign;
        3. audit data/split, dựng model và in backbone, feature stages, pretrained SHA,
           cấu hình neck/head, số parameter, optimizer/scheduler **trước khi train**;
        4. preflight bằng model dùng một lần, sau đó chạy đúng **epoch 1** trên toàn bộ
           train/val với log kiểu YOLO và metric mAP/mIoU/depth;
        5. chỉ khi gate epoch 1 đạt và bạn nhập approval token, strict-resume epoch 2
           để chạy hết campaign.

        Epoch 1 dùng ngay config/scheduler của toàn campaign; không phải một schedule
        1-epoch khác. Mỗi archive được stream-extract riêng lên SSD `/content`, train xong
        shard nào dọn shard đó, nên không cần giải nén đồng thời khoảng 187 GiB. Snapshot
        versioned có SHA-256 được mirror lên Drive sau từng epoch.
        """
    ),
    code(
        r"""
        #@title 1) Mount Drive, clone/pull RepLite và cài dependencies
        import json, os, shutil, subprocess, sys
        from pathlib import Path

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
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", REPO_REF], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)
        SOURCE_COMMIT = subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
        ).strip()

        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q",
                "timm==1.0.28", "huggingface_hub>=0.34,<2",
                "safetensors>=0.5,<1", "Pillow>=10,<13", "tqdm>=4.66,<5",
                "zstandard>=0.23,<1",
            ],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)],
            check=True,
        )
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        import torch
        assert torch.cuda.is_available(), "Chọn Runtime > Change runtime type > GPU"
        print("SOURCE COMMIT:", SOURCE_COMMIT)
        print("Python:", sys.version.split()[0], "| Torch:", torch.__version__)
        print("GPU:", torch.cuda.get_device_name(0))
        print("SSD free:", f"{shutil.disk_usage('/content').free / 1024**3:.1f} GiB")
        """
    ),
    code(
        r"""
        #@title 2) Cấu hình campaign — chỉnh ở đây trước khi chạy audit/pilot
        from datetime import datetime, timezone

        DRIVE_DATA_ROOT = Path("/content/drive/MyDrive/nckh1m_data/sanpo_real_v0_joint_human_only_rgb3")
        LOCAL_WORK_ROOT = Path("/content/replite_sanpo_main")
        DRIVE_RUNS_ROOT = DRIVE_DATA_ROOT / "main_runs"

        RUN_ID = "replite_sanpo_mnv4convs_seed42_v1"  #@param {type:"string"}
        BACKBONE_NAME = "mobilenetv4_conv_small"  #@param ["mobilenetv4_conv_small", "mobilenetv3_small_050"]
        PRETRAINED_IN1K = True  #@param {type:"boolean"}
        EPOCHS = 50  #@param {type:"integer"}
        IMAGE_HEIGHT = 288  #@param {type:"integer"}
        IMAGE_WIDTH = 512  #@param {type:"integer"}
        BATCH_SIZE = 4  #@param {type:"integer"}
        NUM_WORKERS = 2  #@param {type:"integer"}
        PREFETCH_FACTOR = 2  #@param {type:"integer"}
        SEED = 42  #@param {type:"integer"}
        VAL_FRACTION = 0.15  #@param {type:"number"}

        BASE_LR = 3e-4  #@param {type:"number"}
        BACKBONE_LR_MULTIPLIER = 0.1  #@param {type:"number"}
        WEIGHT_DECAY = 1e-2  #@param {type:"number"}
        WARMUP_FRACTION = 0.05  #@param {type:"number"}
        MIN_LR_RATIO = 0.05  #@param {type:"number"}
        GRAD_ACCUM_STEPS = 1  #@param {type:"integer"}
        GRAD_CLIP_NORM = 1.0  #@param {type:"number"}
        AMP_INITIAL_SCALE = 4096.0
        PROGRESS_EVERY_N_STEPS = 20
        MAX_PEAK_VRAM_GIB = 22.0

        assert DRIVE_DATA_ROOT.is_dir(), DRIVE_DATA_ROOT
        assert RUN_ID and "/" not in RUN_ID and RUN_ID not in {".", ".."}
        assert EPOCHS >= 2, "Campaign phải có ít nhất 2 epoch để gate sau epoch 1"
        assert IMAGE_HEIGHT % 32 == 0 and IMAGE_WIDTH % 32 == 0
        assert BATCH_SIZE > 0 and NUM_WORKERS >= 0 and PREFETCH_FACTOR > 0
        assert 0.0 < VAL_FRACTION < 0.5

        CAMPAIGN = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "source_repository": REPO_URL,
            "source_commit": SOURCE_COMMIT,
            "drive_data_root": str(DRIVE_DATA_ROOT),
            "drive_runs_root": str(DRIVE_RUNS_ROOT),
            "local_work_root": str(LOCAL_WORK_ROOT),
            "model": {
                "backbone_name": BACKBONE_NAME,
                "pretrained_in1k": PRETRAINED_IN1K,
                "recurrence_steps": 3,
                "recurrent_c4_channels": 48,
                "recurrent_c5_channels": 64,
                "neck_channels": 48,
                "dense_channels": 32,
                "task_adapter_channels": 32,
                "detection_head_channels": 48,
                "detection_head_blocks": 2,
                "detection_reg_max": 0,
                "use_sppf": False,
            },
            "data": {
                "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
                "clip_length": 3,
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "prefetch_factor": PREFETCH_FACTOR,
                "validation_fraction": VAL_FRACTION,
                "split_seed": SEED,
                "depth_min_metres": 0.1,
                "depth_max_metres": 80.0,
            },
            "train": {
                "epochs": EPOCHS,
                "seed": SEED,
                "base_lr": BASE_LR,
                "backbone_lr_multiplier": BACKBONE_LR_MULTIPLIER,
                "weight_decay": WEIGHT_DECAY,
                "warmup_fraction": WARMUP_FRACTION,
                "min_lr_ratio": MIN_LR_RATIO,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "grad_clip_norm": GRAD_CLIP_NORM,
                "amp": True,
                "amp_dtype": "float16",
                "amp_initial_scale": AMP_INITIAL_SCALE,
                "progress_every_n_steps": PROGRESS_EVERY_N_STEPS,
                "monitor": "val/total",
                "monitor_mode": "min",
                "max_peak_vram_gib": MAX_PEAK_VRAM_GIB,
            },
            "metrics": {
                "detection_score_threshold": 0.001,
                "detection_nms_iou_threshold": 0.6,
                "detection_max_detections": 300,
            },
        }
        CONFIG_DIR = LOCAL_WORK_ROOT / "configs"
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH = CONFIG_DIR / f"{RUN_ID}.json"
        CONFIG_PATH.write_text(
            json.dumps(CAMPAIGN, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        LOCAL_RUN_DIR = LOCAL_WORK_ROOT / "runs" / RUN_ID
        CONSOLE_LOG = LOCAL_RUN_DIR / "console.log"

        def run_live(arguments):
            # Stream output both below the cell and into console.log.
            LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, "-u", str(REPO_DIR / "tools/train_sanpo_main.py"),
                *arguments, "--config", str(CONFIG_PATH),
            ]
            with CONSOLE_LOG.open("a", encoding="utf-8", buffering=1) as log:
                process = subprocess.Popen(
                    command, cwd=REPO_DIR, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                return_code = process.wait()
            if return_code:
                tail = "\n".join(
                    CONSOLE_LOG.read_text(encoding="utf-8").splitlines()[-80:]
                )
                raise RuntimeError(
                    f"RepLite command failed with exit code {return_code}. "
                    f"Full log: {CONSOLE_LOG}\n\n{tail}"
                )

        print("Config:", CONFIG_PATH)
        print("Run Drive:", DRIVE_RUNS_ROOT / RUN_ID)
        print("LƯU Ý: đổi config sau pilot sẽ làm approval token vô hiệu.")
        """
    ),
    code(
        r"""
        #@title 3) Audit 234 archive + freeze split + xem cấu hình/model/parameter (KHÔNG train)
        run_live(["inspect"])
        """
    ),
    markdown(
        r"""
        ## Pilot chính thức: đúng 1 epoch

        Cell dưới đây trước tiên dùng một model disposable để kiểm forward/loss/backward,
        nên model production không bị đổi BatchNorm hay RNG. Sau đó nó train epoch 1 trên
        toàn train split, validation trên toàn val split, in log kiểu YOLO và lưu:

        - detection: mAP50, mAP50–95 và AP từng lớp;
        - segmentation: mIoU, pixel accuracy và IoU từng lớp;
        - depth: AbsRel, RMSE (m) và δ1;
        - `last.pt`, `best.pt`, history, resolved config và snapshot SHA-256 trên Drive.

        Dữ liệu detection của SANPO ở đây là box **dẫn xuất từ panoptic**, nên metric là
        metric nội bộ của protocol này, không được gọi là official SANPO detection benchmark.
        """
    ),
    code(
        r"""
        #@title 4) Chạy pilot epoch 1 và stream log kiểu YOLO
        run_live(["pilot"])
        print("\nCó thể xem lại/tail log tại:", CONSOLE_LOG)
        """
    ),
    code(
        r"""
        #@title 5) Xem gate và metric epoch 1
        DRIVE_RUN_DIR = DRIVE_RUNS_ROOT / RUN_ID
        PILOT_GATE_PATH = DRIVE_RUN_DIR / "pilot_gate.json"
        assert PILOT_GATE_PATH.is_file(), PILOT_GATE_PATH
        PILOT_GATE = json.loads(PILOT_GATE_PATH.read_text(encoding="utf-8"))
        print(json.dumps(PILOT_GATE, indent=2, ensure_ascii=False))
        assert PILOT_GATE["status"] == "pass", "Không được train main khi pilot chưa PASS"
        print("\nAPPROVAL TOKEN (copy sang Cell 6):")
        print(PILOT_GATE["approval_token"])
        print("\nTrong Colab Terminal có thể xem log bằng:")
        print(f"tail -f {CONSOLE_LOG}")
        """
    ),
    code(
        r"""
        #@title 6) APPROVE và strict-resume epoch 2 → EPOCHS
        START_MAIN = False  #@param {type:"boolean"}
        MAIN_APPROVAL_TOKEN = ""  #@param {type:"string"}

        if not START_MAIN:
            print("Main train CHƯA chạy. Xem metric Cell 5, rồi đặt START_MAIN=True và dán token.")
        else:
            assert MAIN_APPROVAL_TOKEN == PILOT_GATE["approval_token"], "Sai approval token"
            run_live(["train", "--approval-token", MAIN_APPROVAL_TOKEN])
        """
    ),
    markdown(
        r"""
        ## Resume sau khi Colab mất session

        Chạy lại Cell 1 → 3, rồi Cell 5 để đọc token từ Drive. Bỏ qua Cell 4 nếu đã có
        snapshot epoch 1. Sau đó chạy Cell 6. Lệnh `train` tự quét snapshot mới nhất xuống
        cũ, bỏ snapshot hỏng, kiểm SHA-256 cùng source/config/catalog/split hash rồi mới
        strict-load model, optimizer, scheduler, GradScaler, RNG, global step và AMP skip.

        Đừng đổi `RUN_ID` hay bất kỳ config nào khi resume. `console.log` được ghi ở local
        runtime; lịch sử epoch và checkpoint nguồn sự thật nằm trong snapshot Drive.
        """
    ),
]


notebook = {
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

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
)
print(OUTPUT)
