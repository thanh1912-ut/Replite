"""Build the checked-in SANPO extraction, visualization, and smoke notebook."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "SANPO_Real_Extract_Visualize_Smoke_Train.ipynb"


def _lines(source: str) -> list[str]:
    text = textwrap.dedent(source).strip("\n") + "\n"
    return text.splitlines(keepends=True)


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
        # RepLite × SANPO-Real — extract, visualize, smoke train

        Notebook này xử lý đúng **3 archive pilot đã tải**: hai session official-train và một
        session official-test. Quy trình:

        1. kiểm byte count, SHA-256, sidecar và member path trước khi giải nén;
        2. giải nén lên SSD `/content` (không train trực tiếp trên Drive);
        3. trực quan RGB, detection box dẫn xuất, semantic mask và metric depth;
        4. smoke train mặc định **3 epoch** trên một train session, validation bằng train
           session còn lại; official-test không tham gia `fit()` hoặc chọn checkpoint;
        5. kiểm checkpoint rồi mới mirror run bundle có checksum lên Drive.

        Detection box được dẫn xuất từ official panoptic mask, **không phải official SANPO
        detection benchmark**. Chạy tuần tự từ trên xuống. Runtime cần GPU.
        """
    ),
    code(
        r"""
        #@title 1) Mount Drive, lấy source RepLite và cài runtime
        import json, os, random, shutil, subprocess, sys
        from datetime import datetime, timezone
        from pathlib import Path, PurePosixPath

        from google.colab import drive
        drive.mount("/content/drive")

        REPO_URL = "https://github.com/thanh1912-ut/Replite.git"
        REPO_REF = "main"  #@param {type:"string"}
        REPO_DIR = Path("/content/Replite")

        if not (REPO_DIR / ".git").is_dir():
            assert not REPO_DIR.exists(), f"Có path không phải Git repo: {REPO_DIR}"
            subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
        else:
            remote = subprocess.check_output(
                ["git", "-C", str(REPO_DIR), "remote", "get-url", "origin"], text=True
            ).strip()
            assert remote.rstrip("/").removesuffix(".git") == REPO_URL.rstrip("/").removesuffix(".git"), remote
            subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)

        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", REPO_REF], check=True)
        SOURCE_COMMIT = subprocess.check_output(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], text=True
        ).strip()

        if shutil.which("zstd") is None:
            subprocess.run(["apt-get", "update", "-qq"], check=True)
            subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], check=True)

        subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "-q",
                "timm==1.0.28", "huggingface_hub>=0.34,<2",
                "safetensors>=0.5,<1", "Pillow>=10,<13", "tqdm>=4.66",
            ],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)],
            check=True,
        )
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        import numpy as np
        import torch
        from tqdm.auto import tqdm

        print("SOURCE COMMIT:", SOURCE_COMMIT)
        print("Torch:", torch.__version__)
        assert torch.cuda.is_available(), "Hãy chọn Colab Runtime > Change runtime type > GPU"
        print("GPU:", torch.cuda.get_device_name(0))
        """
    ),
    code(
        r"""
        #@title 2) Cấu hình khóa cho pilot/smoke
        DRIVE_ROOT = Path("/content/drive/MyDrive/nckh1m_data/sanpo_real_v0_joint_human_only_rgb3")
        LOCAL_EXTRACT_ROOT = Path("/content/sanpo_real_pilot_extracted_v1")
        LOCAL_RUNS_ROOT = Path("/content/replite_sanpo_smoke_runs")
        DRIVE_RUNS_ROOT = DRIVE_ROOT / "smoke_runs"

        SMOKE_EPOCHS = 3  #@param {type:"integer"}
        BACKBONE_NAME = "mobilenetv3_small_050"  #@param ["mobilenetv3_small_050", "mobilenetv4_conv_small"]
        PRETRAINED_IN1K = True  #@param {type:"boolean"}
        ALLOW_RANDOM_INIT_FALLBACK = False  #@param {type:"boolean"}
        IMAGE_HEIGHT = 288  #@param {type:"integer"}
        IMAGE_WIDTH = 512  #@param {type:"integer"}
        BATCH_SIZE = 2  #@param {type:"integer"}
        NUM_WORKERS = 2  #@param {type:"integer"}
        SEED = 42  #@param {type:"integer"}
        DEPTH_MIN_METRES = 0.1
        DEPTH_MAX_METRES = 80.0

        assert 2 <= SMOKE_EPOCHS <= 5, "Smoke phải nằm trong 2–5 epoch"
        assert IMAGE_HEIGHT % 32 == 0 and IMAGE_WIDTH % 32 == 0
        assert BATCH_SIZE > 0 and NUM_WORKERS >= 0
        assert DRIVE_ROOT.is_dir(), f"Không tìm thấy dữ liệu: {DRIVE_ROOT}"

        RUN_ID = datetime.now(timezone.utc).strftime("sanpo_smoke_%Y%m%dT%H%M%S_%fZ")
        LOCAL_RUN_DIR = LOCAL_RUNS_ROOT / RUN_ID
        DRIVE_RUN_DIR = DRIVE_RUNS_ROOT / RUN_ID
        LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=False)

        EXPECTED = (
            {
                "split": "train", "session_id": "eAXv0qwkO9zgffv9lQdSts_uRZCQI3Ro",
                "sensor": "camera_chest", "joint_frames": 107,
            },
            {
                "split": "train", "session_id": "a7bGB6aD6bcUMhl86R9HNNHUVUUxlS2c",
                "sensor": "camera_head", "joint_frames": 73,
            },
            {
                "split": "test", "session_id": "zKsJQMv6IV6seRnaYa_gp6fYkiKFYR3h",
                "sensor": "camera_head", "joint_frames": 78,
            },
        )
        assert len({item["session_id"] for item in EXPECTED}) == 3
        assert [item["split"] for item in EXPECTED].count("train") == 2
        assert [item["split"] for item in EXPECTED].count("test") == 1
        print("Run:", RUN_ID)
        """
    ),
    markdown(
        r"""
        ## Kiểm chứng và giải nén

        Cell kế tiếp không chọn archive bằng glob. Nó nối `current_download_selection.json`
        với ledger theo `(split, session_id, sensor, selection_sha256)`, sau đó kiểm SHA-256
        trong lúc copy từ Drive. Tar chỉ được chứa file/thư mục tương đối dưới `sanpo-real/`;
        symlink, hardlink, device và path traversal đều bị từ chối.
        """
    ),
    code(
        r"""
        #@title 3) Verify SHA-256 và safe-extract đúng 3 archive
        import hashlib

        from replite.data import (
            SANPO_DETECTION_CLASS_NAMES,
            load_sanpo_joint_manifest,
        )

        def read_json(path):
            return json.loads(Path(path).read_text(encoding="utf-8"))

        def sha256_file(path, chunk_size=16 * 1024 * 1024):
            digest = hashlib.sha256()
            with Path(path).open("rb") as handle:
                while True:
                    block = handle.read(chunk_size)
                    if not block:
                        break
                    digest.update(block)
            return digest.hexdigest()

        def inside(child, parent):
            try:
                Path(child).resolve().relative_to(Path(parent).resolve())
                return True
            except ValueError:
                return False

        def copy_and_hash(source, destination, expected_sha):
            source, destination = Path(source), Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                actual = sha256_file(destination)
                assert actual == expected_sha, f"Local cache sai SHA: {destination}"
                return actual
            partial = destination.with_name(destination.name + ".part")
            assert not partial.exists(), f"Có partial chưa xử lý: {partial}"
            digest = hashlib.sha256()
            with source.open("rb") as src, partial.open("xb") as dst, tqdm(
                total=source.stat().st_size, unit="B", unit_scale=True, desc=f"Copy {source.name[:28]}"
            ) as progress:
                while True:
                    block = src.read(16 * 1024 * 1024)
                    if not block:
                        break
                    dst.write(block)
                    digest.update(block)
                    progress.update(len(block))
                dst.flush()
                os.fsync(dst.fileno())
            actual = digest.hexdigest()
            assert actual == expected_sha, f"SHA mismatch: {source}"
            os.replace(partial, destination)
            return actual

        def validate_tar_members(archive):
            names = subprocess.check_output(
                ["tar", "--zstd", "-tf", str(archive)], text=True
            ).splitlines()
            assert names, f"Archive rỗng: {archive}"
            for raw in names:
                path = PurePosixPath(raw)
                assert not path.is_absolute(), f"Absolute tar member: {raw}"
                assert ".." not in path.parts, f"Tar traversal: {raw}"
                assert path.parts and path.parts[0] == "sanpo-real", f"Sai tar root: {raw}"
            verbose = subprocess.check_output(
                ["tar", "--zstd", "-tvf", str(archive)], text=True
            ).splitlines()
            for line in verbose:
                assert line and line[0] in {"-", "d"}, f"Tar có link/device bị từ chối: {line}"
            return len(names)

        def validate_extracted_manifest(manifest_path, expected, selection_sha):
            manifest, info = load_sanpo_joint_manifest(manifest_path)
            assert info.official_split == expected["split"]
            assert info.session_id == expected["session_id"]
            assert info.sensor == expected["sensor"]
            assert info.sample_count == expected["joint_frames"]
            assert manifest["selection_sha256"] == selection_sha
            session_root = info.session_root
            for sample in manifest["samples"]:
                for relative in (*sample["rgb_context_paths"], sample["panoptic_path"], sample["depth_path"], sample["detection_path"]):
                    path = session_root.joinpath(*PurePosixPath(relative).parts)
                    assert inside(path, session_root) and path.is_file(), path
            class_path = session_root.joinpath(*PurePosixPath(manifest["detection"]["class_manifest"]).parts)
            classes = read_json(class_path)
            assert classes["num_classes"] == 15
            ordered = sorted(classes["classes"], key=lambda item: item["detection_class_id"])
            assert [item["name"] for item in ordered] == list(SANPO_DETECTION_CLASS_NAMES)
            assert classes["detection_config_sha256"] == manifest["detection"]["config_sha256"]
            return manifest, info

        selection_path = DRIVE_ROOT / "metadata" / "current_download_selection.json"
        ledger_path = DRIVE_ROOT / "archive_manifest.json"
        selection = read_json(selection_path)
        ledger = read_json(ledger_path)
        assert selection.get("schema_version") == 1
        assert ledger.get("schema_version") == 2

        selected_by_key = {
            (item["split"], item["session_id"], item["sensor"]): item
            for item in selection["records"]
        }
        ledger_entries = list(ledger["archives"].values())
        BUNDLES = []
        for expected in EXPECTED:
            key = (expected["split"], expected["session_id"], expected["sensor"])
            assert key in selected_by_key, f"Selection không chứa {key}"
            selected = selected_by_key[key]
            assert selected["joint_frames"] == expected["joint_frames"]
            matches = [
                entry for entry in ledger_entries
                if (entry["split"], entry["session_id"], entry["sensor"]) == key
                and entry["selection_sha256"] == selected["selection_sha256"]
            ]
            assert len(matches) == 1, f"Ledger cần đúng 1 archive cho {key}, thấy {len(matches)}"
            entry = matches[0]
            archive = Path(entry["archive"])
            assert archive.is_file() and inside(archive, DRIVE_ROOT / "archives")
            assert archive.stat().st_size == entry["archive_bytes"]

            sha_sidecar = archive.with_name(archive.name + ".sha256")
            fields = sha_sidecar.read_text(encoding="utf-8").strip().split()
            assert fields == [entry["archive_sha256"], archive.name]
            sidecar = read_json(archive.with_name(archive.name + ".manifest.json"))
            assert sidecar["schema_version"] == 1
            for field in ("archive_sha256", "archive_bytes", "selection_sha256", "session_id", "sensor", "split"):
                assert sidecar["entry"][field] == entry[field]

            local_archive = Path("/content") / "sanpo_archive_cache" / archive.name
            copy_and_hash(archive, local_archive, entry["archive_sha256"])
            member_count = validate_tar_members(local_archive)

            target = LOCAL_EXTRACT_ROOT / (
                f"{entry['split']}__{entry['session_id']}__{entry['sensor']}__{entry['archive_sha256'][:12]}"
            )
            marker = target / ".extract_complete.json"
            if marker.is_file():
                marker_data = read_json(marker)
                assert marker_data["archive_sha256"] == entry["archive_sha256"]
                manifests = list(target.glob("sanpo-real/*/*/left/_sanpo_joint_manifest.json"))
                assert len(manifests) == 1
                manifest, info = validate_extracted_manifest(
                    manifests[0], expected, selected["selection_sha256"]
                )
            else:
                assert not target.exists(), f"Extract target chưa hoàn chỉnh; không tự xóa: {target}"
                stage = target.with_name(target.name + ".extracting")
                if stage.exists():
                    owned = stage / ".owned_extract.json"
                    assert owned.is_file() and read_json(owned).get("archive_sha256") == entry["archive_sha256"]
                    shutil.rmtree(stage)
                stage.mkdir(parents=True)
                (stage / ".owned_extract.json").write_text(
                    json.dumps({"archive_sha256": entry["archive_sha256"]}, sort_keys=True), encoding="utf-8"
                )
                subprocess.run(
                    [
                        "tar", "--zstd", "-xf", str(local_archive), "-C", str(stage),
                        "--no-same-owner", "--no-same-permissions",
                    ],
                    check=True,
                )
                manifests = list(stage.glob("sanpo-real/*/*/left/_sanpo_joint_manifest.json"))
                assert len(manifests) == 1
                manifest, info = validate_extracted_manifest(
                    manifests[0], expected, selected["selection_sha256"]
                )
                (stage / ".extract_complete.json").write_text(
                    json.dumps(
                        {
                            "archive_sha256": entry["archive_sha256"],
                            "archive_bytes": entry["archive_bytes"],
                            "member_count": member_count,
                            "source_archive": str(archive),
                        },
                        indent=2, sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                (stage / ".owned_extract.json").unlink()
                stage.rename(target)
                manifests = list(target.glob("sanpo-real/*/*/left/_sanpo_joint_manifest.json"))
            local_archive.unlink(missing_ok=True)
            BUNDLES.append({"expected": expected, "entry": entry, "manifest_path": manifests[0]})
            print(f"OK {key}: {expected['joint_frames']} samples | {entry['archive_sha256'][:16]}…")

        assert len(BUNDLES) == 3
        print("\nĐã verify + extract đủ 3 archive vào:", LOCAL_EXTRACT_ROOT)
        """
    ),
    markdown(
        r"""
        ## Trực quan QA

        Mỗi archive lấy frame đầu và frame giữa. Box dương có nhãn chữ; box nhỏ bị ignore
        vẽ nét đứt xám. Semantic dùng bảng màu categorical cố định. Depth dùng `cividis`,
        đơn vị mét, cùng miền hiển thị 0.1–80 m; pixel invalid có màu xám. Ảnh QA được lưu
        riêng, không sửa dữ liệu gốc.
        """
    ),
    code(
        r"""
        #@title 4) Vẽ 6 mẫu: RGB | box | semantic | depth
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.colors import ListedColormap, Normalize
        from PIL import Image

        from replite.data import SANPO_DETECTION_CLASS_NAMES, decode_sanpo_panoptic, read_sanpo_depth

        OKABE_ITO = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000")
        semantic_colors = np.vstack(
            ([0.45, 0.45, 0.45, 1.0], plt.get_cmap("tab20")(np.arange(20)), plt.get_cmap("tab20b")(np.arange(10)))
        )
        semantic_cmap = ListedColormap(semantic_colors[:31])
        depth_cmap = plt.get_cmap("cividis").copy()
        depth_cmap.set_bad("#8c8c8c")
        depth_norm = Normalize(vmin=DEPTH_MIN_METRES, vmax=DEPTH_MAX_METRES, clip=True)

        def path_from_sample(info, value):
            return info.session_root.joinpath(*PurePosixPath(value).parts)

        rows = []
        for bundle in BUNDLES:
            manifest, info = load_sanpo_joint_manifest(bundle["manifest_path"])
            for index in sorted({0, len(manifest["samples"]) // 2}):
                rows.append((manifest, info, index))

        fig, axes = plt.subplots(len(rows), 4, figsize=(20, 4.2 * len(rows)), constrained_layout=True)
        depth_artist = None
        preview_records = []
        for row_index, (manifest, info, sample_index) in enumerate(rows):
            sample = manifest["samples"][sample_index]
            rgb_path = path_from_sample(info, sample["rgb_context_paths"][-1])
            mask_path = path_from_sample(info, sample["panoptic_path"])
            detection_path = path_from_sample(info, sample["detection_path"])
            depth_path = path_from_sample(info, sample["depth_path"])
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
            mask = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.uint8)
            semantic, _ = decode_sanpo_panoptic(mask)
            detection = read_json(detection_path)
            depth = read_sanpo_depth(depth_path)
            depth_valid = np.isfinite(depth) & (depth > DEPTH_MIN_METRES) & (depth <= DEPTH_MAX_METRES)
            shown_depth = np.ma.masked_where(~depth_valid, depth)

            axes[row_index, 0].imshow(rgb)
            axes[row_index, 1].imshow(rgb)
            for box, label in zip(detection["boxes"], detection["labels"]):
                x1, y1, x2, y2 = map(float, box)
                color = OKABE_ITO[int(label) % len(OKABE_ITO)]
                axes[row_index, 1].add_patch(
                    patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.6)
                )
                axes[row_index, 1].text(
                    x1, max(0, y1 - 2), SANPO_DETECTION_CLASS_NAMES[int(label)], color="white", fontsize=7,
                    bbox={"facecolor": "black", "alpha": 0.72, "pad": 1, "edgecolor": "none"},
                )
            for box in detection.get("ignore_boxes", []):
                x1, y1, x2, y2 = map(float, box)
                axes[row_index, 1].add_patch(
                    patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#B3B3B3", linewidth=1.2, linestyle="--")
                )
            axes[row_index, 2].imshow(rgb)
            axes[row_index, 2].imshow(semantic, cmap=semantic_cmap, vmin=0, vmax=30, alpha=0.52, interpolation="nearest")
            depth_artist = axes[row_index, 3].imshow(shown_depth, cmap=depth_cmap, norm=depth_norm, interpolation="nearest")

            label = f"{info.official_split} | {info.session_id[:10]}… | {info.sensor} | frame {sample['target_frame']}"
            axes[row_index, 0].set_ylabel(label, fontsize=9)
            for axis in axes[row_index]:
                axis.set_xticks([]); axis.set_yticks([])
            preview_records.append({
                "split": info.official_split, "session_id": info.session_id, "sensor": info.sensor,
                "target_frame": sample["target_frame"], "positive_boxes": len(detection["boxes"]),
                "ignored_boxes": len(detection.get("ignore_boxes", [])),
                "valid_depth_fraction": float(depth_valid.mean()),
            })

        for axis, title in zip(axes[0], ("Target RGB", "Derived boxes", "Semantic overlay", "Metric depth")):
            axis.set_title(title, fontweight="bold")
        fig.colorbar(depth_artist, ax=axes[:, 3], label="Depth (m)", shrink=0.75)
        preview_path = LOCAL_RUN_DIR / "sanpo_data_preview.png"
        fig.savefig(preview_path, dpi=150, facecolor="white", bbox_inches="tight")
        plt.show()
        (LOCAL_RUN_DIR / "sanpo_data_preview.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "raw_data_modified": False,
                    "depth_display_metres": [DEPTH_MIN_METRES, DEPTH_MAX_METRES],
                    "depth_colormap": "cividis",
                    "invalid_depth_color": "gray",
                    "semantic_rendering": "source semantic IDs 0..30, categorical LUT, alpha=0.52",
                    "samples": preview_records,
                },
                indent=2, sort_keys=True,
            ), encoding="utf-8"
        )
        print("Preview:", preview_path)
        """
    ),
    code(
        r"""
        #@title 5) Dataset, session-disjoint loaders và model/loss
        import math, time
        from dataclasses import asdict
        from torch.utils.data import DataLoader

        from replite import RepLiteConfig, TaskConfig, create_replite_model
        from replite.data import (
            SANPO_DETECTION_CLASS_NAMES,
            SANPO_SEGMENTATION_CLASS_NAMES,
            SanpoJointDataset,
            sanpo_joint_collate,
        )
        from replite.training import (
            CheckpointManager,
            MultiTaskCriterion,
            Trainer,
            TrainerConfig,
            TrainingLogger,
            WarmupCosineScheduler,
            create_adamw,
            move_to_device,
        )

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        bundle_by_session = {
            item["expected"]["session_id"]: item for item in BUNDLES
        }
        train_bundle = bundle_by_session["eAXv0qwkO9zgffv9lQdSts_uRZCQI3Ro"]
        val_bundle = bundle_by_session["a7bGB6aD6bcUMhl86R9HNNHUVUUxlS2c"]
        test_bundle = bundle_by_session["zKsJQMv6IV6seRnaYa_gp6fYkiKFYR3h"]
        assert train_bundle["entry"]["split"] == val_bundle["entry"]["split"] == "train"
        assert test_bundle["entry"]["split"] == "test"
        assert len({item["expected"]["session_id"] for item in BUNDLES}) == 3

        dataset_kwargs = {
            "image_size": (IMAGE_HEIGHT, IMAGE_WIDTH),
            "depth_min": DEPTH_MIN_METRES,
            "depth_max": DEPTH_MAX_METRES,
            "normalize": True,
        }
        train_dataset = SanpoJointDataset(train_bundle["manifest_path"], **dataset_kwargs)
        val_dataset = SanpoJointDataset(val_bundle["manifest_path"], **dataset_kwargs)
        official_test_dataset = SanpoJointDataset(test_bundle["manifest_path"], **dataset_kwargs)
        assert (len(train_dataset), len(val_dataset), len(official_test_dataset)) == (107, 73, 78)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % (2**32)
            random.seed(worker_seed)
            np.random.seed(worker_seed)

        common_loader = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": True,
            "persistent_workers": False,
            "collate_fn": sanpo_joint_collate,
            "worker_init_fn": seed_worker,
        }
        if NUM_WORKERS > 0:
            common_loader["prefetch_factor"] = 2
        train_generator = torch.Generator().manual_seed(SEED)
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=train_generator, drop_last=False, **common_loader
        )
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_loader)
        official_test_loader = DataLoader(
            official_test_dataset, shuffle=False, drop_last=False, **common_loader
        )

        tasks = TaskConfig(
            detection_classes=len(SANPO_DETECTION_CLASS_NAMES),
            segmentation_classes=len(SANPO_SEGMENTATION_CLASS_NAMES),
            depth=True,
            gated_dense_fusion=True,
        )
        model_config = RepLiteConfig(
            tasks=tasks,
            backbone_name=BACKBONE_NAME,
            pretrained=PRETRAINED_IN1K,
            recurrence_steps=3,
            detection_reg_max=0,
        )
        pretrained_fallback = None
        try:
            model = create_replite_model(
                model_config,
                cache_dir=DRIVE_ROOT / "pretrained_cache",
            )
        except Exception as exc:
            if not (PRETRAINED_IN1K and ALLOW_RANDOM_INIT_FALLBACK):
                raise
            pretrained_fallback = f"{type(exc).__name__}: {exc}"
            model_config = RepLiteConfig(
                **{**model_config.as_dict(), "tasks": tasks, "pretrained": False}
            )
            model = create_replite_model(model_config)
            print("CẢNH BÁO: fallback random init được bật công khai:", pretrained_fallback)

        criterion = MultiTaskCriterion(
            detection_num_classes=len(SANPO_DETECTION_CLASS_NAMES),
            detection_reg_max=model_config.detection_reg_max,
            segmentation_ignore_index=255,
            depth_loss_type="log_l1_silog",
            depth_min=DEPTH_MIN_METRES,
            depth_max=DEPTH_MAX_METRES,
        )
        optimizer = create_adamw(
            model, lr=3e-4, weight_decay=1e-2, backbone_lr_multiplier=0.1
        )
        trainer_config = TrainerConfig(
            epochs=SMOKE_EPOCHS,
            amp=True,
            amp_dtype="float16",
            grad_clip_norm=1.0,
            log_every_n_steps=10,
            validate_every_n_epochs=1,
            checkpoint_every_n_epochs=1,
            monitor="val/total",
            monitor_mode="min",
        )
        total_steps = SMOKE_EPOCHS * math.ceil(len(train_loader) / trainer_config.grad_accum_steps)
        warmup_steps = min(10, total_steps // 10)
        scheduler = WarmupCosineScheduler(
            optimizer, total_steps=total_steps, warmup_steps=warmup_steps, min_lr_ratio=0.05
        )
        logger = TrainingLogger(LOCAL_RUN_DIR, run_id=RUN_ID, fsync=False)
        checkpoint_manager = CheckpointManager(LOCAL_RUN_DIR / "checkpoints")

        class NotebookTrainer(Trainer):
            def train_epoch(self, loader, *, epoch):
                started = time.perf_counter()
                result = super().train_epoch(
                    tqdm(loader, desc=f"train {epoch + 1}/{self.config.epochs}", leave=True), epoch=epoch
                )
                print(f"[train] epoch {epoch + 1}/{self.config.epochs} | total={result['total']:.6f} | {time.perf_counter()-started:.1f}s")
                return result

            def validate(self, loader, *, epoch=None):
                started = time.perf_counter()
                result = super().validate(
                    tqdm(loader, desc=f"val {0 if epoch is None else epoch + 1}/{self.config.epochs}", leave=True), epoch=epoch
                )
                print(f"[val] total={result['total']:.6f} | {time.perf_counter()-started:.1f}s")
                return result

        trainer = NotebookTrainer(
            model, criterion, optimizer, trainer_config,
            device="cuda", scheduler=scheduler, logger=logger,
            checkpoint_manager=checkpoint_manager, validation_metrics=None,
        )
        print("Train/val/test:", len(train_dataset), len(val_dataset), len(official_test_dataset))
        print("Backbone:", BACKBONE_NAME, "| pretrained:", model_config.pretrained)
        print("Steps:", total_steps, "| warmup:", warmup_steps)
        """
    ),
    code(
        r"""
        #@title 6) Preflight một batch: shape, loss, gradient contract
        sample_inputs, sample_targets = next(iter(train_loader))
        assert sample_inputs.shape[1:] == (3, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert sample_targets["segmentation"].dtype == torch.int64
        assert sample_targets["depth_valid"].dtype == torch.bool
        assert isinstance(sample_targets["detection"], list)
        sample_inputs = move_to_device(sample_inputs, trainer.device, non_blocking=True)
        sample_targets = move_to_device(sample_targets, trainer.device, non_blocking=True)
        trainer.model.train()
        with trainer._autocast():
            sample_outputs = trainer.model(sample_inputs)
            sample_losses = trainer.criterion(sample_outputs, sample_targets)
        assert torch.isfinite(sample_losses["total"]), sample_losses
        sample_losses["total"].backward()
        assert any(parameter.grad is not None for parameter in trainer.model.parameters() if parameter.requires_grad)
        trainer.optimizer.zero_grad(set_to_none=True)
        print("Preflight OK | input:", tuple(sample_inputs.shape))
        print({key: float(value.detach().float().cpu()) for key, value in sample_losses.items() if value.ndim == 0})
        del sample_inputs, sample_targets, sample_outputs, sample_losses
        torch.cuda.empty_cache()
        """
    ),
    code(
        r"""
        #@title 7) Smoke train 2–5 epoch (mặc định 3) + official-test forward-only
        torch.cuda.reset_peak_memory_stats()
        run_started = time.perf_counter()
        history = trainer.fit(train_loader, val_loader)
        elapsed_seconds = time.perf_counter() - run_started
        logger.close()

        assert len(history) == SMOKE_EPOCHS
        scalar_history = []
        for record in history:
            compact = {"epoch": int(record["epoch"])}
            for split in ("train", "val"):
                compact[split] = {
                    key: float(value) for key, value in record[split].items()
                    if isinstance(value, (int, float))
                }
                assert compact[split] and all(np.isfinite(list(compact[split].values())))
            scalar_history.append(compact)
        assert trainer.global_step > 0
        assert trainer.amp_skip_count == 0, f"AMP skips: {trainer.amp_skip_count}"
        for name in ("last.pt", "best.pt"):
            path = checkpoint_manager.directory / name
            assert path.is_file() and path.with_name(path.name + ".sha256").is_file(), path

        # Official-test không đi vào fit(), scheduler, early stop hay checkpoint selection.
        # Chỉ forward đúng một batch để xác nhận archive thứ ba dùng được.
        trainer.model.eval()
        test_inputs, _ = next(iter(official_test_loader))
        with torch.inference_mode(), trainer._autocast():
            test_outputs = trainer.model(test_inputs.to("cuda", non_blocking=True))
        assert test_outputs.segmentation.shape[-2:] == (IMAGE_HEIGHT, IMAGE_WIDTH)
        assert test_outputs.depth.shape[-2:] == (IMAGE_HEIGHT, IMAGE_WIDTH)
        del test_inputs, test_outputs

        peak_vram_gib = torch.cuda.max_memory_allocated() / 1024**3
        print(f"SMOKE PASS | {SMOKE_EPOCHS} epoch | {elapsed_seconds/60:.2f} min | peak VRAM {peak_vram_gib:.2f} GiB")
        print("Official-test: forward-only QA; KHÔNG dùng chọn model.")
        """
    ),
    code(
        r"""
        #@title 8) Ghi provenance, checksum và mirror run bundle lên Drive
        def jsonable(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {str(key): jsonable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [jsonable(item) for item in value]
            return value

        resolved = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "source_repository": REPO_URL,
            "source_commit": SOURCE_COMMIT,
            "seed": SEED,
            "model_config": model_config.as_dict(),
            "trainer_config": asdict(trainer_config),
            "optimizer": {"name": "AdamW", "lr": 3e-4, "weight_decay": 1e-2, "backbone_lr_multiplier": 0.1},
            "scheduler": {"name": "warmup_cosine", "total_steps": total_steps, "warmup_steps": warmup_steps, "min_lr_ratio": 0.05},
            "image_size_hw": [IMAGE_HEIGHT, IMAGE_WIDTH],
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "depth_range_metres": [DEPTH_MIN_METRES, DEPTH_MAX_METRES],
            "pretrained_random_init_fallback": pretrained_fallback,
            "split_protocol": {
                "train_session": train_dataset.info.session_id,
                "validation_session": val_dataset.info.session_id,
                "official_test_session": official_test_dataset.info.session_id,
                "official_test_role": "one-batch forward-only pipeline QA; never checkpoint selection",
            },
            "archives": [
                {
                    "split": item["entry"]["split"],
                    "session_id": item["entry"]["session_id"],
                    "sensor": item["entry"]["sensor"],
                    "archive_sha256": item["entry"]["archive_sha256"],
                    "selection_sha256": item["entry"]["selection_sha256"],
                }
                for item in BUNDLES
            ],
        }
        (LOCAL_RUN_DIR / "resolved_config.json").write_text(
            json.dumps(jsonable(resolved), indent=2, sort_keys=True), encoding="utf-8"
        )
        (LOCAL_RUN_DIR / "history.json").write_text(
            json.dumps(scalar_history, indent=2, sort_keys=True), encoding="utf-8"
        )
        (LOCAL_RUN_DIR / "run_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "smoke_pass",
                    "epochs": SMOKE_EPOCHS,
                    "global_step": trainer.global_step,
                    "amp_skip_count": trainer.amp_skip_count,
                    "elapsed_seconds": elapsed_seconds,
                    "peak_vram_gib": peak_vram_gib,
                    "best_metrics": trainer.best_metrics,
                },
                indent=2, sort_keys=True,
            ), encoding="utf-8"
        )
        (LOCAL_RUN_DIR / "source_commit.txt").write_text(SOURCE_COMMIT + "\n", encoding="utf-8")

        artifact_records = []
        for path in sorted(item for item in LOCAL_RUN_DIR.rglob("*") if item.is_file()):
            artifact_records.append({
                "path": path.relative_to(LOCAL_RUN_DIR).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        artifact_manifest = {
            "schema_version": 1,
            "run_id": RUN_ID,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifact_records,
        }
        (LOCAL_RUN_DIR / "artifact_manifest.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        DRIVE_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        assert not DRIVE_RUN_DIR.exists(), f"Không ghi đè run: {DRIVE_RUN_DIR}"
        uploading = DRIVE_RUN_DIR.with_name(DRIVE_RUN_DIR.name + ".uploading")
        assert not uploading.exists(), f"Có upload dở: {uploading}"
        shutil.copytree(LOCAL_RUN_DIR, uploading)
        for record in artifact_records:
            copied = uploading / record["path"]
            assert copied.stat().st_size == record["bytes"]
            assert sha256_file(copied) == record["sha256"]
        os.replace(uploading, DRIVE_RUN_DIR)

        print("HOÀN TẤT")
        print("Local run:", LOCAL_RUN_DIR)
        print("Drive run:", DRIVE_RUN_DIR)
        print("Preview:", DRIVE_RUN_DIR / "sanpo_data_preview.png")
        print("Best checkpoint:", DRIVE_RUN_DIR / "checkpoints" / "best.pt")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
