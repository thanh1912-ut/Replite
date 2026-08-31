#@title SANPO-Real: lọc RGB(t-2,t-1,t) + panoptic(t) + depth(t), rồi lưu Drive

import base64
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

# ============================================================
# CẤU HÌNH PILOT — CHẠY THẲNG 2 TRAIN + 1 TEST
# ============================================================

ACTION = "download"  #@param ["inventory", "download"]

# human_only: chỉ dùng mask do người gán nhãn.
# human_and_machine: dùng cả mask lan truyền; lớn hơn đáng kể.
ANNOTATION_POLICY = "human_only"  #@param ["human_only", "human_and_machine"]

MIN_JOINT_FRAMES = 20  #@param {type:"integer"}

# Có thể tải official test để đánh giá cuối cùng.
# Không dùng official test cho early stopping.
INCLUDE_OFFICIAL_TEST = True  #@param {type:"boolean"}

# Pilot khóa đúng hai official-train session và một official-test session.
# Mỗi session chỉ chọn một camera; không chọn hai camera cùng session.
MAX_TRAIN_SESSION_CAMERAS = 2  #@param {type:"integer"}
MAX_TEST_SESSION_CAMERAS = 1   #@param {type:"integer"}

# Box được sinh từ từng connected component của một panoptic label.
# 100 px là ngưỡng thống kê object của SANPO paper, không phải chuẩn
# detection chính thức; giá trị này được ghi vào manifest để tái lập.
DETECTION_MIN_COMPONENT_PIXELS = 100  #@param {type:"integer"}

# Bật nếu muốn quét lại bucket thay vì dùng inventory đã lưu.
REFRESH_INVENTORY = False  #@param {type:"boolean"}

# Khi archive đã tồn tại, True sẽ đọc lại toàn bộ để kiểm SHA-256.
VERIFY_EXISTING_ARCHIVES = False  #@param {type:"boolean"}

DISCOVERY_WORKERS = 24
INVENTORY_WORKERS = 10
DOWNLOAD_WORKERS = 8

assert ACTION in {"inventory", "download"}
assert ANNOTATION_POLICY in {"human_only", "human_and_machine"}
assert MIN_JOINT_FRAMES > 0
assert MAX_TRAIN_SESSION_CAMERAS >= 0
assert MAX_TEST_SESSION_CAMERAS >= 0
assert DETECTION_MIN_COMPONENT_PIXELS > 0

if ANNOTATION_POLICY == "human_only":
    ALLOWED_ANNOTATION_TYPES = {"HUMAN_ANNOTATED"}
else:
    ALLOWED_ANNOTATION_TYPES = {
        "HUMAN_ANNOTATED",
        "MACHINE_ANNOTATED",
    }

KNOWN_ANNOTATION_TYPES = {
    "HUMAN_ANNOTATED",
    "MACHINE_ANNOTATED",
}

# ============================================================
# CÀI THƯ VIỆN VÀ MOUNT DRIVE
# ============================================================

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "google-cloud-storage>=2.18",
        "google-crc32c",
        "numpy>=1.26,<3",
        "Pillow>=10,<13",
        "tqdm",
    ],
    check=True,
)

import google_crc32c
import numpy as np
from google.api_core.retry import Retry
from google.cloud import storage
from google.colab import drive
from PIL import Image
from tqdm.auto import tqdm

drive.mount("/content/drive")

# ============================================================
# ĐƯỜNG DẪN
# ============================================================

BUCKET_NAME = "gresearch"
SOURCE_ROOT = "sanpo_dataset/v0"
REAL_ROOT = f"{SOURCE_ROOT}/sanpo-real"
SENSORS = ("camera_head", "camera_chest")

DRIVE_ROOT = (
    Path("/content/drive/MyDrive/nckh1m_data")
    / f"sanpo_real_v0_joint_{ANNOTATION_POLICY}_rgb3"
)

DRIVE_META = DRIVE_ROOT / "metadata"
DRIVE_ARCHIVES = DRIVE_ROOT / "archives"

LOCAL_ROOT = Path("/content/sanpo_joint_stage")
LOCAL_DATA = LOCAL_ROOT / "data"
LOCAL_ARCHIVES = LOCAL_ROOT / "archives"

for folder in (
    DRIVE_ROOT,
    DRIVE_META,
    DRIVE_ARCHIVES,
    LOCAL_DATA,
    LOCAL_ARCHIVES,
):
    folder.mkdir(parents=True, exist_ok=True)

GCS_RETRY = Retry(
    initial=1.0,
    maximum=30.0,
    multiplier=2.0,
    deadline=600.0,
)

gcs_client = storage.Client.create_anonymous_client()
bucket = gcs_client.bucket(BUCKET_NAME)

# Tên file chuẩn theo SANPO chính thức.
RGB_RE = re.compile(r"^video_frames/(\d{6})\.png$")
MASK_RE = re.compile(r"^segmentation_masks/(\d{6})\.png$")
DEPTH_RE = re.compile(r"^depth_maps/(\d{6})\.float16\.gz$")

# ============================================================
# HÀM CHUNG
# ============================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def crc32c_file(path, chunk_size=16 * 1024 * 1024):
    checksum = google_crc32c.Checksum()

    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            checksum.update(chunk)

    return base64.b64encode(checksum.digest()).decode("ascii")


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_text(
        path,
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
    )


def blob_metadata(blob):
    return {
        "name": blob.name,
        "bytes": int(blob.size or 0),
        "generation": str(blob.generation),
        "crc32c": blob.crc32c,
        "md5_hash": blob.md5_hash,
        "updated": (
            blob.updated.isoformat()
            if blob.updated is not None
            else None
        ),
    }


def get_exact_blob(name):
    blob = bucket.get_blob(
        name,
        timeout=120,
        retry=GCS_RETRY,
    )

    if blob is None:
        raise FileNotFoundError(
            f"Không tìm thấy: gs://{BUCKET_NAME}/{name}"
        )

    return blob


def snapshot_blob(meta):
    return bucket.blob(
        meta["name"],
        generation=int(meta["generation"]),
    )


def local_matches_meta(path, meta):
    path = Path(path)

    if not path.is_file():
        return False

    if path.stat().st_size != int(meta["bytes"]):
        return False

    expected_crc = meta.get("crc32c")
    if expected_crc and crc32c_file(path) != expected_crc:
        return False

    return True


def download_snapshot(meta, destination):
    """Tải đúng object generation và xác minh size + CRC32C."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if local_matches_meta(destination, meta):
        return

    partial = destination.with_name(destination.name + ".part")

    if partial.exists():
        partial.unlink()

    blob = snapshot_blob(meta)
    blob.download_to_filename(
        str(partial),
        checksum="auto",
        timeout=300,
        retry=GCS_RETRY,
    )

    if not local_matches_meta(partial, meta):
        raise RuntimeError(
            f"Object tải về sai checksum/kích thước: {meta['name']}"
        )

    os.replace(partial, destination)


def safe_relative_object_path(object_name, session_prefix):
    if not object_name.startswith(session_prefix):
        raise RuntimeError(f"Object nằm ngoài session: {object_name}")

    relative = PurePosixPath(object_name[len(session_prefix):])

    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe GCS path: {object_name}")

    return relative


def human_bytes(value):
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)

    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024


# ============================================================
# TẢI METADATA VÀ OFFICIAL SPLIT
# ============================================================

official_files = {
    f"{SOURCE_ROOT}/labelmap.json": "labelmap.json",
    f"{SOURCE_ROOT}/labeltype.json": "labeltype.json",
    f"{REAL_ROOT}/splits/train_session_ids.txt": "train_session_ids.txt",
    f"{REAL_ROOT}/splits/test_session_ids.txt": "test_session_ids.txt",
}

official_manifest = {}

for remote_name, local_name in official_files.items():
    blob = get_exact_blob(remote_name)
    meta = blob_metadata(blob)
    local_path = DRIVE_META / local_name

    download_snapshot(meta, local_path)

    official_manifest[local_name] = {
        **meta,
        "sha256": sha256_file(local_path),
    }

atomic_json(
    DRIVE_META / "official_metadata_manifest.json",
    {
        "schema_version": 1,
        "source": f"gs://{BUCKET_NAME}/{SOURCE_ROOT}",
        "files": official_manifest,
    },
)

split_ids = {
    "train": [
        line.strip()
        for line in (
            DRIVE_META / "train_session_ids.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ],
    "test": [
        line.strip()
        for line in (
            DRIVE_META / "test_session_ids.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ],
}

if set(split_ids["train"]) & set(split_ids["test"]):
    raise RuntimeError("Official train/test session bị giao nhau")

split_hashes = {
    split: hashlib.sha256(
        ("\n".join(ids) + "\n").encode("utf-8")
    ).hexdigest()
    for split, ids in split_ids.items()
}

# Sinh mapping lớp detection dẫn xuất từ panoptic.
labelmap = json.loads(
    (DRIVE_META / "labelmap.json").read_text(encoding="utf-8")
)
labeltype = json.loads(
    (DRIVE_META / "labeltype.json").read_text(encoding="utf-8")
)

thing_classes = [
    {
        "source_semantic_id": int(labelmap[name]),
        "name": name,
    }
    for name, label_kind in labeltype.items()
    if label_kind == "panoptic"
]

thing_classes.sort(key=lambda item: item["source_semantic_id"])

expected_thing_source_ids = (
    5, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 28
)
actual_thing_source_ids = tuple(
    item["source_semantic_id"] for item in thing_classes
)

if actual_thing_source_ids != expected_thing_source_ids:
    raise RuntimeError(
        "Official SANPO thing taxonomy không đúng schema đã kiểm thử: "
        f"{actual_thing_source_ids}"
    )

for detection_id, item in enumerate(thing_classes):
    item["detection_class_id"] = detection_id

source_to_detection = {
    item["source_semantic_id"]: item["detection_class_id"]
    for item in thing_classes
}

detection_config = {
    "schema_version": 1,
    "bbox_format": "XYXY half-open, absolute target-RGB pixels",
    "component_connectivity": 8,
    "min_component_pixels": DETECTION_MIN_COMPONENT_PIXELS,
    "small_component_policy": "ignore_boxes",
    "instance_zero_policy": "keep_as_valid_panoptic_instance",
    "crowd_policy": "none; SANPO does not publish iscrowd",
}
detection_config_sha = canonical_sha256(detection_config)
detection_classes_path = DRIVE_META / "derived_detection_classes.json"

atomic_json(
    detection_classes_path,
    {
        "schema_version": 2,
        "num_classes": len(thing_classes),
        "classes": thing_classes,
        "taxonomy_sha256": {
            "labelmap.json": official_manifest["labelmap.json"]["sha256"],
            "labeltype.json": official_manifest["labeltype.json"]["sha256"],
        },
        "panoptic_encoding": {
            "semantic_id": "PNG RGB channel 0",
            "instance_id": "PNG RGB channel 1 * 256 + channel 2",
        },
        "detection_config": detection_config,
        "detection_config_sha256": detection_config_sha,
        "warning": (
            "Boxes được dẫn xuất từ official panoptic masks; đây không "
            "phải official SANPO detection benchmark. Ngưỡng component "
            "là lựa chọn preprocessing được version hóa."
        ),
    },
)


def _find_component_root(parent, node):
    root = node

    while parent[root] != root:
        root = parent[root]

    while parent[node] != node:
        next_node = parent[node]
        parent[node] = root
        node = next_node

    return root


def _union_components(parent, ranks, left, right):
    left = _find_component_root(parent, left)
    right = _find_component_root(parent, right)

    if left == right:
        return

    if ranks[left] < ranks[right]:
        left, right = right, left

    parent[right] = left

    if ranks[left] == ranks[right]:
        ranks[left] += 1


def _thing_runs(row):
    """Constant-key thing runs as half-open (x1, x2, key)."""

    changes = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), changes))
    ends = np.concatenate((changes, np.asarray([row.shape[0]])))
    runs = []

    for start, end in zip(starts.tolist(), ends.tolist()):
        key = int(row[start])

        if key >> 16 in source_to_detection:
            runs.append((int(start), int(end), key))

    return runs


def _extract_panoptic_components(mask_rgb):
    """8-connected components for each (semantic_id, instance_id)."""

    if mask_rgb.dtype != np.uint8 or mask_rgb.ndim != 3:
        raise ValueError("Panoptic mask phải là RGB uint8 HxWx3")

    if mask_rgb.shape[2] != 3:
        raise ValueError("Panoptic mask phải có đúng ba RGB channels")

    semantic = mask_rgb[:, :, 0].astype(np.uint32)
    instance = (
        mask_rgb[:, :, 1].astype(np.uint32) * np.uint32(256)
        + mask_rgb[:, :, 2].astype(np.uint32)
    )
    keys = (semantic << np.uint32(16)) | instance

    parent = []
    ranks = []
    run_x1 = []
    run_x2 = []
    run_y = []
    run_key = []
    previous = []

    for y in range(keys.shape[0]):
        current = []

        for x1, x2, key in _thing_runs(keys[y]):
            node = len(parent)
            parent.append(node)
            ranks.append(0)
            run_x1.append(x1)
            run_x2.append(x2)
            run_y.append(y)
            run_key.append(key)
            current.append((x1, x2, key, node))

        # Half-open runs touch diagonally when prev_x2 == x1 or
        # prev_x1 == x2; both count as 8-connected.
        first_possible = 0

        for x1, x2, key, node in current:
            while (
                first_possible < len(previous)
                and previous[first_possible][1] < x1
            ):
                first_possible += 1

            candidate = first_possible

            while (
                candidate < len(previous)
                and previous[candidate][0] <= x2
            ):
                prev_x1, prev_x2, prev_key, prev_node = previous[candidate]

                if (
                    prev_x2 >= x1
                    and prev_x1 <= x2
                    and prev_key == key
                ):
                    _union_components(parent, ranks, node, prev_node)

                candidate += 1

        previous = current

    aggregated = {}

    for node in range(len(parent)):
        root = _find_component_root(parent, node)
        area = run_x2[node] - run_x1[node]

        if root not in aggregated:
            aggregated[root] = {
                "key": run_key[node],
                "area": area,
                "box": [
                    run_x1[node],
                    run_y[node],
                    run_x2[node],
                    run_y[node] + 1,
                ],
            }
            continue

        component = aggregated[root]
        component["area"] += area
        box = component["box"]
        box[0] = min(box[0], run_x1[node])
        box[1] = min(box[1], run_y[node])
        box[2] = max(box[2], run_x2[node])
        box[3] = max(box[3], run_y[node] + 1)

    components = []

    for component in aggregated.values():
        key = component.pop("key")
        component["source_semantic_id"] = key >> 16
        component["instance_id"] = key & 0xFFFF
        component["detection_class_id"] = source_to_detection[
            component["source_semantic_id"]
        ]
        components.append(component)

    components.sort(
        key=lambda component: (
            component["source_semantic_id"],
            component["instance_id"],
            component["box"][1],
            component["box"][0],
            component["box"][3],
            component["box"][2],
        )
    )
    component_indices = Counter()

    for component in components:
        identity = (
            component["source_semantic_id"],
            component["instance_id"],
        )
        component["component_index"] = component_indices[identity]
        component_indices[identity] += 1

    return components, semantic


def derive_detection_target(mask_path, rgb_path, sample):
    """Create one versioned RepLite target from a SANPO mask."""

    mask_path = Path(mask_path)
    rgb_path = Path(rgb_path)
    mask_bytes = mask_path.read_bytes()

    with Image.open(io.BytesIO(mask_bytes)) as image:
        mask_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    with Image.open(rgb_path) as image:
        rgb_width, rgb_height = image.size

    mask_height, mask_width, _ = mask_rgb.shape

    if (rgb_height, rgb_width) != (mask_height, mask_width):
        raise RuntimeError(
            "Target RGB và panoptic mask khác shape; từ chối sinh box: "
            f"RGB={(rgb_height, rgb_width)}, "
            f"mask={(mask_height, mask_width)}, "
            f"frame={sample['target_frame']}"
        )

    components, semantic = _extract_panoptic_components(mask_rgb)
    positives = [
        component
        for component in components
        if component["area"] >= DETECTION_MIN_COMPONENT_PIXELS
    ]
    ignored_small = [
        component
        for component in components
        if component["area"] < DETECTION_MIN_COMPONENT_PIXELS
    ]

    return {
        "schema_version": 1,
        "dataset": "SANPO-Real-v0-derived-detection",
        "target_frame": sample["target_frame"],
        "annotation_type": sample["annotation_type"],
        "rgb_path": sample["rgb_context_paths"][-1],
        "panoptic_path": sample["panoptic_path"],
        "image_size_hw": [mask_height, mask_width],
        "valid_size": [mask_height, mask_width],
        "bbox_format": detection_config["bbox_format"],
        "boxes": [
            [float(coordinate) for coordinate in component["box"]]
            for component in positives
        ],
        "labels": [
            component["detection_class_id"] for component in positives
        ],
        "source_semantic_ids": [
            component["source_semantic_id"] for component in positives
        ],
        "instance_ids": [
            component["instance_id"] for component in positives
        ],
        "component_indices": [
            component["component_index"] for component in positives
        ],
        "mask_area_pixels": [
            component["area"] for component in positives
        ],
        "ignore_boxes": [
            [float(coordinate) for coordinate in component["box"]]
            for component in ignored_small
        ],
        "ignore_reasons": [
            "small_component" for _ in ignored_small
        ],
        "valid_negative": len(positives) == 0,
        "void_fraction": float(np.mean(semantic == 0)),
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "detection_config_sha256": detection_config_sha,
    }

# ============================================================
# DISCOVERY: TÌM SESSION-CAMERA CÓ PANOPTIC METADATA
# ============================================================

discovery_path = DRIVE_META / "panoptic_session_camera_discovery.json"

discovery_config = {
    "schema_version": 2,
    "source": f"gs://{BUCKET_NAME}/{REAL_ROOT}",
    "sensors": list(SENSORS),
    "split_hashes": split_hashes,
    "required_metadata": "frame_segmentation_annotation_type.json",
}

use_cached_discovery = False

if discovery_path.exists() and not REFRESH_INVENTORY:
    cached = json.loads(discovery_path.read_text(encoding="utf-8"))

    if cached.get("config") == discovery_config:
        annotated_records = cached["records"]
        use_cached_discovery = True
        print(
            "Dùng discovery cache:",
            len(annotated_records),
            "session-camera",
        )

if not use_cached_discovery:

    def discover_one(item):
        split, session_id, sensor = item

        annotation_name = (
            f"{REAL_ROOT}/{session_id}/{sensor}/left/"
            "frame_segmentation_annotation_type.json"
        )

        blob = bucket.get_blob(
            annotation_name,
            timeout=120,
            retry=GCS_RETRY,
        )

        if blob is None:
            return None

        return {
            "split": split,
            "session_id": session_id,
            "sensor": sensor,
            "annotation_object": blob_metadata(blob),
        }

    discovery_tasks = [
        (split, session_id, sensor)
        for split, ids in split_ids.items()
        for session_id in ids
        for sensor in SENSORS
    ]

    annotated_records = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=DISCOVERY_WORKERS
    ) as executor:
        futures = [
            executor.submit(discover_one, task)
            for task in discovery_tasks
        ]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Tìm panoptic session-camera",
        ):
            result = future.result()
            if result is not None:
                annotated_records.append(result)

    annotated_records.sort(
        key=lambda record: (
            record["split"],
            record["session_id"],
            record["sensor"],
        )
    )

    atomic_json(
        discovery_path,
        {
            "schema_version": 2,
            "config": discovery_config,
            "created_utc": utc_now(),
            "records": annotated_records,
        },
    )

    print(
        "Phát hiện:",
        len(annotated_records),
        "session-camera có panoptic metadata",
    )

if len(annotated_records) != 237:
    print(
        "CẢNH BÁO: paper công bố 237 annotated session-camera; "
        f"bucket hiện phát hiện {len(annotated_records)}."
    )

# ============================================================
# INVENTORY: GIAO RGB3 ∩ PANOPTIC ∩ DENSE DEPTH
# ============================================================

inventory_path = DRIVE_META / "joint_inventory.json"
inventory_csv_path = DRIVE_META / "joint_inventory.csv"

inventory_config = {
    "schema_version": 3,
    "source": f"gs://{BUCKET_NAME}/{REAL_ROOT}",
    "split_hashes": split_hashes,
    "allowed_annotation_types": sorted(ALLOWED_ANNOTATION_TYPES),
    "context_offsets": [-2, -1, 0],
    "target_modalities": [
        "left_rgb",
        "panoptic",
        "dense_CREStereo_depth",
        "derived_detection",
    ],
    "min_joint_frames": MIN_JOINT_FRAMES,
}

inventory_config_sha = canonical_sha256(inventory_config)
use_cached_inventory = False

if inventory_path.exists() and not REFRESH_INVENTORY:
    cached_inventory = json.loads(
        inventory_path.read_text(encoding="utf-8")
    )

    if (
        cached_inventory.get("inventory_config_sha256")
        == inventory_config_sha
    ):
        inventory = cached_inventory
        use_cached_inventory = True
        print(
            "Dùng inventory cache:",
            len(inventory["records"]),
            "session-camera",
        )

if not use_cached_inventory:

    def inventory_one(record):
        split = record["split"]
        session_id = record["session_id"]
        sensor = record["sensor"]

        session_prefix = f"{REAL_ROOT}/{session_id}/"
        left_prefix = f"{session_prefix}{sensor}/left/"

        rgb_by_id = {}
        mask_by_id = {}
        depth_by_id = {}
        annotation_meta = None

        def add_frame(index, frame_id, meta, modality):
            if frame_id in index:
                raise RuntimeError(
                    f"Duplicate {modality} frame {frame_id}: "
                    f"{session_id}/{sensor}"
                )
            index[frame_id] = meta

        for blob in bucket.list_blobs(
            prefix=left_prefix,
            timeout=120,
            retry=GCS_RETRY,
        ):
            if blob.name.endswith("_$folder$"):
                continue

            relative = blob.name[len(left_prefix):]
            meta = blob_metadata(blob)

            match = RGB_RE.fullmatch(relative)
            if match:
                add_frame(
                    rgb_by_id,
                    int(match.group(1)),
                    meta,
                    "RGB",
                )
                continue

            match = MASK_RE.fullmatch(relative)
            if match:
                add_frame(
                    mask_by_id,
                    int(match.group(1)),
                    meta,
                    "mask",
                )
                continue

            match = DEPTH_RE.fullmatch(relative)
            if match:
                add_frame(
                    depth_by_id,
                    int(match.group(1)),
                    meta,
                    "depth",
                )
                continue

            if relative == "frame_segmentation_annotation_type.json":
                annotation_meta = meta

        if annotation_meta is None:
            raise RuntimeError(
                f"Thiếu annotation metadata: {session_id}/{sensor}"
            )

        annotation_blob = snapshot_blob(annotation_meta)
        annotation_raw = annotation_blob.download_as_bytes(
            checksum="auto",
            timeout=300,
            retry=GCS_RETRY,
        )

        annotation_json = json.loads(
            annotation_raw.decode("utf-8")
        )

        if not isinstance(annotation_json, dict):
            raise RuntimeError(
                f"Annotation JSON không phải object: {session_id}/{sensor}"
            )

        annotation_type = {}

        for frame_key, provenance in annotation_json.items():
            frame_id = int(frame_key)

            if frame_id in annotation_type:
                raise RuntimeError(
                    f"Duplicate annotation ID {frame_id}"
                )

            if provenance not in KNOWN_ANNOTATION_TYPES:
                raise RuntimeError(
                    f"Annotation type lạ {provenance!r}: "
                    f"{session_id}/{sensor}/{frame_key}"
                )

            annotation_type[frame_id] = provenance

        annotation_ids = set(annotation_type)
        rgb_ids = set(rgb_by_id)
        mask_ids = set(mask_by_id)
        depth_ids = set(depth_by_id)

        candidate_ids = {
            frame_id
            for frame_id, provenance in annotation_type.items()
            if provenance in ALLOWED_ANNOTATION_TYPES
        }

        # t-2,t-1,t là ID số học thật trong cùng session-camera.
        joint_ids = sorted(
            t
            for t in candidate_ids
            if (
                t >= 2
                and t in rgb_ids
                and t - 1 in rgb_ids
                and t - 2 in rgb_ids
                and t in mask_ids
                and t in depth_ids
            )
        )

        required_rgb_ids = sorted({
            frame_id
            for target in joint_ids
            for frame_id in (target - 2, target - 1, target)
        })

        selected_objects = {}

        for frame_id in required_rgb_ids:
            meta = rgb_by_id[frame_id]
            selected_objects[meta["name"]] = meta

        for frame_id in joint_ids:
            for index in (mask_by_id, depth_by_id):
                meta = index[frame_id]
                selected_objects[meta["name"]] = meta

        # Giữ nguyên official provenance JSON.
        selected_objects[annotation_meta["name"]] = annotation_meta

        # Giữ session description/camera metadata nếu tồn tại.
        description_name = f"{session_prefix}description.json"
        description_blob = bucket.get_blob(
            description_name,
            timeout=120,
            retry=GCS_RETRY,
        )

        if description_blob is not None:
            description_meta = blob_metadata(description_blob)
            selected_objects[description_meta["name"]] = description_meta

        selected_objects = [
            selected_objects[name]
            for name in sorted(selected_objects)
        ]

        selected_bytes = sum(
            int(meta["bytes"])
            for meta in selected_objects
        )

        samples = []

        for target in joint_ids:
            samples.append({
                "target_frame": target,
                "annotation_type": annotation_type[target],
                "rgb_context_frames": [
                    target - 2,
                    target - 1,
                    target,
                ],
                "rgb_context_paths": [
                    (
                        f"{sensor}/left/video_frames/"
                        f"{frame_id:06d}.png"
                    )
                    for frame_id in (
                        target - 2,
                        target - 1,
                        target,
                    )
                ],
                "panoptic_path": (
                    f"{sensor}/left/segmentation_masks/"
                    f"{target:06d}.png"
                ),
                "depth_path": (
                    f"{sensor}/left/depth_maps/"
                    f"{target:06d}.float16.gz"
                ),
                "detection_target": (
                    "derive boxes from panoptic semantic+instance IDs"
                ),
            })

        provenance_counts = dict(
            sorted(Counter(annotation_type.values()).items())
        )

        source_selection = {
            "split": split,
            "session_id": session_id,
            "sensor": sensor,
            "allowed_annotation_types": sorted(
                ALLOWED_ANNOTATION_TYPES
            ),
            "samples": samples,
            "selected_objects": selected_objects,
        }

        selection_sha = canonical_sha256(source_selection)

        return {
            "split": split,
            "session_id": session_id,
            "sensor": sensor,
            "rgb_frames_available": len(rgb_ids),
            "mask_frames_available": len(mask_ids),
            "depth_frames_available": len(depth_ids),
            "annotation_frames_available": len(annotation_ids),
            "annotation_counts": provenance_counts,
            "annotation_without_mask": sorted(
                annotation_ids - mask_ids
            ),
            "mask_without_annotation": sorted(
                mask_ids - annotation_ids
            ),
            "joint_frames": len(joint_ids),
            "unique_rgb_context_frames": len(required_rgb_ids),
            "selected_object_count": len(selected_objects),
            "selected_bytes": selected_bytes,
            "qualifies": len(joint_ids) >= MIN_JOINT_FRAMES,
            "selection_sha256": selection_sha,
            "samples": samples,
            "selected_objects": selected_objects,
        }

    inventory_records = []
    inventory_errors = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=INVENTORY_WORKERS
    ) as executor:
        future_to_record = {
            executor.submit(inventory_one, record): record
            for record in annotated_records
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_record),
            total=len(future_to_record),
            desc="Quét RGB3 ∩ panoptic ∩ depth",
        ):
            source_record = future_to_record[future]

            try:
                inventory_records.append(future.result())
            except Exception as exc:
                inventory_errors.append({
                    "record": source_record,
                    "error": repr(exc),
                })

    if inventory_errors:
        error_path = DRIVE_META / "joint_inventory_errors.json"

        atomic_json(
            error_path,
            {
                "created_utc": utc_now(),
                "errors": inventory_errors,
            },
        )

        raise RuntimeError(
            f"Có {len(inventory_errors)} lỗi inventory. "
            f"Xem: {error_path}"
        )

    inventory_records.sort(
        key=lambda record: (
            record["split"],
            record["session_id"],
            record["sensor"],
        )
    )

    inventory = {
        "schema_version": 3,
        "dataset": "SANPO-Real-v0-joint",
        "created_utc": utc_now(),
        "inventory_config": inventory_config,
        "inventory_config_sha256": inventory_config_sha,
        "records": inventory_records,
    }

    atomic_json(inventory_path, inventory)

# ============================================================
# XUẤT BẢNG INVENTORY
# ============================================================

csv_buffer = io.StringIO()
csv_writer = csv.DictWriter(
    csv_buffer,
    fieldnames=[
        "split",
        "session_id",
        "sensor",
        "human_annotations",
        "machine_annotations",
        "rgb_frames_available",
        "mask_frames_available",
        "depth_frames_available",
        "joint_frames",
        "unique_rgb_context_frames",
        "selected_object_count",
        "selected_bytes",
        "selected_gib",
        "qualifies",
        "selection_sha256",
    ],
)
csv_writer.writeheader()

for record in inventory["records"]:
    csv_writer.writerow({
        "split": record["split"],
        "session_id": record["session_id"],
        "sensor": record["sensor"],
        "human_annotations": record[
            "annotation_counts"
        ].get("HUMAN_ANNOTATED", 0),
        "machine_annotations": record[
            "annotation_counts"
        ].get("MACHINE_ANNOTATED", 0),
        "rgb_frames_available": record[
            "rgb_frames_available"
        ],
        "mask_frames_available": record[
            "mask_frames_available"
        ],
        "depth_frames_available": record[
            "depth_frames_available"
        ],
        "joint_frames": record["joint_frames"],
        "unique_rgb_context_frames": record[
            "unique_rgb_context_frames"
        ],
        "selected_object_count": record[
            "selected_object_count"
        ],
        "selected_bytes": record["selected_bytes"],
        "selected_gib": (
            f"{record['selected_bytes'] / 1024**3:.6f}"
        ),
        "qualifies": record["qualifies"],
        "selection_sha256": record["selection_sha256"],
    })

atomic_text(inventory_csv_path, csv_buffer.getvalue())

qualified_records = [
    record
    for record in inventory["records"]
    if record["qualifies"]
]

print("\n========== INVENTORY SUMMARY ==========")

for split in ("train", "test"):
    rows = [
        record
        for record in qualified_records
        if record["split"] == split
    ]

    print(
        f"{split:5s}: "
        f"{len(rows):3d} session-camera | "
        f"{sum(r['joint_frames'] for r in rows):,} targets | "
        f"{human_bytes(sum(r['selected_bytes'] for r in rows))}"
    )

print("\n15 session-camera đầu tiên đạt điều kiện:")

for record in qualified_records[:15]:
    print(
        f"{record['split']:5s} | "
        f"{record['session_id'][:14]:14s} | "
        f"{record['sensor']:12s} | "
        f"joint={record['joint_frames']:4d} | "
        f"RGBctx={record['unique_rgb_context_frames']:4d} | "
        f"{human_bytes(record['selected_bytes'])}"
    )

# ============================================================
# CHỌN SUBSET DOWNLOAD
# ============================================================

def seeded_order(record):
    identity = (
        f"seed42:{record['split']}:"
        f"{record['session_id']}:{record['sensor']}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_split(split, limit):
    """Chọn deterministically, tối đa một camera cho mỗi session_id."""
    rows = sorted(
        [
            record
            for record in qualified_records
            if record["split"] == split
        ],
        key=seeded_order,
    )

    selected = []
    seen_session_ids = set()

    for record in rows:
        if record["session_id"] in seen_session_ids:
            continue

        seen_session_ids.add(record["session_id"])
        selected.append(record)

        if limit > 0 and len(selected) == limit:
            break

    return selected


selected_records = select_split(
    "train",
    MAX_TRAIN_SESSION_CAMERAS,
)

if INCLUDE_OFFICIAL_TEST:
    selected_records += select_split(
        "test",
        MAX_TEST_SESSION_CAMERAS,
    )

if not selected_records:
    raise RuntimeError(
        "Không có session-camera nào đạt MIN_JOINT_FRAMES"
    )

selected_split_counts = Counter(
    record["split"] for record in selected_records
)
selected_session_ids = [
    record["session_id"] for record in selected_records
]

if selected_split_counts != {"train": 2, "test": 1}:
    raise RuntimeError(
        "Pilot phải có đúng 2 session train + 1 session test, "
        f"nhưng nhận {dict(selected_split_counts)}"
    )

if len(set(selected_session_ids)) != 3:
    raise RuntimeError("Pilot phải gồm ba session_id khác nhau")

selected_bytes = sum(
    record["selected_bytes"]
    for record in selected_records
)
selected_targets = sum(
    record["joint_frames"]
    for record in selected_records
)

selection_manifest_path = (
    DRIVE_META / "current_download_selection.json"
)

atomic_json(
    selection_manifest_path,
    {
        "schema_version": 1,
        "created_utc": utc_now(),
        "inventory_config_sha256": inventory_config_sha,
        "annotation_policy": ANNOTATION_POLICY,
        "min_joint_frames": MIN_JOINT_FRAMES,
        "include_official_test": INCLUDE_OFFICIAL_TEST,
        "max_train_session_cameras": MAX_TRAIN_SESSION_CAMERAS,
        "max_test_session_cameras": MAX_TEST_SESSION_CAMERAS,
        "session_camera_count": len(selected_records),
        "unique_session_count": len(set(selected_session_ids)),
        "joint_target_count": selected_targets,
        "exact_selected_source_bytes": selected_bytes,
        "records": [
            {
                "split": record["split"],
                "session_id": record["session_id"],
                "sensor": record["sensor"],
                "joint_frames": record["joint_frames"],
                "selected_bytes": record["selected_bytes"],
                "selection_sha256": record["selection_sha256"],
            }
            for record in selected_records
        ],
        "protocol_note": (
            "Official test split is reserved for final evaluation. "
            "Validation must be carved from official train by session."
        ),
    },
)

print("\n========== DOWNLOAD SELECTION ==========")
print("Session-camera:", len(selected_records))
print("Joint target:", f"{selected_targets:,}")
print("Exact source bytes:", human_bytes(selected_bytes))
print(
    "Archive thực tế thường gần mức trên vì PNG và depth.gz "
    "đã được nén sẵn."
)
print("Inventory JSON:", inventory_path)
print("Inventory CSV :", inventory_csv_path)
print("Selection     :", selection_manifest_path)

# ============================================================
# DOWNLOAD + PACKAGE
# ============================================================

def ensure_zstd():
    if shutil.which("zstd") is not None:
        return

    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "zstd"],
        check=True,
    )


def copy_with_progress(source, destination):
    source = Path(source)
    destination = Path(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)

    total = source.stat().st_size
    chunk_size = 16 * 1024 * 1024

    with (
        open(source, "rb") as src,
        open(destination, "wb") as dst,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Drive {source.name[:28]}",
        ) as progress,
    ):
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break

            dst.write(chunk)
            progress.update(len(chunk))

        dst.flush()
        os.fsync(dst.fileno())


def safe_remove_stage(path, marker):
    path = Path(path).resolve()
    local_root = LOCAL_DATA.resolve()

    path.relative_to(local_root)

    if not Path(marker).is_file():
        raise RuntimeError(
            f"Từ chối xóa stage không có marker: {path}"
        )

    shutil.rmtree(path)


def run_download(records):
    ensure_zstd()

    ledger_path = DRIVE_ROOT / "archive_manifest.json"

    if ledger_path.exists():
        ledger = json.loads(
            ledger_path.read_text(encoding="utf-8")
        )
    else:
        ledger = {
            "schema_version": 2,
            "dataset": (
                f"SANPO-Real-v0-joint-{ANNOTATION_POLICY}"
            ),
            "source": f"gs://{BUCKET_NAME}/{SOURCE_ROOT}",
            "archives": {},
        }

    for index, record in enumerate(records, start=1):
        split = record["split"]
        session_id = record["session_id"]
        sensor = record["sensor"]
        selection_sha = record["selection_sha256"]
        package_sha = canonical_sha256({
            "selection_sha256": selection_sha,
            "detection_config_sha256": detection_config_sha,
        })

        archive_name = (
            f"{session_id}__{sensor}__"
            f"{package_sha[:12]}.tar.zst"
        )

        final_archive = (
            DRIVE_ARCHIVES / split / archive_name
        )
        final_archive.parent.mkdir(parents=True, exist_ok=True)

        archive_manifest_path = final_archive.with_name(
            final_archive.name + ".manifest.json"
        )
        sha_path = final_archive.with_name(
            final_archive.name + ".sha256"
        )

        ledger_key = (
            f"{split}/{session_id}/{sensor}/{package_sha}"
        )
        existing_entry = ledger["archives"].get(ledger_key)

        if final_archive.exists():
            if existing_entry is not None:
                if (
                    existing_entry["selection_sha256"]
                    != selection_sha
                ):
                    raise RuntimeError(
                        f"Selection SHA mismatch: {final_archive}"
                    )

                if (
                    existing_entry.get("detection_config_sha256")
                    != detection_config_sha
                ):
                    raise RuntimeError(
                        f"Detection config mismatch: {final_archive}"
                    )

                if (
                    final_archive.stat().st_size
                    != int(existing_entry["archive_bytes"])
                ):
                    raise RuntimeError(
                        f"Archive Drive sai kích thước: "
                        f"{final_archive}"
                    )

                if VERIFY_EXISTING_ARCHIVES:
                    actual_sha = sha256_file(final_archive)

                    if (
                        actual_sha
                        != existing_entry["archive_sha256"]
                    ):
                        raise RuntimeError(
                            f"Archive Drive sai SHA-256: "
                            f"{final_archive}"
                        )

                print(
                    f"[{index}/{len(records)}] Đã có, bỏ qua: "
                    f"{archive_name}"
                )
                continue

            if archive_manifest_path.exists():
                sidecar = json.loads(
                    archive_manifest_path.read_text(
                        encoding="utf-8"
                    )
                )
                recovered_entry = sidecar["entry"]

                if (
                    recovered_entry["selection_sha256"]
                    != selection_sha
                ):
                    raise RuntimeError(
                        f"Sidecar không khớp: {final_archive}"
                    )

                if (
                    recovered_entry.get("detection_config_sha256")
                    != detection_config_sha
                ):
                    raise RuntimeError(
                        f"Sidecar detection config không khớp: "
                        f"{final_archive}"
                    )

                actual_sha = sha256_file(final_archive)

                if (
                    actual_sha
                    != recovered_entry["archive_sha256"]
                ):
                    raise RuntimeError(
                        f"Archive không khớp sidecar: "
                        f"{final_archive}"
                    )

                ledger["archives"][ledger_key] = recovered_entry
                ledger["updated_utc"] = utc_now()
                atomic_json(ledger_path, ledger)

                print(
                    f"[{index}/{len(records)}] "
                    f"Đã phục hồi ledger: {archive_name}"
                )
                continue

            raise FileExistsError(
                "Archive tồn tại nhưng thiếu ledger/sidecar; "
                f"không ghi đè: {final_archive}"
            )

        print(
            f"\n[{index}/{len(records)}] "
            f"{split} | {session_id} | {sensor}\n"
            f"joint={record['joint_frames']:,} | "
            f"source={human_bytes(record['selected_bytes'])}"
        )

        required_local = (
            int(record["selected_bytes"] * 2.2)
            + 2 * 1024**3
        )
        free_local = shutil.disk_usage("/content").free

        if free_local < required_local:
            raise RuntimeError(
                "SSD Colab không đủ cho source + archive. "
                f"Cần khoảng {human_bytes(required_local)}, "
                f"còn {human_bytes(free_local)}."
            )

        bundle_root = (
            LOCAL_DATA
            / f"{session_id}__{sensor}__{selection_sha[:12]}"
        )
        local_session = (
            bundle_root / "sanpo-real" / session_id
        )
        marker = bundle_root / ".sanpo_joint_owned.json"

        bundle_root.mkdir(parents=True, exist_ok=True)

        if marker.exists():
            marker_data = json.loads(
                marker.read_text(encoding="utf-8")
            )

            if (
                marker_data.get("selection_sha256")
                != selection_sha
            ):
                raise RuntimeError(
                    f"Stage local thuộc selection khác: "
                    f"{bundle_root}"
                )
        else:
            atomic_json(
                marker,
                {
                    "selection_sha256": selection_sha,
                    "session_id": session_id,
                    "sensor": sensor,
                },
            )

        session_prefix = f"{REAL_ROOT}/{session_id}/"

        def download_one(meta):
            relative = safe_relative_object_path(
                meta["name"],
                session_prefix,
            )
            destination = local_session.joinpath(
                *relative.parts
            )
            download_snapshot(meta, destination)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=DOWNLOAD_WORKERS
        ) as executor:
            futures = [
                executor.submit(download_one, meta)
                for meta in record["selected_objects"]
            ]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"Download {session_id[:10]}",
            ):
                future.result()

        # Sinh detection box TRƯỚC KHI đóng archive. Mỗi sample có một
        # JSON riêng và một JSONL tổng hợp để DataLoader đọc tuần tự.
        left_root = local_session / sensor / "left"
        detection_dir = left_root / "detection_boxes"
        detection_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            detection_classes_path,
            left_root / "derived_detection_classes.json",
        )

        enriched_samples = []
        detection_records = []
        positive_box_count = 0
        ignored_box_count = 0

        for sample in record["samples"]:
            target_frame = int(sample["target_frame"])
            mask_path = local_session / sample["panoptic_path"]
            target_rgb_path = (
                local_session / sample["rgb_context_paths"][-1]
            )
            detection_target = derive_detection_target(
                mask_path,
                target_rgb_path,
                sample,
            )
            detection_relative = (
                f"{sensor}/left/detection_boxes/"
                f"{target_frame:06d}.json"
            )
            atomic_json(
                local_session / detection_relative,
                detection_target,
            )

            enriched_sample = dict(sample)
            enriched_sample["detection_path"] = detection_relative
            enriched_samples.append(enriched_sample)
            detection_records.append(detection_target)
            positive_box_count += len(detection_target["boxes"])
            ignored_box_count += len(detection_target["ignore_boxes"])

        atomic_text(
            left_root / "derived_detection.jsonl",
            "".join(
                json.dumps(
                    target,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for target in detection_records
            ),
        )
        atomic_json(
            left_root / "derived_detection_manifest.json",
            {
                "schema_version": 1,
                "session_id": session_id,
                "sensor": sensor,
                "official_split": split,
                "selection_sha256": selection_sha,
                "detection_config": detection_config,
                "detection_config_sha256": detection_config_sha,
                "target_count": len(detection_records),
                "positive_box_count": positive_box_count,
                "ignored_small_component_box_count": ignored_box_count,
                "jsonl": f"{sensor}/left/derived_detection.jsonl",
                "class_manifest": (
                    f"{sensor}/left/derived_detection_classes.json"
                ),
                "warning": (
                    "Derived labels, not an official SANPO detection task."
                ),
            },
        )

        # Manifest này là đầu vào cho DataLoader RepLite.
        # Không dùng loader SANPO official để scan subset sparse.
        joint_manifest = {
            "schema_version": 2,
            "dataset": "SANPO-Real-v0-joint",
            "official_split": split,
            "session_id": session_id,
            "sensor": sensor,
            "lens": "left",
            "selection_sha256": selection_sha,
            "inventory_config_sha256": inventory_config_sha,
            "annotation_policy": ANNOTATION_POLICY,
            "joint_frames": record["joint_frames"],
            "samples": enriched_samples,
            "source_objects": record["selected_objects"],
            "loader_warning": (
                "Frame IDs are sparse and are not renumbered. "
                "DataLoader must read this manifest explicitly; "
                "do not infer frames using range(len(files))."
            ),
            "depth_encoding": (
                "gzip little-endian float16; first two values are "
                "height,width, followed by H*W depth values in meters"
            ),
            "panoptic_encoding": {
                "semantic_id": "PNG RGB channel 0",
                "instance_id": "PNG RGB channel 1 * 256 + channel 2",
            },
            "shape_note": (
                "RGB/mask and dense depth may have different native "
                "resolutions. Align them later in preprocessing."
            ),
            "detection": {
                "derived": True,
                "config": detection_config,
                "config_sha256": detection_config_sha,
                "class_manifest": (
                    f"{sensor}/left/derived_detection_classes.json"
                ),
                "jsonl": f"{sensor}/left/derived_detection.jsonl",
                "manifest": (
                    f"{sensor}/left/derived_detection_manifest.json"
                ),
                "target_count": len(detection_records),
                "positive_box_count": positive_box_count,
                "ignored_small_component_box_count": ignored_box_count,
                "negative_frame_policy": (
                    "present panoptic mask with zero positive boxes is a "
                    "valid detection negative"
                ),
                "benchmark_warning": (
                    "not an official SANPO detection benchmark"
                ),
            },
            "test_protocol": (
                "Official test must not be used for early stopping."
            ),
        }

        joint_manifest_path = (
            local_session
            / sensor
            / "left"
            / "_sanpo_joint_manifest.json"
        )
        atomic_json(joint_manifest_path, joint_manifest)

        # Xác minh lại mọi official object trước khi đóng gói.
        for meta in record["selected_objects"]:
            relative = safe_relative_object_path(
                meta["name"],
                session_prefix,
            )
            local_path = local_session.joinpath(
                *relative.parts
            )

            if not local_matches_meta(local_path, meta):
                raise RuntimeError(
                    f"Local object sai checksum: {meta['name']}"
                )

        local_archive = LOCAL_ARCHIVES / archive_name

        if local_archive.exists():
            local_archive.unlink()

        subprocess.run(
            [
                "tar",
                "--sort=name",
                "--mtime=@0",
                "--owner=0",
                "--group=0",
                "--numeric-owner",
                "-I",
                "zstd -1 -T0",
                "-cf",
                str(local_archive),
                "-C",
                str(bundle_root),
                f"sanpo-real/{session_id}",
            ],
            check=True,
        )

        # Xác nhận archive đọc được.
        subprocess.run(
            [
                "tar",
                "-I",
                "zstd",
                "-tf",
                str(local_archive),
            ],
            stdout=subprocess.DEVNULL,
            check=True,
        )

        archive_sha = sha256_file(local_archive)
        archive_bytes = local_archive.stat().st_size

        uploading = final_archive.with_name(
            final_archive.name + ".uploading"
        )

        if uploading.exists():
            uploading.unlink()

        copy_with_progress(local_archive, uploading)

        drive_sha = sha256_file(uploading)

        if drive_sha != archive_sha:
            raise RuntimeError(
                f"SHA-256 sau upload không khớp: {archive_name}"
            )

        os.replace(uploading, final_archive)

        entry = {
            "split": split,
            "session_id": session_id,
            "sensor": sensor,
            "annotation_policy": ANNOTATION_POLICY,
            "selection_sha256": selection_sha,
            "package_sha256": package_sha,
            "detection_config_sha256": detection_config_sha,
            "archive": str(final_archive),
            "archive_bytes": archive_bytes,
            "archive_sha256": archive_sha,
            "joint_frames": record["joint_frames"],
            "unique_rgb_context_frames": record[
                "unique_rgb_context_frames"
            ],
            "source_object_count": record[
                "selected_object_count"
            ],
            "source_bytes": record["selected_bytes"],
            "detection_target_count": len(detection_records),
            "detection_positive_box_count": positive_box_count,
            "detection_ignored_box_count": ignored_box_count,
            "verified_utc": utc_now(),
        }

        atomic_json(
            archive_manifest_path,
            {
                "schema_version": 1,
                "entry": entry,
            },
        )
        atomic_text(
            sha_path,
            f"{archive_sha}  {archive_name}\n",
        )

        ledger["archives"][ledger_key] = entry
        ledger["updated_utc"] = utc_now()
        atomic_json(ledger_path, ledger)

        # Chỉ dọn local sau khi archive Drive + checksum + ledger an toàn.
        safe_remove_stage(bundle_root, marker)
        local_archive.resolve().relative_to(
            LOCAL_ARCHIVES.resolve()
        )
        local_archive.unlink()

        print(
            f"OK: {final_archive}\n"
            f"SHA-256: {archive_sha}"
        )

    print("\nHOÀN TẤT DOWNLOAD")
    print("Archives:", DRIVE_ARCHIVES)
    print("Ledger  :", ledger_path)


if ACTION == "inventory":
    print(
        "\nChưa tải dữ liệu vì ACTION='inventory'.\n"
        "Kiểm tra bảng phía trên và joint_inventory.csv.\n"
        "Nếu dung lượng phù hợp, đổi ACTION='download' rồi chạy lại cell."
    )
else:
    run_download(selected_records)
