"""SANPO panoptic-mask decoding and detection-target conversion.

SANPO stores a panoptic label as an RGB ``uint8`` PNG.  The red channel is
the source semantic ID and the remaining channels encode a 16-bit instance
ID as ``green * 256 + blue``.  Detection boxes in this module follow the
``replite.training`` contract: absolute, half-open XYXY coordinates and
contiguous zero-based labels.

The official SANPO masks do not provide a COCO-style ``iscrowd`` flag.  In
particular, instance ID zero is used by valid panoptic classes and must not be
interpreted as void or crowd.  Each equal ``(semantic_id, instance_id)``
region is therefore split into 8-connected components.  Components smaller
than ``min_area`` are emitted as ``ignore_boxes`` instead of positive ground
truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from types import MappingProxyType

import numpy as np
from PIL import Image


# These values are transcribed from the official SANPO labelmap.json and
# labeltype.json.  Source ID zero is the dataset's unlabeled semantic class.
SANPO_LABELMAP: Mapping[str, int] = MappingProxyType(
    {
        "unlabeled": 0,
        "road": 1,
        "curb": 2,
        "sidewalk": 3,
        "guard rail/road barrier": 4,
        "crosswalk": 5,
        "paved trail": 6,
        "building": 7,
        "wall/fence": 8,
        "hand rail": 9,
        "opening-door": 10,
        "opening-gate": 11,
        "pedestrian": 12,
        "rider": 13,
        "animal": 14,
        "stairs": 15,
        "water body": 16,
        "other walkable surface": 17,
        "inaccessible surface": 18,
        "railway track": 19,
        "obstacle": 20,
        "vehicle": 21,
        "traffic sign": 22,
        "traffic light": 23,
        "pole": 24,
        "bus stop": 25,
        "bike rack": 26,
        "sky": 27,
        "tree": 28,
        "vegetation": 29,
        "terrain": 30,
    }
)

SANPO_LABEL_TYPES: Mapping[str, str] = MappingProxyType(
    {
        "unlabeled": "semantic",
        "road": "semantic",
        "curb": "semantic",
        "sidewalk": "semantic",
        "guard rail/road barrier": "semantic",
        "crosswalk": "panoptic",
        "paved trail": "semantic",
        "building": "semantic",
        "wall/fence": "semantic",
        "hand rail": "semantic",
        "opening-door": "panoptic",
        "opening-gate": "panoptic",
        "pedestrian": "panoptic",
        "rider": "panoptic",
        "animal": "panoptic",
        "stairs": "panoptic",
        "water body": "semantic",
        "other walkable surface": "semantic",
        "inaccessible surface": "semantic",
        "railway track": "semantic",
        "obstacle": "panoptic",
        "vehicle": "panoptic",
        "traffic sign": "panoptic",
        "traffic light": "panoptic",
        "pole": "panoptic",
        "bus stop": "panoptic",
        "bike rack": "panoptic",
        "sky": "semantic",
        "tree": "panoptic",
        "vegetation": "semantic",
        "terrain": "semantic",
    }
)

_SOURCE_NAME_BY_ID = {source_id: name for name, source_id in SANPO_LABELMAP.items()}
SANPO_THING_SOURCE_IDS = tuple(
    sorted(
        SANPO_LABELMAP[name]
        for name, label_type in SANPO_LABEL_TYPES.items()
        if label_type == "panoptic"
    )
)
SANPO_DETECTION_CLASS_NAMES = tuple(
    _SOURCE_NAME_BY_ID[source_id] for source_id in SANPO_THING_SOURCE_IDS
)
SANPO_SOURCE_TO_DETECTION_LABEL: Mapping[int, int] = MappingProxyType(
    {
        source_id: detection_id
        for detection_id, source_id in enumerate(SANPO_THING_SOURCE_IDS)
    }
)

_KNOWN_SOURCE_IDS = np.asarray(sorted(_SOURCE_NAME_BY_ID), dtype=np.uint8)


@dataclass(frozen=True)
class SanpoComponent:
    """One 8-connected component of a SANPO thing-class panoptic label."""

    source_semantic_id: int
    instance_id: int
    area: int
    box: tuple[int, int, int, int]

    @property
    def detection_label(self) -> int:
        """Return this component's contiguous zero-based detection label."""

        return SANPO_SOURCE_TO_DETECTION_LABEL[self.source_semantic_id]


def _as_rgb_array(panoptic: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(panoptic, Image.Image) and panoptic.mode != "RGB":
        raise ValueError("SANPO panoptic images must use RGB mode")
    array = np.asarray(panoptic)
    if array.dtype != np.uint8:
        raise TypeError("SANPO panoptic masks must use uint8 dtype")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("SANPO panoptic masks must have shape H,W,3")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("SANPO panoptic masks must have positive height and width")
    return array


def decode_sanpo_panoptic(
    panoptic: Image.Image | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode an RGB SANPO mask into semantic and 16-bit instance arrays.

    Channel order is intentionally strict.  Arrays read through OpenCV must be
    converted from BGR to RGB before calling this function.
    """

    array = _as_rgb_array(panoptic)
    semantic = array[..., 0].copy()
    unknown = np.setdiff1d(np.unique(semantic), _KNOWN_SOURCE_IDS)
    if unknown.size:
        values = ", ".join(str(int(value)) for value in unknown)
        raise ValueError(f"unknown SANPO semantic IDs: {values}")

    # Cast before multiplying so values in the green channel cannot overflow
    # uint8. uint32 keeps subsequent key construction straightforward.
    instance = (
        array[..., 1].astype(np.uint32) * np.uint32(256)
        + array[..., 2].astype(np.uint32)
    )
    return semantic, instance


def _find(parent: list[int], node: int) -> int:
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != node:
        next_node = parent[node]
        parent[node] = root
        node = next_node
    return root


def _union(parent: list[int], ranks: list[int], left: int, right: int) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    if ranks[left_root] < ranks[right_root]:
        left_root, right_root = right_root, left_root
    parent[right_root] = left_root
    if ranks[left_root] == ranks[right_root]:
        ranks[left_root] += 1


def _thing_runs(row: np.ndarray) -> list[tuple[int, int, int]]:
    """Return constant-panoptic-key thing runs as ``(x1, x2, key)``."""

    width = int(row.shape[0])
    changes = np.flatnonzero(row[1:] != row[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), changes))
    ends = np.concatenate((changes, np.asarray([width])))
    result: list[tuple[int, int, int]] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        key = int(row[start])
        if key >> 16 in SANPO_SOURCE_TO_DETECTION_LABEL:
            result.append((int(start), int(end), key))
    return result


def _extract_components(
    semantic: np.ndarray,
    instance: np.ndarray,
) -> tuple[SanpoComponent, ...]:
    # Packing into uint32 is collision-free because semantic IDs fit in the
    # upper 16 bits and SANPO instance IDs occupy exactly the lower 16 bits.
    keys = (semantic.astype(np.uint32) << np.uint32(16)) | instance

    parent: list[int] = []
    ranks: list[int] = []
    run_x1: list[int] = []
    run_x2: list[int] = []
    run_y: list[int] = []
    run_key: list[int] = []
    previous: list[tuple[int, int, int, int]] = []

    for y in range(keys.shape[0]):
        current: list[tuple[int, int, int, int]] = []
        for x1, x2, key in _thing_runs(keys[y]):
            node = len(parent)
            parent.append(node)
            ranks.append(0)
            run_x1.append(x1)
            run_x2.append(x2)
            run_y.append(y)
            run_key.append(key)
            current.append((x1, x2, key, node))

        # For 8-connectivity, half-open runs on adjacent rows connect when
        # their horizontal intervals overlap or touch at one boundary.
        first_possible = 0
        for x1, x2, key, node in current:
            while (
                first_possible < len(previous)
                and previous[first_possible][1] < x1
            ):
                first_possible += 1
            candidate = first_possible
            while candidate < len(previous) and previous[candidate][0] <= x2:
                prev_x1, prev_x2, prev_key, prev_node = previous[candidate]
                if prev_x2 >= x1 and prev_x1 <= x2 and prev_key == key:
                    _union(parent, ranks, node, prev_node)
                candidate += 1
        previous = current

    aggregated: dict[int, list[int]] = {}
    for node in range(len(parent)):
        root = _find(parent, node)
        area = run_x2[node] - run_x1[node]
        if root not in aggregated:
            aggregated[root] = [
                run_key[node],
                area,
                run_x1[node],
                run_y[node],
                run_x2[node],
                run_y[node] + 1,
            ]
            continue
        component = aggregated[root]
        component[1] += area
        component[2] = min(component[2], run_x1[node])
        component[3] = min(component[3], run_y[node])
        component[4] = max(component[4], run_x2[node])
        component[5] = max(component[5], run_y[node] + 1)

    components = []
    for key, area, x1, y1, x2, y2 in aggregated.values():
        components.append(
            SanpoComponent(
                source_semantic_id=key >> 16,
                instance_id=key & 0xFFFF,
                area=area,
                box=(x1, y1, x2, y2),
            )
        )
    components.sort(
        key=lambda item: (
            item.source_semantic_id,
            item.instance_id,
            item.box[1],
            item.box[0],
            item.box[3],
            item.box[2],
        )
    )
    return tuple(components)


def extract_sanpo_components(
    panoptic: Image.Image | np.ndarray,
) -> tuple[SanpoComponent, ...]:
    """Extract every 8-connected component belonging to a thing class."""

    semantic, instance = decode_sanpo_panoptic(panoptic)
    return _extract_components(semantic, instance)


def _checked_min_area(min_area: int) -> int:
    if isinstance(min_area, bool) or not isinstance(min_area, Integral):
        raise TypeError("min_area must be an integer")
    if min_area < 0:
        raise ValueError("min_area must be non-negative")
    return int(min_area)


def _box_array(boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def sanpo_panoptic_to_detection(
    panoptic: Image.Image | np.ndarray,
    *,
    min_area: int = 100,
) -> dict[str, np.ndarray | tuple[int, int]]:
    """Convert one SANPO panoptic mask to a RepLite detection target.

    Args:
        panoptic: RGB ``uint8`` SANPO mask, either a Pillow image or NumPy
            array with shape ``H,W,3``.
        min_area: Minimum visible component area in source-mask pixels. A
            component with area exactly equal to this threshold is positive;
            smaller thing components become rectangular ``ignore_boxes``.

    Returns:
        A mapping accepted directly by :class:`replite.training.DetectionCriterion`.
        Boxes are float32 absolute half-open XYXY, labels are int64 and
        ``valid_size`` is ``(height, width)``.
    """

    min_area = _checked_min_area(min_area)
    semantic, instance = decode_sanpo_panoptic(panoptic)
    components = _extract_components(semantic, instance)

    boxes: list[tuple[int, int, int, int]] = []
    labels: list[int] = []
    ignore_boxes: list[tuple[int, int, int, int]] = []
    for component in components:
        if component.area >= min_area:
            boxes.append(component.box)
            labels.append(component.detection_label)
        else:
            ignore_boxes.append(component.box)

    return {
        "boxes": _box_array(boxes),
        "labels": np.asarray(labels, dtype=np.int64),
        "valid_size": (int(semantic.shape[0]), int(semantic.shape[1])),
        "ignore_boxes": _box_array(ignore_boxes),
    }


def load_sanpo_detection(
    filename: str | PathLike[str],
    *,
    min_area: int = 100,
) -> dict[str, np.ndarray | tuple[int, int]]:
    """Read an RGB SANPO PNG and convert it to a detection target."""

    with Image.open(filename) as image:
        image.load()
        return sanpo_panoptic_to_detection(image, min_area=min_area)


__all__ = [
    "SANPO_DETECTION_CLASS_NAMES",
    "SANPO_LABELMAP",
    "SANPO_LABEL_TYPES",
    "SANPO_SOURCE_TO_DETECTION_LABEL",
    "SANPO_THING_SOURCE_IDS",
    "SanpoComponent",
    "decode_sanpo_panoptic",
    "extract_sanpo_components",
    "load_sanpo_detection",
    "sanpo_panoptic_to_detection",
]
