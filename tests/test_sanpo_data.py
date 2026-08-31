"""Tests for SANPO panoptic decoding and derived detection targets."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from replite.data import (
    SANPO_DETECTION_CLASS_NAMES,
    SANPO_SOURCE_TO_DETECTION_LABEL,
    SANPO_THING_SOURCE_IDS,
    decode_sanpo_panoptic,
    extract_sanpo_components,
    sanpo_panoptic_to_detection,
)
from replite.multitask.heads import DetectionOutput
from replite.training import DetectionCriterion


def _mask(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _paint(
    mask: np.ndarray,
    ys: slice,
    xs: slice,
    *,
    semantic_id: int,
    instance_id: int,
) -> None:
    mask[ys, xs, 0] = semantic_id
    mask[ys, xs, 1] = instance_id // 256
    mask[ys, xs, 2] = instance_id % 256


def test_official_thing_mapping_is_stable_and_contiguous() -> None:
    assert SANPO_THING_SOURCE_IDS == (
        5,
        10,
        11,
        12,
        13,
        14,
        15,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        28,
    )
    assert SANPO_DETECTION_CLASS_NAMES == (
        "crosswalk",
        "opening-door",
        "opening-gate",
        "pedestrian",
        "rider",
        "animal",
        "stairs",
        "obstacle",
        "vehicle",
        "traffic sign",
        "traffic light",
        "pole",
        "bus stop",
        "bike rack",
        "tree",
    )
    assert list(SANPO_SOURCE_TO_DETECTION_LABEL.values()) == list(range(15))


def test_rgb_channel_decode_preserves_full_uint16_instance_id() -> None:
    mask = _mask(1, 2)
    _paint(mask, slice(None), slice(0, 1), semantic_id=12, instance_id=0)
    _paint(mask, slice(None), slice(1, 2), semantic_id=21, instance_id=64260)

    semantic, instance = decode_sanpo_panoptic(mask)

    assert semantic.tolist() == [[12, 21]]
    assert instance.dtype == np.uint32
    assert instance.tolist() == [[0, 64260]]


def test_instance_zero_is_kept_and_disconnected_regions_are_split() -> None:
    mask = _mask(8, 10)
    _paint(mask, slice(1, 3), slice(1, 4), semantic_id=28, instance_id=0)
    _paint(mask, slice(5, 7), slice(7, 9), semantic_id=28, instance_id=0)

    components = extract_sanpo_components(mask)

    assert [(item.instance_id, item.area, item.box) for item in components] == [
        (0, 6, (1, 1, 4, 3)),
        (0, 4, (7, 5, 9, 7)),
    ]


def test_components_use_eight_connectivity_and_half_open_boxes() -> None:
    mask = _mask(3, 3)
    _paint(mask, slice(0, 1), slice(0, 1), semantic_id=12, instance_id=17)
    _paint(mask, slice(1, 2), slice(1, 2), semantic_id=12, instance_id=17)

    components = extract_sanpo_components(mask)

    assert len(components) == 1
    assert components[0].area == 2
    assert components[0].box == (0, 0, 2, 2)


def test_min_area_boundary_small_ignore_and_stuff_exclusion() -> None:
    mask = _mask(20, 25)
    _paint(mask, slice(0, 10), slice(0, 10), semantic_id=21, instance_id=1)
    _paint(mask, slice(11, 20), slice(0, 11), semantic_id=20, instance_id=2)
    _paint(mask, slice(0, 20), slice(12, 25), semantic_id=1, instance_id=9)

    target = sanpo_panoptic_to_detection(mask, min_area=100)

    np.testing.assert_array_equal(
        target["boxes"], np.asarray([[0, 0, 10, 10]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        target["labels"],
        np.asarray([SANPO_SOURCE_TO_DETECTION_LABEL[21]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        target["ignore_boxes"],
        np.asarray([[0, 11, 11, 20]], dtype=np.float32),
    )
    assert target["valid_size"] == (20, 25)


def test_empty_thing_mask_is_a_valid_negative() -> None:
    mask = _mask(5, 7)
    mask[..., 0] = 3

    target = sanpo_panoptic_to_detection(mask)

    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["ignore_boxes"].shape == (0, 4)
    assert target["valid_size"] == (5, 7)


def test_invalid_mask_contract_is_rejected() -> None:
    with pytest.raises(TypeError, match="uint8"):
        decode_sanpo_panoptic(np.zeros((2, 2, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="H,W,3"):
        decode_sanpo_panoptic(np.zeros((2, 2), dtype=np.uint8))

    unknown = _mask(2, 2)
    unknown[..., 0] = 255
    with pytest.raises(ValueError, match="unknown SANPO semantic IDs"):
        decode_sanpo_panoptic(unknown)


def test_run_length_components_match_naive_eight_connected_reference() -> None:
    rng = np.random.default_rng(42)
    choices = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 9],  # stuff: excluded
            [12, 0, 0],
            [21, 250, 4],  # instance 64004
            [28, 0, 0],
        ],
        dtype=np.uint8,
    )
    mask = choices[rng.integers(0, len(choices), size=(11, 13))]
    semantic, instance = decode_sanpo_panoptic(mask)
    visited = np.zeros(semantic.shape, dtype=bool)
    expected: list[tuple[int, int, int, tuple[int, int, int, int]]] = []

    for y in range(semantic.shape[0]):
        for x in range(semantic.shape[1]):
            source_id = int(semantic[y, x])
            if visited[y, x] or source_id not in SANPO_SOURCE_TO_DETECTION_LABEL:
                continue
            instance_id = int(instance[y, x])
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if (
                            (dy == 0 and dx == 0)
                            or ny < 0
                            or nx < 0
                            or ny >= semantic.shape[0]
                            or nx >= semantic.shape[1]
                            or visited[ny, nx]
                            or int(semantic[ny, nx]) != source_id
                            or int(instance[ny, nx]) != instance_id
                        ):
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            ys = [pixel[0] for pixel in pixels]
            xs = [pixel[1] for pixel in pixels]
            expected.append(
                (
                    source_id,
                    instance_id,
                    len(pixels),
                    (min(xs), min(ys), max(xs) + 1, max(ys) + 1),
                )
            )

    expected.sort(key=lambda item: (item[0], item[1], item[3][1], item[3][0]))
    actual = [
        (
            item.source_semantic_id,
            item.instance_id,
            item.area,
            item.box,
        )
        for item in extract_sanpo_components(mask)
    ]

    assert actual == expected


def test_derived_numpy_target_is_accepted_by_detection_criterion() -> None:
    mask = _mask(64, 64)
    _paint(mask, slice(8, 40), slice(8, 40), semantic_id=21, instance_id=3)
    target = sanpo_panoptic_to_detection(mask, min_area=100)

    predictions = DetectionOutput(
        cls_logits=tuple(
            torch.zeros(1, 15, size, size, requires_grad=True)
            for size in (8, 4, 2)
        ),
        box_regression=tuple(
            torch.ones(1, 4, size, size, requires_grad=True)
            for size in (8, 4, 2)
        ),
        quality=tuple(
            torch.zeros(1, 1, size, size, requires_grad=True)
            for size in (8, 4, 2)
        ),
    )

    losses = DetectionCriterion(15)(predictions, (target,), image_size=(64, 64))

    assert torch.isfinite(losses.total)
    assert losses.num_targets.item() == 1
    losses.total.backward()
