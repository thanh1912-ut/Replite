"""Tests for the strict NYUv2 segmentation+depth data contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from replite.data.nyuv2 import (
    Nyuv2Augmentation,
    Nyuv2Dataset,
    discover_nyuv2,
    nyuv2_collate,
    read_nyuv2_depth,
    scan_nyuv2_label_ids,
)


def _write_sample(root: Path, stem: str, *, unknown_label: bool = False) -> None:
    height, width = 4, 6
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :3, 0] = 255
    rgb[:, 3:, 2] = 255
    segmentation = np.ones((height, width), dtype=np.uint8)
    segmentation[:, 3:] = 2
    segmentation[0, 0] = 0
    if unknown_label:
        segmentation[-1, -1] = 7
    depth = np.full((height, width), 1000, dtype=np.uint16)
    depth[:, 3:] = 2000
    depth[0, 0] = 0
    depth[-1, -1] = 11000
    for directory in ("images", "segmentation", "depth"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(root / "images" / f"image_{stem}.png")
    Image.fromarray(segmentation).save(
        root / "segmentation" / f"label_{stem}.png"
    )
    Image.fromarray(depth).save(root / "depth" / f"depth_{stem}.png")


def _build_dataset(tmp_path: Path, *, unknown_label: bool = False) -> Path:
    root = tmp_path / "outer" / "NYUDv2"
    _write_sample(root, "0001")
    _write_sample(root, "0002", unknown_label=unknown_label)
    _write_sample(root, "0003")
    (root / "gt_sets").mkdir(parents=True)
    (root / "gt_sets" / "train.txt").write_text(
        "images/image_0001.png\n0002\n", encoding="utf-8"
    )
    (root / "gt_sets" / "val.txt").write_text(
        "image_0003.png\n", encoding="utf-8"
    )
    # These are present in the supplied archive but intentionally not active.
    (root / "normals").mkdir()
    (root / "edge").mkdir()
    return root


def _dataset(root: Path, **overrides: object) -> Nyuv2Dataset:
    options: dict[str, object] = {
        "split": "train",
        "num_classes": 2,
        "label_mapping": {1: 0, 2: 1},
        "source_ignore_labels": (0,),
        "depth_unit_scale": 0.001,
        "image_size": (4, 6),
        "ignore_index": 255,
        "depth_min": 0.1,
        "depth_max": 10.0,
        "normalize": False,
    }
    options.update(overrides)
    return Nyuv2Dataset(root, **options)  # type: ignore[arg-type]


def test_discovery_matches_modalities_and_keeps_official_test_separate(
    tmp_path: Path,
) -> None:
    root = _build_dataset(tmp_path)
    index = discover_nyuv2(tmp_path)

    assert index.root == root.resolve()
    assert [sample.key for sample in index.train] == ["1", "2"]
    assert [sample.key for sample in index.test] == ["3"]
    assert index.heldout_source_name == "val"
    assert index.sample_count == 3
    assert not (
        {sample.key for sample in index.train}
        & {sample.key for sample in index.test}
    )
    assert scan_nyuv2_label_ids(index) == (0, 1, 2)


def test_discovery_rejects_missing_modality_and_split_overlap(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    (root / "depth" / "depth_0002.png").unlink()
    with pytest.raises(ValueError, match="not one-to-one matched"):
        discover_nyuv2(root)

    root = _build_dataset(tmp_path / "second")
    (root / "gt_sets" / "val.txt").write_text(
        "0002\n0003\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="train/test overlap"):
        discover_nyuv2(root)


def test_discovery_rejects_uncovered_sample_and_duplicate_stem(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    (root / "gt_sets" / "train.txt").write_text("0001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not cover every matched sample"):
        discover_nyuv2(root)

    root = _build_dataset(tmp_path / "duplicate")
    Image.open(root / "images" / "image_0001.png").save(
        root / "images" / "img_0001.jpg"
    )
    with pytest.raises(ValueError, match="duplicate canonical image"):
        discover_nyuv2(root)


def test_discovery_rejects_ambiguous_val_and_test_files(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    (root / "gt_sets" / "test.txt").write_text("0003\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one held-out"):
        discover_nyuv2(root)


def test_discovery_aligns_unique_stems_across_different_subdirectories(
    tmp_path: Path,
) -> None:
    root = _build_dataset(tmp_path)
    nested = root / "images" / "rgb"
    nested.mkdir()
    for path in tuple((root / "images").glob("*.png")):
        path.replace(nested / path.name)

    index = discover_nyuv2(root)

    assert len(index.train) == 2
    assert len(index.test) == 1


def test_static_dataset_contract_maps_labels_and_metric_depth(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    dataset = _dataset(root)

    inputs, targets = dataset[0]

    assert inputs.shape == (3, 4, 6)
    assert inputs.dtype == torch.float32
    assert targets["segmentation"].shape == (4, 6)
    assert targets["segmentation"].dtype == torch.int64
    assert targets["segmentation"][0, 0].item() == 255
    assert not targets["segmentation_valid"][0, 0]
    assert set(targets["segmentation"][1:].unique().tolist()) == {0, 1}
    assert targets["depth"].shape == (1, 4, 6)
    assert targets["depth"].dtype == torch.float32
    assert targets["depth"][0, 1, 1].item() == pytest.approx(1.0)
    assert targets["depth"][0, 1, 4].item() == pytest.approx(2.0)
    assert not targets["depth_valid"][0, 0, 0]
    assert not targets["depth_valid"][0, -1, -1]
    assert torch.all(targets["depth"][~targets["depth_valid"]] == 0)
    assert set(targets) == {
        "segmentation",
        "segmentation_valid",
        "depth",
        "depth_valid",
    }
    assert dataset.sample_provenance(0)["official_split"] == "train"


def test_raw_label_mapping_is_explicit_and_unknown_ids_fail(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path, unknown_label=True)
    dataset = _dataset(root)
    with pytest.raises(ValueError, match=r"unmapped raw IDs \[7\]"):
        dataset[1]

    with pytest.raises(ValueError, match="explicitly map"):
        _dataset(root, label_mapping={})
    with pytest.raises(ValueError, match="both mapped and ignored"):
        _dataset(root, source_ignore_labels=(0, 1))
    with pytest.raises(ValueError, match="outside"):
        _dataset(root, label_mapping={1: 0, 2: 2})


def test_label_audit_does_not_read_heldout_labels_without_opt_in(
    tmp_path: Path,
) -> None:
    root = _build_dataset(tmp_path)
    heldout_path = root / "segmentation" / "label_0003.png"
    heldout = np.asarray(Image.open(heldout_path)).copy()
    heldout[-1, -1] = 9
    Image.fromarray(heldout).save(heldout_path)
    index = discover_nyuv2(root)

    assert scan_nyuv2_label_ids(index) == (0, 1, 2)
    assert scan_nyuv2_label_ids(index, include_heldout=True) == (0, 1, 2, 9)


def test_depth_decoder_requires_explicit_units_for_png_and_npy(tmp_path: Path) -> None:
    png = tmp_path / "depth.png"
    Image.fromarray(np.asarray([[0, 1000]], dtype=np.uint16)).save(png)
    np.testing.assert_allclose(
        read_nyuv2_depth(png, unit_scale=0.001), [[0.0, 1.0]]
    )
    npy = tmp_path / "depth.npy"
    np.save(npy, np.asarray([[1.5, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(
        read_nyuv2_depth(npy, unit_scale=1.0), [[1.5, 2.0]]
    )
    with pytest.raises(ValueError, match="explicit positive"):
        read_nyuv2_depth(png, unit_scale=0.0)


def test_light_augmentation_is_synchronized_and_epoch_deterministic(
    tmp_path: Path,
) -> None:
    root = _build_dataset(tmp_path)
    flip = Nyuv2Augmentation(
        horizontal_flip_probability=1.0,
        scale_min=1.0,
        scale_max=1.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
    )
    dataset = _dataset(root, augmentation=flip)
    first_rgb, first_targets = dataset[0]
    second_rgb, second_targets = dataset[0]

    torch.testing.assert_close(first_rgb, second_rgb, rtol=0, atol=0)
    for name in first_targets:
        torch.testing.assert_close(
            first_targets[name], second_targets[name], rtol=0, atol=0
        )
    # Source right half is blue, raw semantic class 2 and 2 m depth. A forced
    # horizontal flip moves all three signals to the left together.
    assert first_rgb[2, 2, 0].item() == pytest.approx(1.0)
    assert first_targets["segmentation"][2, 0].item() == 1
    assert first_targets["depth"][0, 2, 0].item() == pytest.approx(2.0)

    jitter = Nyuv2Augmentation(
        horizontal_flip_probability=0.5,
        scale_min=1.0,
        scale_max=1.10,
        brightness=0.1,
        contrast=0.1,
        saturation=0.08,
    )
    random_dataset = _dataset(root, augmentation=jitter, seed=9)
    random_dataset.set_epoch(3)
    a = random_dataset[0]
    b = random_dataset[0]
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    for name in a[1]:
        torch.testing.assert_close(a[1][name], b[1][name], rtol=0, atol=0)
    epoch_three_rgb = a[0]
    assert any(
        not torch.equal(
            epoch_three_rgb,
            (random_dataset.set_epoch(epoch), random_dataset[0][0])[1],
        )
        for epoch in range(4, 9)
    )


def test_test_split_is_deterministic_and_rejects_train_augmentation(
    tmp_path: Path,
) -> None:
    root = _build_dataset(tmp_path)
    with pytest.raises(ValueError, match="test preprocessing must be deterministic"):
        _dataset(root, split="test", augmentation=Nyuv2Augmentation())
    dataset = _dataset(root, split="test")
    assert len(dataset) == 1
    first = dataset[0]
    dataset.set_epoch(99)
    second = dataset[0]
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
    for name in first[1]:
        torch.testing.assert_close(first[1][name], second[1][name], rtol=0, atol=0)


def test_explicit_subset_supports_inner_train_validation_without_test_leakage(
    tmp_path: Path,
) -> None:
    root = _build_dataset(tmp_path)
    index = discover_nyuv2(root)
    fit = _dataset(root, index=index, samples=index.train[:1])
    inner_val = _dataset(root, index=index, samples=index.train[1:], augmentation=None)

    assert len(fit) == len(inner_val) == 1
    assert fit.sample_provenance(0)["sample_key"] == "1"
    assert inner_val.sample_provenance(0)["sample_key"] == "2"
    with pytest.raises(ValueError, match="outside official train"):
        _dataset(root, index=index, samples=index.test)


def test_aspect_ratio_is_preserved_by_cover_resize_and_center_crop(
    tmp_path: Path,
) -> None:
    root = _build_dataset(tmp_path)
    dataset = _dataset(root, image_size=(4, 4))
    rgb, targets = dataset[0]

    assert rgb.shape == (3, 4, 4)
    assert targets["segmentation"].shape == (4, 4)
    # A 4x6 source becomes a centred 4x4 crop, not a squeezed 4x4 image.
    assert rgb[0, 2, 0].item() == pytest.approx(1.0)
    assert rgb[2, 2, -1].item() == pytest.approx(1.0)


def test_collate_stacks_static_inputs_and_dense_targets(tmp_path: Path) -> None:
    dataset = _dataset(_build_dataset(tmp_path))
    inputs, targets = nyuv2_collate([dataset[0], dataset[1]])

    assert inputs.shape == (2, 3, 4, 6)
    assert targets["segmentation"].shape == (2, 4, 6)
    assert targets["depth"].shape == (2, 1, 4, 6)
    with pytest.raises(ValueError, match="empty"):
        nyuv2_collate([])


def test_json_and_npy_official_split_formats_are_supported(tmp_path: Path) -> None:
    root = _build_dataset(tmp_path)
    (root / "gt_sets" / "train.txt").unlink()
    (root / "gt_sets" / "val.txt").unlink()
    (root / "gt_sets" / "official_train.json").write_text(
        json.dumps({"samples": ["0001", "0002"]}), encoding="utf-8"
    )
    np.save(root / "gt_sets" / "official_test.npy", np.asarray(["0003"]))

    index = discover_nyuv2(root)

    assert len(index.train) == 2
    assert len(index.test) == 1
    assert index.heldout_source_name == "test"
