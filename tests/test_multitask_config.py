"""Validation tests for static RepLite task and architecture configs."""

from __future__ import annotations

import json

import pytest

from replite.multitask.config import RepLiteConfig, TaskConfig


def test_task_config_reports_stable_active_tasks() -> None:
    tasks = TaskConfig(
        classification_classes=5,
        depth=True,
        detection_classes=3,
    )
    assert tasks.active_tasks == ("detection", "depth", "classification")
    assert tasks.uses_dense_path
    assert not tasks.uses_dense_fusion


def test_dense_fusion_requires_both_dense_tasks() -> None:
    both = TaskConfig(segmentation_classes=4, depth=True)
    segmentation_only = TaskConfig(segmentation_classes=4)
    assert both.uses_dense_fusion
    assert not segmentation_only.uses_dense_fusion


def test_directional_dense_fusion_config_is_serialized_and_subset_stable() -> None:
    tasks = TaskConfig(
        segmentation_classes=4,
        depth=True,
        dense_fusion_direction="seg_to_depth",
        dense_fusion_detach_source=True,
    )
    payload = RepLiteConfig(tasks=tasks).as_dict()["tasks"]

    assert payload["dense_fusion_direction"] == "seg_to_depth"
    assert payload["dense_fusion_detach_source"] is True
    subset = tasks.subset(["depth"])
    assert subset.dense_fusion_direction == "seg_to_depth"
    assert subset.dense_fusion_detach_source is True
    assert not subset.uses_dense_fusion


def test_legacy_task_config_payload_keeps_bidirectional_defaults() -> None:
    legacy_payload = {
        "segmentation_classes": 4,
        "depth": True,
        "gated_dense_fusion": True,
    }

    tasks = TaskConfig(**legacy_payload)

    assert tasks.dense_fusion_direction == "bidirectional"
    assert tasks.dense_fusion_detach_source is False


def test_task_subset_preserves_only_requested_head_metadata() -> None:
    config = RepLiteConfig(
        tasks=TaskConfig(
            detection_classes=7,
            segmentation_classes=4,
            depth=True,
        )
    )
    subset = config.for_tasks(["segmentation"])
    assert subset.active_tasks == ("segmentation",)
    assert subset.tasks.segmentation_classes == 4
    assert not subset.tasks.uses_dense_fusion


@pytest.mark.parametrize(
    "selection",
    [[], ["classification"], ["depth", "depth"], [1]],
)
def test_task_subset_rejects_invalid_or_unavailable_selection(selection) -> None:
    tasks = TaskConfig(detection_classes=3, depth=True)
    with pytest.raises(ValueError):
        tasks.subset(selection)


def test_config_is_json_serializable_and_immutable() -> None:
    config = RepLiteConfig(tasks=TaskConfig(segmentation_classes=4))
    payload = config.as_dict()
    assert json.loads(json.dumps(payload))["recurrence_steps"] == 3
    with pytest.raises(Exception):
        config.recurrence_steps = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"detection_classes": 0},
        {"segmentation_classes": True},
        {"classification_classes": -1},
        {"depth": 1},
        {"gated_dense_fusion": 1, "depth": True},
        {
            "segmentation_classes": 2,
            "depth": True,
            "dense_fusion_direction": "sideways",
        },
        {
            "segmentation_classes": 2,
            "depth": True,
            "dense_fusion_detach_source": 1,
        },
    ],
)
def test_task_config_rejects_empty_or_invalid_tasks(kwargs) -> None:
    with pytest.raises(ValueError):
        TaskConfig(**kwargs)


@pytest.mark.parametrize(
    "overrides",
    [
        {"backbone_name": ""},
        {"pretrained": 1},
        {"tasks": {"depth": True}},
        {"recurrence_steps": 0},
        {"neck_channels": True},
        {"detection_reg_max": -1},
        {"use_sppf": 1},
    ],
)
def test_model_config_rejects_invalid_values(overrides) -> None:
    overrides.setdefault("tasks", TaskConfig(depth=True))
    with pytest.raises(ValueError):
        RepLiteConfig(**overrides)
