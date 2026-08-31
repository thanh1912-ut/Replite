"""End-to-end contracts for the complete RepLite multi-task model."""

from __future__ import annotations

import json

import pytest
import torch

from replite.multitask.config import RepLiteConfig, TaskConfig
from replite.multitask.model import (
    RepLiteMultiTaskModel,
    TaskExportWrapper,
    detach_state,
)


def _config(backbone: str, tasks: TaskConfig, **kwargs) -> RepLiteConfig:
    return RepLiteConfig(
        backbone_name=backbone,
        tasks=tasks,
        recurrent_c4_channels=16,
        recurrent_c5_channels=24,
        neck_channels=16,
        dense_channels=12,
        task_adapter_channels=12,
        detection_head_channels=16,
        **kwargs,
    )


@pytest.mark.parametrize(
    "backbone",
    ["mobilenetv3_small_050", "mobilenetv4_conv_small"],
)
def test_full_model_outputs_every_enabled_task(backbone) -> None:
    tasks = TaskConfig(
        detection_classes=3,
        segmentation_classes=4,
        depth=True,
        classification_classes=5,
    )
    model = RepLiteMultiTaskModel(_config(backbone, tasks)).eval()
    images = torch.randn(2, 3, 65, 97)

    with torch.no_grad():
        output = model(images)

    assert model.active_tasks == (
        "detection",
        "segmentation",
        "depth",
        "classification",
    )
    assert output.detection is not None
    assert output.segmentation is not None
    assert output.depth is not None
    assert output.classification is not None
    assert output.segmentation.shape == (2, 4, 65, 97)
    assert output.depth.shape == (2, 1, 65, 97)
    assert output.classification.shape == (2, 5)
    assert torch.all(output.depth > 0)
    assert [tensor.shape[1] for tensor in output.detection.cls_logits] == [3, 3, 3]


@pytest.mark.parametrize(
    "tasks,expected_neck_path",
    [
        (TaskConfig(detection_classes=3), "detection"),
        (TaskConfig(segmentation_classes=2), "dense"),
        (TaskConfig(depth=True), "dense"),
        (TaskConfig(classification_classes=5), "neither"),
    ],
)
def test_inactive_task_paths_are_physically_pruned(tasks, expected_neck_path) -> None:
    model = RepLiteMultiTaskModel(_config("mobilenetv3_small_050", tasks))
    has_detection = hasattr(model.neck, "detection_path")
    has_dense = hasattr(model.neck, "dense_path")
    assert has_detection == (expected_neck_path == "detection")
    assert has_dense == (expected_neck_path == "dense")
    assert hasattr(model, "detection_head") == ("detection" in tasks.active_tasks)
    assert hasattr(model, "segmentation_head") == ("segmentation" in tasks.active_tasks)
    assert hasattr(model, "depth_head") == ("depth" in tasks.active_tasks)
    assert hasattr(model, "classification_head") == (
        "classification" in tasks.active_tasks
    )
    needs_level4 = expected_neck_path in ("detection", "dense")
    needs_level5 = (
        expected_neck_path == "detection" or "classification" in tasks.active_tasks
    )
    assert hasattr(model.neck, "recurrent4") is needs_level4
    assert hasattr(model.neck, "recurrent5") is needs_level5
    assert hasattr(model.neck, "c4_projection") is needs_level4
    assert hasattr(model.neck, "c5_projection") is needs_level5
    expected_indices = (0, 1, 2, 3) if needs_level5 else (0, 1, 2)
    assert model.backbone.out_indices == expected_indices


def test_dense_cross_task_fusion_is_physically_optional() -> None:
    fused = RepLiteMultiTaskModel(
        _config(
            "mobilenetv3_small_050",
            TaskConfig(segmentation_classes=2, depth=True),
        )
    )
    independent = RepLiteMultiTaskModel(
        _config(
            "mobilenetv3_small_050",
            TaskConfig(
                segmentation_classes=2,
                depth=True,
                gated_dense_fusion=False,
            ),
        )
    )
    assert hasattr(fused, "dense_fusion")
    assert not hasattr(independent, "dense_fusion")
    assert sum(p.numel() for p in independent.parameters()) < sum(
        p.numel() for p in fused.parameters()
    )


def test_static_refinement_runs_backbone_once_and_matches_repeated_clip() -> None:
    model = RepLiteMultiTaskModel(
        _config(
            "mobilenetv3_small_050",
            TaskConfig(segmentation_classes=2, depth=True),
        )
    ).eval()
    images = torch.randn(1, 3, 65, 97)
    calls = 0

    def count_backbone_call(module, args):
        nonlocal calls
        calls += 1

    handle = model.backbone.register_forward_pre_hook(count_backbone_call)
    with torch.no_grad():
        static = model.forward_static(images, steps=3)
    handle.remove()
    assert calls == 1

    clip = images[:, None].expand(-1, 3, -1, -1, -1).contiguous()
    with torch.no_grad():
        sequential, _ = model.forward_sequence(clip)
    torch.testing.assert_close(static.segmentation, sequential.segmentation)
    torch.testing.assert_close(static.depth, sequential.depth)


def test_streaming_continuation_matches_whole_clip() -> None:
    model = RepLiteMultiTaskModel(
        _config("mobilenetv3_small_050", TaskConfig(segmentation_classes=2))
    ).eval()
    frames = torch.randn(1, 3, 3, 64, 96)
    with torch.no_grad():
        whole, whole_state = model.forward_sequence(frames)
        _, prefix_state = model.forward_sequence(frames[:, :2])
        continued, continued_state = model.forward_step(frames[:, 2], prefix_state)

    torch.testing.assert_close(whole.segmentation, continued.segmentation)
    for expected_level, actual_level in zip(whole_state, continued_state):
        if expected_level is None or actual_level is None:
            assert expected_level is actual_level is None
            continue
        for expected, actual in zip(expected_level, actual_level):
            torch.testing.assert_close(expected, actual)


def test_detach_state_breaks_history_without_changing_values() -> None:
    model = RepLiteMultiTaskModel(
        _config("mobilenetv3_small_050", TaskConfig(depth=True))
    )
    _, state = model.forward_step(torch.randn(2, 3, 64, 96))
    detached = detach_state(state)
    for original_level, detached_level in zip(state, detached):
        if original_level is None or detached_level is None:
            assert original_level is detached_level is None
            continue
        for original, clean in zip(original_level, detached_level):
            torch.testing.assert_close(original, clean)
            assert original.requires_grad
            assert not clean.requires_grad


@pytest.mark.parametrize(
    "tasks",
    [
        TaskConfig(detection_classes=3),
        TaskConfig(segmentation_classes=2),
        TaskConfig(depth=True),
        TaskConfig(classification_classes=4),
        TaskConfig(segmentation_classes=2, depth=True),
    ],
)
def test_every_single_task_artifact_has_no_disconnected_parameters(tasks) -> None:
    model = RepLiteMultiTaskModel(_config("mobilenetv3_small_050", tasks)).train()
    output = model(torch.randn(2, 3, 64, 96))
    losses = []
    if output.detection is not None:
        losses.extend(tensor.mean() for group in output.detection for tensor in group)
    if output.segmentation is not None:
        losses.append(output.segmentation.mean())
    if output.depth is not None:
        losses.append(output.depth.mean())
    if output.classification is not None:
        losses.append(output.classification.mean())
    sum(losses).backward()

    missing = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    assert not missing, f"parameters disconnected from task loss: {missing}"


def test_sequence_decodes_only_the_final_frame() -> None:
    model = RepLiteMultiTaskModel(
        _config(
            "mobilenetv3_small_050",
            TaskConfig(detection_classes=3, segmentation_classes=2),
        )
    ).train()
    calls = {"detection": 0, "dense": 0}

    def count_detection(module, args, output):
        calls["detection"] += 1

    def count_dense(module, args, output):
        calls["dense"] += 1

    detection_handle = model.neck.detection_path["lateral5"].register_forward_hook(
        count_detection
    )
    dense_handle = model.neck.dense_path["lateral2"].register_forward_hook(count_dense)
    model.forward_sequence(torch.randn(2, 4, 3, 64, 96))
    detection_handle.remove()
    dense_handle.remove()

    assert calls == {"detection": 1, "dense": 1}


def test_full_multitask_backward_reaches_every_registered_parameter() -> None:
    tasks = TaskConfig(
        detection_classes=3,
        segmentation_classes=2,
        depth=True,
        classification_classes=4,
    )
    model = RepLiteMultiTaskModel(_config("mobilenetv3_small_050", tasks))
    output = model(torch.randn(2, 3, 64, 96))
    assert output.detection is not None
    assert output.segmentation is not None
    assert output.depth is not None
    assert output.classification is not None

    loss = output.segmentation.mean() + output.depth.mean()
    loss = loss + output.classification.mean()
    for group in output.detection:
        loss = loss + sum(tensor.mean() for tensor in group)
    loss.backward()

    missing = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    assert not missing, f"parameters disconnected from the full loss: {missing}"


def test_metadata_is_json_serializable_and_contains_backbone_provenance() -> None:
    model = RepLiteMultiTaskModel(
        _config("mobilenetv4_conv_small", TaskConfig(depth=True))
    )
    metadata = json.loads(json.dumps(model.model_metadata))
    assert metadata["schema_version"] == 1
    assert metadata["config"]["tasks"]["depth"] is True
    assert metadata["backbone"]["architecture"] == "mobilenetv4_conv_small"


def test_multitask_checkpoint_round_trip_is_exact(tmp_path) -> None:
    config = _config(
        "mobilenetv3_small_050",
        TaskConfig(detection_classes=3, segmentation_classes=2, depth=True),
    )
    torch.manual_seed(2026)
    model = RepLiteMultiTaskModel(config).eval()
    checkpoint = tmp_path / "replite_multitask.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": model.model_metadata,
        },
        checkpoint,
    )
    restored = RepLiteMultiTaskModel(config).eval()
    payload = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(payload["model"], strict=True)
    assert payload["metadata"] == model.model_metadata

    images = torch.randn(1, 3, 65, 97)
    with torch.no_grad():
        expected = model(images)
        actual = restored(images)
    assert expected.detection is not None and actual.detection is not None
    for expected_group, actual_group in zip(expected.detection, actual.detection):
        for expected_tensor, actual_tensor in zip(expected_group, actual_group):
            torch.testing.assert_close(expected_tensor, actual_tensor, rtol=0, atol=0)
    torch.testing.assert_close(
        expected.segmentation, actual.segmentation, rtol=0, atol=0
    )
    torch.testing.assert_close(expected.depth, actual.depth, rtol=0, atol=0)


@pytest.mark.parametrize(
    "backbone,max_parameters",
    [
        ("mobilenetv3_small_050", 400_000),
        ("mobilenetv4_conv_small", 1_300_000),
    ],
)
def test_default_width_full_model_stays_within_lightweight_budget(
    backbone, max_parameters
) -> None:
    tasks = TaskConfig(
        detection_classes=10,
        segmentation_classes=3,
        depth=True,
    )
    model = RepLiteMultiTaskModel(RepLiteConfig(tasks=tasks, backbone_name=backbone))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count < max_parameters


def test_export_wrappers_return_tensor_only_task_contracts() -> None:
    tasks = TaskConfig(
        detection_classes=3,
        segmentation_classes=2,
        depth=True,
        classification_classes=4,
    )
    model = RepLiteMultiTaskModel(_config("mobilenetv3_small_050", tasks)).eval()
    images = torch.randn(1, 3, 64, 96)

    assert not model.export_task("segmentation").training
    with torch.no_grad():
        detection = model.export_task("detection")(images)
        segmentation = model.export_task("segmentation")(images)
        depth = model.export_task("depth")(images)
        classification = model.export_task("classification")(images)
    assert isinstance(detection, tuple) and len(detection) == 9
    assert segmentation.shape == (1, 2, 64, 96)
    assert depth.shape == (1, 1, 64, 96)
    assert classification.shape == (1, 4)
    assert all(isinstance(tensor, torch.Tensor) for tensor in detection)

    with pytest.raises(ValueError):
        TaskExportWrapper(model, "pose")


def test_export_snapshot_does_not_advance_global_rng() -> None:
    model = RepLiteMultiTaskModel(
        _config(
            "mobilenetv3_small_050",
            TaskConfig(segmentation_classes=2, depth=True),
        )
    ).eval()
    torch.manual_seed(909)
    before = torch.get_rng_state().clone()
    model.export_task("segmentation")
    after = torch.get_rng_state()
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_export_wrappers_match_full_outputs_and_skip_unrelated_paths() -> None:
    tasks = TaskConfig(
        detection_classes=3,
        segmentation_classes=2,
        depth=True,
        classification_classes=4,
    )
    model = RepLiteMultiTaskModel(_config("mobilenetv3_small_050", tasks)).eval()
    with torch.no_grad():
        model.dense_fusion.depth_to_seg_scale.fill_(0.7)
        model.dense_fusion.seg_to_depth_scale.fill_(-0.4)

    images = torch.randn(1, 3, 65, 97)
    with torch.no_grad():
        full = model(images)

    expected_routes = {
        "detection": {"det_decoder", "det_head"},
        "segmentation": {
            "dense_decoder",
            "seg_adapter",
            "depth_adapter",
            "fusion",
            "seg_head",
        },
        "depth": {
            "dense_decoder",
            "seg_adapter",
            "depth_adapter",
            "fusion",
            "depth_head",
        },
        "classification": {"cls_head"},
    }
    for task, expected_route in expected_routes.items():
        wrapper = model.export_task(task)
        assert wrapper.active_tasks == (task,)
        assert not hasattr(wrapper, "model")
        metadata = json.loads(json.dumps(wrapper.model_metadata))
        assert metadata["task"] == task
        assert metadata["config"] == model.config.for_tasks([task]).as_dict()
        assert metadata["source"] == model.model_metadata
        assert metadata["dependencies"]["dense_fusion"] is (
            task in ("segmentation", "depth")
        )
        target = wrapper._task_model
        assert target.active_tasks == (task,)
        present = set()
        if hasattr(target.neck, "detection_path"):
            present.add("det_decoder")
        if hasattr(target.neck, "dense_path"):
            present.add("dense_decoder")
        for name, attribute in (
            ("det_head", "detection_head"),
            ("seg_adapter", "segmentation_adapter"),
            ("depth_adapter", "depth_adapter"),
            ("fusion", "dense_fusion"),
            ("seg_head", "segmentation_head"),
            ("depth_head", "depth_head"),
            ("cls_head", "classification_head"),
        ):
            if hasattr(target, attribute):
                present.add(name)
        assert present == expected_route
        if task in ("segmentation", "depth"):
            assert target.backbone.out_indices == (0, 1, 2)
            assert hasattr(target.neck, "recurrent4")
            assert not hasattr(target.neck, "recurrent5")
            if task == "segmentation":
                assert hasattr(target.dense_fusion, "depth_to_seg")
                assert not hasattr(target.dense_fusion, "seg_to_depth")
                assert not hasattr(target.dense_fusion, "seg_to_depth_scale")
            else:
                assert hasattr(target.dense_fusion, "seg_to_depth")
                assert not hasattr(target.dense_fusion, "depth_to_seg")
                assert not hasattr(target.dense_fusion, "depth_to_seg_scale")
        elif task == "classification":
            assert not hasattr(target.neck, "recurrent4")
            assert hasattr(target.neck, "recurrent5")
        else:
            assert hasattr(target.neck, "recurrent4")
            assert hasattr(target.neck, "recurrent5")
        with torch.no_grad():
            routed = wrapper(images)
        reference = getattr(full, task)
        if task == "detection":
            assert reference is not None
            reference = tuple(tensor for group in reference for tensor in group)
            for actual_tensor, expected_tensor in zip(routed, reference):
                torch.testing.assert_close(
                    actual_tensor, expected_tensor, rtol=0, atol=0
                )
        else:
            torch.testing.assert_close(routed, reference, rtol=0, atol=0)


@pytest.mark.parametrize("task", ["segmentation", "depth"])
def test_dense_export_preserves_disabled_fusion_runtime_state(task) -> None:
    model = RepLiteMultiTaskModel(
        _config(
            "mobilenetv3_small_050",
            TaskConfig(segmentation_classes=2, depth=True),
        )
    ).eval()
    with torch.no_grad():
        model.dense_fusion.depth_to_seg_scale.fill_(0.9)
        model.dense_fusion.seg_to_depth_scale.fill_(-0.8)
    model.dense_fusion.enabled = False
    images = torch.randn(1, 3, 65, 97)

    with torch.no_grad():
        reference = getattr(model(images), task)
        wrapper = model.export_task(task)
        actual = wrapper(images)

    assert not hasattr(wrapper._task_model, "dense_fusion")
    assert wrapper.model_metadata["dependencies"]["dense_fusion"] is False
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "task",
    ["detection", "segmentation", "depth", "classification"],
)
def test_task_export_snapshot_has_no_disconnected_parameters(task) -> None:
    source = RepLiteMultiTaskModel(
        _config(
            "mobilenetv3_small_050",
            TaskConfig(
                detection_classes=3,
                segmentation_classes=2,
                depth=True,
                classification_classes=4,
            ),
        )
    ).train()
    wrapper = source.export_task(task).train()
    output = wrapper(torch.randn(2, 3, 64, 96))
    tensors = output if isinstance(output, tuple) else (output,)
    sum(tensor.mean() for tensor in tensors).backward()

    missing = [
        name for name, parameter in wrapper.named_parameters() if parameter.grad is None
    ]
    assert not missing, f"disconnected export parameters: {missing}"


def test_switch_to_deploy_preserves_eval_predictions() -> None:
    model = RepLiteMultiTaskModel(
        _config(
            "mobilenetv3_small_050",
            TaskConfig(segmentation_classes=2, depth=True),
        )
    ).eval()
    images = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        before = model(images)
        model.switch_to_deploy()
        after = model(images)

    torch.testing.assert_close(
        before.segmentation, after.segmentation, atol=2e-6, rtol=1e-5
    )
    torch.testing.assert_close(before.depth, after.depth, atol=2e-6, rtol=1e-5)
    assert all(block.deploy for block in model.modules() if hasattr(block, "deploy"))


@pytest.mark.parametrize(
    "bad_input,error",
    [
        (torch.zeros(1, 1, 32, 32), ValueError),
        (torch.zeros(1, 3, 0, 32), ValueError),
        (torch.zeros(1, 3, 32, 32, dtype=torch.int64), TypeError),
        (torch.zeros(1, 0, 3, 32, 32), ValueError),
    ],
)
def test_model_rejects_invalid_image_or_clip_inputs(bad_input, error) -> None:
    model = RepLiteMultiTaskModel(
        _config("mobilenetv3_small_050", TaskConfig(depth=True))
    )
    with pytest.raises(error):
        model(bad_input)
