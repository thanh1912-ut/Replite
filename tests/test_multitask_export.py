"""torch.export and ONNX Runtime checks for task-pruned RepLite models."""

from __future__ import annotations

import pytest
import torch

from replite.multitask import RepLiteConfig, RepLiteMultiTaskModel, TaskConfig


def _all_task_config() -> RepLiteConfig:
    return RepLiteConfig(
        tasks=TaskConfig(
            detection_classes=3,
            segmentation_classes=2,
            depth=True,
            classification_classes=4,
        ),
        backbone_name="mobilenetv3_small_050",
        recurrent_c4_channels=16,
        recurrent_c5_channels=24,
        neck_channels=16,
        dense_channels=12,
        task_adapter_channels=12,
        detection_head_channels=16,
    )


def _as_tuple(output) -> tuple[torch.Tensor, ...]:
    return output if isinstance(output, tuple) else (output,)


@pytest.mark.parametrize(
    "task",
    ["detection", "segmentation", "depth", "classification"],
)
def test_task_wrapper_torch_export_parity(task) -> None:
    torch.manual_seed(2026)
    model = RepLiteMultiTaskModel(_all_task_config()).eval()
    wrapper = model.export_task(task)
    example = torch.randn(2, 3, 64, 96)
    batch = torch.export.Dim("batch", min=1)
    height = torch.export.Dim("height", min=33)
    width = torch.export.Dim("width", min=33)
    with torch.no_grad():
        exported = torch.export.export(
            wrapper,
            (example,),
            dynamic_shapes={"images": {0: batch, 2: height, 3: width}},
        )

    candidate = torch.randn(1, 3, 65, 97)
    with torch.no_grad():
        reference = _as_tuple(wrapper(candidate))
        actual = _as_tuple(exported.module()(candidate))
    assert len(reference) == len(actual)
    for expected, observed in zip(reference, actual):
        torch.testing.assert_close(expected, observed, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "task",
    ["detection", "segmentation", "depth", "classification"],
)
def test_task_onnx_opset17_runtime_parity(task, tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    model = RepLiteMultiTaskModel(_all_task_config()).eval()
    wrapper = model.export_task(task)
    example = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        example_outputs = _as_tuple(wrapper(example))
    output_names = [f"output_{index}" for index in range(len(example_outputs))]
    dynamic_axes = {"images": {0: "N", 2: "INPUT_H", 3: "INPUT_W"}}
    for index, (name, output) in enumerate(zip(output_names, example_outputs)):
        axes = {0: "N"}
        if output.ndim == 4:
            axes.update({2: f"OUTPUT_{index}_H", 3: f"OUTPUT_{index}_W"})
        dynamic_axes[name] = axes

    path = tmp_path / f"replite_{task}.onnx"
    torch.onnx.export(
        wrapper,
        (example,),
        str(path),
        opset_version=17,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    onnx.checker.check_model(str(path))
    session = onnxruntime.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )

    images = torch.randn(1, 3, 65, 97)
    with torch.no_grad():
        expected = _as_tuple(wrapper(images))
    arrays = session.run(None, {"images": images.numpy()})
    assert len(arrays) == len(expected)
    for expected_tensor, array in zip(expected, arrays):
        actual = torch.from_numpy(array)
        assert actual.shape == expected_tensor.shape
        torch.testing.assert_close(
            expected_tensor,
            actual,
            rtol=1e-3,
            atol=5e-4,
        )
