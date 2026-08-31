"""Export tests: torch.jit.trace, torch.export and ONNX Runtime parity."""

from __future__ import annotations

import pytest
import torch

from replite.backbone import create_backbone

JIT_RTOL, JIT_ATOL = 1e-5, 1e-6
EXPORT_RTOL, EXPORT_ATOL = 1e-4, 1e-5
# Cross-runtime FP32 convolution reductions differ most near zero. This gate
# is based on representative normalized inputs for both random and pinned
# pretrained weights; shape/stride/weight defects remain orders of magnitude larger.
ORT_RTOL, ORT_ATOL = 1e-3, 5e-4
EXPORT_OUT_INDICES = ((0, 1, 2, 3), (0,), (1, 3))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _eval_model(backbone_name: str, out_indices) -> torch.nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(2026)
        model = create_backbone(backbone_name, out_indices=out_indices)
    model.eval()
    return model


def _normalized_input(make_input, *shape: int, seed: int = 1234) -> torch.Tensor:
    inputs = make_input(*shape, seed=seed)
    mean = inputs.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = inputs.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (inputs - mean) / std


def _compare(reference, candidate, rtol: float, atol: float) -> None:
    assert len(reference) == len(
        candidate
    ), f"output count mismatch: expected {len(reference)}, got {len(candidate)}"
    for output_index, (ref, actual) in enumerate(zip(reference, candidate)):
        torch.testing.assert_close(
            ref,
            actual,
            rtol=rtol,
            atol=atol,
            msg=lambda message: f"output {output_index}: {message}",
        )


def test_compare_rejects_missing_outputs() -> None:
    with pytest.raises(AssertionError, match="output count mismatch"):
        _compare((torch.zeros(1), torch.zeros(1)), (torch.zeros(1),), 0.0, 0.0)


@pytest.mark.parametrize("out_indices", EXPORT_OUT_INDICES)
def test_jit_trace(backbone_name, out_indices, make_input) -> None:
    model = _eval_model(backbone_name, out_indices)
    example = _normalized_input(make_input, 1, 3, 64, 96)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)

    for shape in [(1, 3, 64, 96), (2, 3, 128, 256)]:
        inputs = _normalized_input(make_input, *shape, seed=99)
        with torch.no_grad():
            _compare(model(inputs), traced(inputs), JIT_RTOL, JIT_ATOL)


@pytest.mark.parametrize("out_indices", EXPORT_OUT_INDICES)
def test_torch_export_dynamic_height_width(
    backbone_name, out_indices, make_input
) -> None:
    model = _eval_model(backbone_name, out_indices)
    example = _normalized_input(make_input, 2, 3, 64, 96)
    batch = torch.export.Dim("batch", min=1)
    height = torch.export.Dim("height", min=33)
    width = torch.export.Dim("width", min=33)

    with torch.no_grad():
        exported = torch.export.export(
            model,
            (example,),
            dynamic_shapes={"x": {0: batch, 2: height, 3: width}},
        )

    for shape in [(2, 3, 33, 33), (1, 3, 224, 224)]:
        inputs = _normalized_input(make_input, *shape, seed=99)
        with torch.no_grad():
            exported_features = exported.module()(inputs)
            reference = model(inputs)
        _compare(reference, exported_features, EXPORT_RTOL, EXPORT_ATOL)


@pytest.mark.parametrize("out_indices", EXPORT_OUT_INDICES)
def test_onnx_opset17_export_and_runtime_parity(
    backbone_name, out_indices, tmp_path, make_input
) -> None:
    """Validate graph metadata and ORT parity on realistic normalized input."""

    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")

    model = _eval_model(backbone_name, out_indices)
    example = _normalized_input(make_input, 1, 3, 64, 96)
    output_names = [f"c{stage_index + 2}" for stage_index in out_indices]
    output_symbols = {
        name: {
            0: "N",
            2: f"{name.upper()}_H",
            3: f"{name.upper()}_W",
        }
        for name in output_names
    }
    dynamic_axes = {
        "images": {0: "N", 2: "INPUT_H", 3: "INPUT_W"},
        **output_symbols,
    }
    suffix = "_".join(str(index) for index in out_indices)
    onnx_path = tmp_path / f"{backbone_name}_{suffix}.onnx"
    torch.onnx.export(
        model,
        (example,),
        str(onnx_path),
        opset_version=17,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    assert session.get_inputs()[0].shape == ["N", 3, "INPUT_H", "INPUT_W"]
    assert [output.name for output in session.get_outputs()] == output_names
    channels = model.feature_info.channels()
    for name, channel_count, output in zip(
        output_names, channels, session.get_outputs()
    ):
        assert output.shape == [
            "N",
            channel_count,
            f"{name.upper()}_H",
            f"{name.upper()}_W",
        ]

    for shape in [(1, 3, 65, 97), (2, 3, 128, 256)]:
        inputs = _normalized_input(make_input, *shape, seed=99)
        with torch.no_grad():
            reference = model(inputs)
        outputs = session.run(None, {"images": inputs.numpy()})
        converted = [torch.from_numpy(array) for array in outputs]
        assert [tuple(output.shape) for output in converted] == [
            tuple(output.shape) for output in reference
        ]
        _compare(reference, converted, ORT_RTOL, ORT_ATOL)


@pytest.mark.network
def test_pinned_pretrained_torch_and_onnx_export_parity(
    backbone_name, tmp_path, make_input
) -> None:
    """Exercise export numerics with the actual pinned ImageNet-1K weights."""

    onnx = pytest.importorskip("onnx")
    onnxruntime = pytest.importorskip("onnxruntime")
    model = create_backbone(backbone_name, pretrained=True).eval()
    example = _normalized_input(make_input, 2, 3, 64, 96)
    batch = torch.export.Dim("batch", min=1)
    height = torch.export.Dim("height", min=33)
    width = torch.export.Dim("width", min=33)
    with torch.no_grad():
        exported = torch.export.export(
            model,
            (example,),
            dynamic_shapes={"x": {0: batch, 2: height, 3: width}},
        )

    inputs = _normalized_input(make_input, 1, 3, 65, 97, seed=2026)
    with torch.no_grad():
        reference = model(inputs)
        exported_outputs = exported.module()(inputs)
    _compare(reference, exported_outputs, EXPORT_RTOL, EXPORT_ATOL)

    output_names = ["c2", "c3", "c4", "c5"]
    dynamic_axes = {
        "images": {0: "N", 2: "INPUT_H", 3: "INPUT_W"},
        **{
            name: {0: "N", 2: f"{name.upper()}_H", 3: f"{name.upper()}_W"}
            for name in output_names
        },
    }
    onnx_path = tmp_path / f"{backbone_name}_pretrained.onnx"
    torch.onnx.export(
        model,
        (example,),
        str(onnx_path),
        opset_version=17,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    onnx.checker.check_model(str(onnx_path))
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    ort_outputs = [
        torch.from_numpy(array)
        for array in session.run(None, {"images": inputs.numpy()})
    ]
    _compare(reference, ort_outputs, ORT_RTOL, ORT_ATOL)
