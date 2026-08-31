"""Expected contract values for every registered backbone."""

from __future__ import annotations

from typing import Any

BACKBONE_NAMES: tuple[str, ...] = ("mobilenetv3_small_050", "mobilenetv4_conv_small")

EXPECTED: dict[str, dict[str, Any]] = {
    "mobilenetv3_small_050": {
        "param_count": 257_888,
        "full_timm_arch": "mobilenetv3_small_050",
        "trunk_block_groups": 5,
        "removed_block_groups": (5,),
        "channels": (8, 16, 24, 48),
        "reduction": (4, 8, 16, 32),
        "module_names": ("blocks.0", "blocks.1", "blocks.3", "blocks.4"),
        "feature_block_ends": (0, 1, 3, 4),
        "projection_channels": (288,),
        "pretrained_cfg": {
            "architecture": "mobilenetv3_small_050",
            "dataset": "imagenet-1k",
            "filename": "model.safetensors",
            "input_size": (3, 224, 224),
            "test_input_size": (3, 224, 224),
            "fixed_input_size": False,
            "interpolation": "bicubic",
            "crop_pct": 0.875,
            "test_crop_pct": 0.875,
            "crop_mode": "center",
            "license": "apache-2.0",
            "mean": (0.485, 0.456, 0.406),
            "repository": "timm/mobilenetv3_small_050.lamb_in1k",
            "revision": "f58e7345afe2832abd6f81cc60f67cd1ddf7ce00",
            "sha256": "2e3f6937afd4b3704450518a2710168775d6c70ebfb7a0e9aaf06200c6fbe0c4",
            "std": (0.229, 0.224, 0.225),
        },
    },
    "mobilenetv4_conv_small": {
        "param_count": 1_136_864,
        "full_timm_arch": "mobilenetv4_conv_small",
        "trunk_block_groups": 4,
        "removed_block_groups": (4,),
        "channels": (32, 64, 96, 128),
        "reduction": (4, 8, 16, 32),
        "module_names": ("blocks.0", "blocks.1", "blocks.2", "blocks.3"),
        "feature_block_ends": (0, 1, 2, 3),
        "projection_channels": (960,),
        "pretrained_cfg": {
            "architecture": "mobilenetv4_conv_small",
            "dataset": "imagenet-1k",
            "filename": "model.safetensors",
            "input_size": (3, 224, 224),
            "test_input_size": (3, 256, 256),
            "fixed_input_size": False,
            "interpolation": "bicubic",
            "crop_pct": 0.875,
            "test_crop_pct": 0.95,
            "crop_mode": "center",
            "license": "apache-2.0",
            "mean": (0.485, 0.456, 0.406),
            "repository": "timm/mobilenetv4_conv_small.e2400_r224_in1k",
            "revision": "7249cacba963f438597f373327119b22f4d3a848",
            "sha256": "7a7102ec18f62bbfb555b6fe829bbb5af749516b84174926c29ffdfdfc03aec4",
            "std": (0.229, 0.224, 0.225),
        },
    },
}

REQUIRED_CFG_KEYS = (
    "dataset",
    "architecture",
    "repository",
    "revision",
    "sha256",
    "filename",
    "mean",
    "std",
    "input_size",
    "test_input_size",
    "fixed_input_size",
    "interpolation",
    "crop_pct",
    "test_crop_pct",
    "crop_mode",
    "license",
)
