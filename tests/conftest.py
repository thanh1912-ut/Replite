"""Shared fixtures for the backbone test suite."""

from __future__ import annotations

from typing import Callable

import pytest
import torch

from replite.backbone import list_backbones

from .specs import EXPECTED


@pytest.fixture(params=list_backbones(), ids=list_backbones())
def backbone_name(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture()
def make_input() -> Callable[..., torch.Tensor]:
    """Deterministic input factory so every comparison uses fixed seeds."""

    def _make(*shape: int, seed: int = 1234) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        return torch.rand(*shape, generator=generator)

    return _make


@pytest.fixture()
def expected_for(backbone_name: str) -> dict:
    return EXPECTED[backbone_name]
