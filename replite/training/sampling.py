"""Deterministic batch sampling without a disproportionately small tail.

``torch.utils.data.DataLoader(batch_size=...)`` puts every remainder sample in
one final batch.  For example, 1,809 samples at batch size 16 produce 113
full batches and one singleton.  That singleton is undesirable for BatchNorm
and, when gradient accumulation is disabled, receives a full optimizer update.

The sampler below keeps every sample exactly once while distributing the
remainder over all batches.  Batch sizes therefore differ by at most one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from numbers import Integral
from typing import Any

import torch
from torch.utils.data import Sampler


def balanced_batch_sizes(num_samples: int, batch_size: int) -> tuple[int, ...]:
    """Return all-sample batch sizes bounded by ``batch_size``.

    The number of batches matches ordinary non-dropping batching
    (``ceil(num_samples / batch_size)``), but remainder samples are spread
    across those batches instead of being placed in one small tail batch.
    """

    for name, value, allow_zero in (
        ("num_samples", num_samples, True),
        ("batch_size", batch_size, False),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
        if int(value) < (0 if allow_zero else 1):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {qualifier}")
    num_samples = int(num_samples)
    batch_size = int(batch_size)
    if num_samples == 0:
        return ()
    num_batches = (num_samples + batch_size - 1) // batch_size
    smaller, larger_count = divmod(num_samples, num_batches)
    return (smaller + 1,) * larger_count + (smaller,) * (
        num_batches - larger_count
    )


class BalancedBatchSampler(Sampler[list[int]]):
    """Yield deterministic, optionally shuffled, balanced index batches.

    Args:
        data_source: A sized dataset or the integer number of samples.
        batch_size: Maximum requested batch size.
        shuffle: Shuffle every epoch when true.
        seed: Base seed.  Epoch ``e`` uses ``seed + e`` through an isolated
            :class:`torch.Generator`, so resume at an epoch boundary is exact.

    Call :meth:`set_epoch` before iterating a new epoch.  :class:`Trainer`
    does this automatically when the sampler is installed as a loader's
    ``batch_sampler``.
    """

    def __init__(
        self,
        data_source: Sized | int,
        batch_size: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if isinstance(data_source, bool):
            raise ValueError("data_source must be sized or an integer sample count")
        if isinstance(data_source, Integral):
            num_samples = int(data_source)
        else:
            try:
                num_samples = len(data_source)
            except TypeError as exc:
                raise TypeError("data_source must provide len()") from exc
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        self.num_samples = num_samples
        self.batch_size = int(batch_size)
        self.batch_sizes = balanced_batch_sizes(num_samples, batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.batch_sizes)

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic ordering for ``epoch``."""

        if isinstance(epoch, bool) or not isinstance(epoch, Integral) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        if not self.batch_sizes:
            return
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=generator).tolist()
        else:
            indices = list(range(self.num_samples))
        offset = 0
        for size in self.batch_sizes:
            yield indices[offset : offset + size]
            offset += size

    def state_dict(self) -> dict[str, Any]:
        """Return the only mutable state for custom training loops."""

        return {"epoch": self.epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state written by :meth:`state_dict`."""

        if not isinstance(state, dict) or set(state) != {"epoch"}:
            raise ValueError("balanced sampler state must contain only 'epoch'")
        self.set_epoch(state["epoch"])


__all__ = ["BalancedBatchSampler", "balanced_batch_sizes"]
