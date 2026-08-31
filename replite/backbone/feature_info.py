"""Metadata describing feature stages exposed by a Replite backbone."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Sequence


@dataclass(frozen=True)
class StageSpec:
    """Description of one native trunk stage (C2..C5)."""

    module: str
    num_chs: int
    reduction: int
    blocks_end: int


def validate_out_indices(
    out_indices: Sequence[int], *, num_stages: int = 4
) -> tuple[int, ...]:
    """Return canonical output indices or raise a consistent ``ValueError``."""

    try:
        raw_indices = tuple(out_indices)
    except TypeError as exc:
        raise ValueError(
            "out_indices must be a sequence of integer stage indices"
        ) from exc
    if not raw_indices:
        raise ValueError("out_indices must not be empty")
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in raw_indices
    ):
        raise ValueError(
            f"out_indices must contain only integer stage indices, got {raw_indices!r}"
        )

    indices = tuple(int(index) for index in raw_indices)
    if len(set(indices)) != len(indices):
        raise ValueError(f"out_indices must not contain duplicates, got {indices!r}")
    if tuple(sorted(indices)) != indices:
        raise ValueError(f"out_indices must be strictly increasing, got {indices!r}")
    invalid = [index for index in indices if not 0 <= index < num_stages]
    if invalid:
        raise ValueError(
            f"out_indices must only contain stage indices 0..{num_stages - 1}, "
            f"got {indices!r}"
        )
    return indices


class FeatureInfo:
    """Timm-compatible metadata for native C2..C5 feature stages.

    Omitting ``idx`` returns values for ``out_indices``. Passing an integer or
    sequence addresses the complete stage table, matching timm semantics.
    """

    def __init__(self, stages: Sequence[StageSpec], out_indices: Sequence[int]) -> None:
        stages = tuple(stages)
        if not stages:
            raise ValueError("stages must not be empty")
        self.out_indices = validate_out_indices(out_indices, num_stages=len(stages))

        previous_reduction = 0
        previous_blocks_end = -1
        info: list[dict[str, Any]] = []
        for index, stage in enumerate(stages):
            if not stage.module:
                raise ValueError(f"stage {index} must define a module name")
            if stage.num_chs <= 0:
                raise ValueError(f"stage {index} must have positive channels")
            if stage.reduction <= 0 or stage.reduction < previous_reduction:
                raise ValueError("stage reductions must be positive and non-decreasing")
            if stage.blocks_end <= previous_blocks_end:
                raise ValueError("stage blocks_end values must be strictly increasing")
            previous_reduction = stage.reduction
            previous_blocks_end = stage.blocks_end
            info.append(
                {
                    "index": index,
                    "module": stage.module,
                    "num_chs": stage.num_chs,
                    "reduction": stage.reduction,
                }
            )
        self.info = info

    @classmethod
    def _from_info(
        cls, info: list[dict[str, Any]], out_indices: Sequence[int]
    ) -> "FeatureInfo":
        instance = cls.__new__(cls)
        instance.out_indices = validate_out_indices(out_indices, num_stages=len(info))
        instance.info = deepcopy(info)
        return instance

    def from_other(self, out_indices: Sequence[int]) -> "FeatureInfo":
        """Return an independent view selecting different output stages."""

        return type(self)._from_info(self.info, out_indices)

    def get(self, key: str, idx: int | Sequence[int] | None = None) -> Any:
        """Return ``key`` at one, several, or all selected stage indices."""

        if idx is None:
            return [self.info[index][key] for index in self.out_indices]
        if isinstance(idx, Sequence) and not isinstance(idx, (str, bytes)):
            return [self.info[int(index)][key] for index in idx]
        return self.info[int(idx)][key]

    def get_dicts(
        self,
        keys: Sequence[str] | None = None,
        idx: int | Sequence[int] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return complete or key-filtered metadata dictionaries."""

        indices: int | Sequence[int] = self.out_indices if idx is None else idx

        def select(index: int) -> dict[str, Any]:
            entry = self.info[index]
            if keys is None:
                return entry.copy()
            return {key: entry[key] for key in keys}

        if isinstance(indices, Sequence) and not isinstance(indices, (str, bytes)):
            return [select(int(index)) for index in indices]
        return select(int(indices))

    def channels(self, idx: int | Sequence[int] | None = None) -> Any:
        """Return feature channel counts using timm indexing semantics."""

        return self.get("num_chs", idx)

    def reduction(self, idx: int | Sequence[int] | None = None) -> Any:
        """Return feature reductions using timm indexing semantics."""

        return self.get("reduction", idx)

    def module_name(self, idx: int | Sequence[int] | None = None) -> Any:
        """Return feature module names using timm indexing semantics."""

        return self.get("module", idx)

    @property
    def num_outputs(self) -> int:
        """Number of stages selected by ``out_indices``."""

        return len(self.out_indices)

    def __getitem__(self, item: int | slice) -> Any:
        return self.info[item]

    def __len__(self) -> int:
        return len(self.info)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(out_indices={self.out_indices}, "
            f"channels={self.channels()}, reduction={self.reduction()}, "
            f"module_name={self.module_name()})"
        )
