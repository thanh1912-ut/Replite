"""Atomic trusted-local training checkpoints with exact epoch resume state."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Optimizer


_SCHEMA_VERSION = 1


class CheckpointIntegrityError(RuntimeError):
    """A checkpoint or its SHA-256 sidecar is missing, corrupt, or unreadable."""


def _unwrap_model(model: nn.Module) -> nn.Module:
    module = getattr(model, "module", None)
    return module if isinstance(module, nn.Module) else model


def _jsonable(value: object) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _verify_checksum(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    try:
        fields = sidecar.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise CheckpointIntegrityError(
            f"missing or unreadable checkpoint checksum: {sidecar}"
        ) from exc
    if len(fields) != 2 or fields[1] != path.name:
        raise CheckpointIntegrityError(f"invalid checkpoint checksum sidecar: {sidecar}")
    expected = fields[0].lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise CheckpointIntegrityError(f"invalid SHA-256 in sidecar: {sidecar}")
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        raise CheckpointIntegrityError(f"checkpoint is unreadable: {path}") from exc
    if actual != expected:
        raise CheckpointIntegrityError(
            f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
        )


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy (when installed), Torch, and CUDA RNG streams."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        import numpy as np
    except ImportError:
        state["numpy"] = None
    else:
        state["numpy"] = np.random.get_state()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    required = {"python", "torch_cpu", "torch_cuda", "numpy"}
    missing = required - set(state)
    if missing:
        raise ValueError("RNG state is missing: " + ", ".join(sorted(missing)))
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(cuda_states)
    numpy_state = state["numpy"]
    if numpy_state is not None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("checkpoint contains NumPy RNG state but NumPy is absent") from exc
        np.random.set_state(numpy_state)


def _model_metadata(model: nn.Module) -> Any:
    metadata = getattr(_unwrap_model(model), "model_metadata", None)
    if callable(metadata):
        metadata = metadata()
    return copy.deepcopy(metadata)


def _metadata_signature(metadata: object) -> Any:
    """Compare architecture while ignoring initialization-only provenance."""

    if not isinstance(metadata, Mapping):
        return _jsonable(metadata)
    if "config" not in metadata and "backbone" not in metadata:
        return _jsonable(metadata)
    signature: dict[str, Any] = {}
    if "model" in metadata:
        signature["model"] = metadata["model"]
    if isinstance(metadata.get("config"), Mapping):
        config = dict(metadata["config"])
        config.pop("pretrained", None)
        signature["config"] = config
    if isinstance(metadata.get("backbone"), Mapping):
        backbone = dict(metadata["backbone"])
        backbone.pop("weights", None)
        signature["backbone"] = backbone
    return _jsonable(signature)


def _state_dict_or_none(value: object) -> Any:
    method = getattr(value, "state_dict", None)
    return None if value is None or not callable(method) else method()


def _build_payload(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch_completed: int,
    global_step: int,
    trainer_config: object,
    scheduler: object | None = None,
    scaler: object | None = None,
    criterion: object | None = None,
    best_metrics: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(epoch_completed, bool) or not isinstance(epoch_completed, int):
        raise ValueError("epoch_completed must be an integer")
    if epoch_completed < -1:
        raise ValueError("epoch_completed must be at least -1")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        raise ValueError("global_step must be a non-negative integer")
    if global_step < 0:
        raise ValueError("global_step must be a non-negative integer")
    config_value = _jsonable(trainer_config)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "checkpoint_kind": "replite_training_epoch_boundary",
        "progress": {
            "epoch_completed": epoch_completed,
            "next_epoch": epoch_completed + 1,
            "batch_in_epoch": 0,
            "accumulation_index": 0,
            "global_step": global_step,
        },
        "model": _unwrap_model(model).state_dict(),
        "model_metadata": _model_metadata(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": _state_dict_or_none(scheduler),
        "scaler": _state_dict_or_none(scaler),
        "criterion": _state_dict_or_none(criterion),
        "rng": capture_rng_state(),
        "trainer_config": config_value,
        "trainer_config_sha256": _canonical_hash(config_value),
        "best_metrics": dict(best_metrics or {}),
        "extra": dict(extra or {}),
    }
    return payload


def _publish_payload(
    payload: dict[str, Any],
    target: Path,
    *,
    previous: Path | None = None,
) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        _fsync_file(temporary)
        # Check that the exact bytes about to be published are a readable,
        # complete trusted-local checkpoint.
        verified = torch.load(temporary, map_location="cpu", weights_only=False)
        if verified.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError("checkpoint verification failed")
        digest = _sha256_file(temporary)
        if previous is not None and target.exists():
            previous.parent.mkdir(parents=True, exist_ok=True)
            os.replace(target, previous)
            target_hash = target.with_name(target.name + ".sha256")
            previous_hash = previous.with_name(previous.name + ".sha256")
            if target_hash.exists():
                previous_digest = target_hash.read_text(encoding="utf-8").split()[0]
                target_hash.unlink()
                _atomic_text(
                    previous_hash,
                    f"{previous_digest}  {previous.name}\n",
                )
            else:
                previous_hash.unlink(missing_ok=True)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
        _atomic_text(target.with_name(target.name + ".sha256"), f"{digest}  {target.name}\n")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def save_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    epoch_completed: int,
    global_step: int,
    trainer_config: object,
    scheduler: object | None = None,
    scaler: object | None = None,
    criterion: object | None = None,
    best_metrics: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Atomically write an epoch-boundary checkpoint and return its SHA-256.

    Checkpoints use Python pickle through ``torch.save`` and are intended only
    for trusted local artifacts produced by this training stack.
    """

    payload = _build_payload(
        model=model,
        optimizer=optimizer,
        epoch_completed=epoch_completed,
        global_step=global_step,
        trainer_config=trainer_config,
        scheduler=scheduler,
        scaler=scaler,
        criterion=criterion,
        best_metrics=best_metrics,
        extra=extra,
    )
    return _publish_payload(payload, Path(path).expanduser().resolve())


def _optimizer_to_parameter_devices(optimizer: Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor):
            continue
        for key, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(parameter.device)


@dataclass(frozen=True)
class ResumeState:
    """Progress and metadata returned by strict checkpoint restoration."""

    checkpoint_path: Path
    next_epoch: int
    global_step: int
    best_metrics: dict[str, float]
    extra: dict[str, Any]


def load_training_checkpoint(
    path: str | os.PathLike[str],
    *,
    model: nn.Module,
    optimizer: Optimizer,
    trainer_config: object,
    scheduler: object | None = None,
    scaler: object | None = None,
    criterion: object | None = None,
    restore_rng: bool = True,
    strict_config: bool = True,
    strict_model_metadata: bool = True,
    verify_checksum: bool = True,
) -> ResumeState:
    """Strictly restore a trusted epoch-boundary checkpoint.

    Model weights are always strict-loaded.  Config and architecture metadata
    mismatches are rejected by default.  Initialization-only fields such as
    ImageNet loading provenance do not change the architecture signature.
    """

    resolved = Path(path).expanduser().resolve()
    if verify_checksum:
        _verify_checksum(resolved)
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointIntegrityError(
            f"checkpoint cannot be deserialized: {resolved}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise CheckpointIntegrityError("checkpoint payload must be a mapping")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("checkpoint_kind") != "replite_training_epoch_boundary":
        raise ValueError("checkpoint is not an epoch-boundary training checkpoint")
    progress = payload.get("progress", {})
    if progress.get("batch_in_epoch") != 0 or progress.get("accumulation_index") != 0:
        raise ValueError("checkpoint is not at a clean epoch boundary")

    current_config = _jsonable(trainer_config)
    if strict_config and payload.get("trainer_config_sha256") != _canonical_hash(
        current_config
    ):
        raise ValueError("trainer config hash mismatch")
    if strict_model_metadata and _metadata_signature(
        payload.get("model_metadata")
    ) != _metadata_signature(_model_metadata(model)):
        raise ValueError("model metadata/architecture mismatch")

    _unwrap_model(model).load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _optimizer_to_parameter_devices(optimizer)

    for name, target in (
        ("scheduler", scheduler),
        ("scaler", scaler),
        ("criterion", criterion),
    ):
        saved_state = payload.get(name)
        load_method = getattr(target, "load_state_dict", None)
        if (saved_state is None) != (target is None or not callable(load_method)):
            raise ValueError(f"checkpoint/current {name} presence mismatch")
        if saved_state is not None:
            load_method(saved_state)

    if restore_rng:
        restore_rng_state(payload["rng"])
    next_epoch = progress.get("next_epoch")
    global_step = progress.get("global_step")
    if not isinstance(next_epoch, int) or next_epoch < 0:
        raise ValueError("invalid checkpoint next_epoch")
    if not isinstance(global_step, int) or global_step < 0:
        raise ValueError("invalid checkpoint global_step")
    return ResumeState(
        checkpoint_path=resolved,
        next_epoch=next_epoch,
        global_step=global_step,
        best_metrics=dict(payload.get("best_metrics", {})),
        extra=dict(payload.get("extra", {})),
    )


class CheckpointManager:
    """Manage ``last.pt``, ``last.prev.pt``, and named atomic checkpoints."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.last_path = self.directory / "last.pt"
        self.previous_path = self.directory / "last.prev.pt"

    def save_last(self, **checkpoint_kwargs: Any) -> str:
        """Save and rotate the previous complete ``last.pt``."""

        payload = _build_payload(**checkpoint_kwargs)
        return _publish_payload(
            payload,
            self.last_path,
            previous=self.previous_path,
        )

    def save_named(self, name: str, **checkpoint_kwargs: Any) -> Path:
        """Write a named checkpoint; ``name`` must be a plain ``.pt`` file."""

        candidate = Path(name)
        if candidate.name != name or candidate.suffix != ".pt":
            raise ValueError("checkpoint name must be a plain .pt filename")
        target = self.directory / candidate.name
        payload = _build_payload(**checkpoint_kwargs)
        _publish_payload(payload, target)
        return target

    def resume_candidates(self) -> tuple[Path, ...]:
        """Return complete local candidates in newest-to-oldest order."""

        return tuple(path for path in (self.last_path, self.previous_path) if path.exists())

    def load_latest(self, **load_kwargs: Any) -> ResumeState:
        """Restore the newest checksum-valid local checkpoint.

        Integrity/deserialization failures fall back from ``last.pt`` to
        ``last.prev.pt``. Semantic failures such as a config or architecture
        mismatch are intentionally not hidden by fallback.
        """

        candidates = self.resume_candidates()
        if not candidates:
            raise FileNotFoundError(f"no local checkpoint in {self.directory}")
        failures: list[str] = []
        for candidate in candidates:
            try:
                return load_training_checkpoint(candidate, **load_kwargs)
            except CheckpointIntegrityError as exc:
                failures.append(f"{candidate.name}: {exc}")
        raise CheckpointIntegrityError(
            "no checksum-valid local checkpoint; " + "; ".join(failures)
        )


__all__ = [
    "CheckpointIntegrityError",
    "CheckpointManager",
    "ResumeState",
    "capture_rng_state",
    "load_training_checkpoint",
    "restore_rng_state",
    "save_training_checkpoint",
]
