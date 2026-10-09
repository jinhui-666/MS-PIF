"""Configuration loading, dynamic adapters, and artifact signatures."""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from . import __version__
from .api import JsonValue, SolverAdapter


def load_object(path: str) -> Any:
    try:
        module_name, object_name = path.split(":", 1)
    except ValueError as exc:
        raise ValueError(
            "Factory must use the form 'module.submodule:function'."
        ) from exc
    module = importlib.import_module(module_name)
    try:
        return getattr(module, object_name)
    except AttributeError as exc:
        raise ImportError(f"{module_name!r} has no object {object_name!r}.") from exc


def create_adapter(
    factory_path: str,
    options: Mapping[str, Any] | None = None,
) -> SolverAdapter:
    factory: Callable[..., SolverAdapter] = load_object(factory_path)
    adapter = factory(dict(options or {}))
    manifest = adapter.manifest()
    manifest.validate()
    if adapter.problem.problem_id != manifest.problem_id:
        raise ValueError(
            f"Adapter problem {adapter.problem.problem_id!r} does not match "
            f"manifest problem {manifest.problem_id!r}."
        )
    return adapter


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_environment(device: str) -> dict[str, JsonValue]:
    environment: dict[str, JsonValue] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
    }
    try:
        import torch

        environment["torch"] = torch.__version__
        environment["cuda"] = torch.version.cuda
        if device.startswith("cuda") and torch.cuda.is_available():
            index = torch.device(device).index
            environment["device_name"] = torch.cuda.get_device_name(index)
    except ImportError:
        environment["torch"] = None
    return environment


def solver_signature(adapter: SolverAdapter, device: str) -> str:
    manifest = adapter.manifest()
    checkpoints = []
    for filename in manifest.checkpoint_files:
        path = Path(filename).expanduser().resolve()
        checkpoints.append({"sha256": file_sha256(path)})
    payload = {
        "framework": {
            "version": __version__,
            "signature_schema": "3",
            "measurement_contract": "facm-lhs-batch-executor-v2-safe-cleanup",
        },
        "manifest": {
            "api_version": manifest.api_version,
            "solver_id": manifest.solver_id,
            "problem_id": manifest.problem_id,
            "source_version": manifest.source_version,
            "inference_parameters": dict(manifest.inference_parameters),
            "stochastic": manifest.stochastic,
        },
        "checkpoints": checkpoints,
        "environment": runtime_environment(device),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data
