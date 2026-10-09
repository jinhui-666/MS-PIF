from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True)
class SolverManifest:
    """Everything that fixes solver behavior and resource cost."""

    api_version: str
    solver_id: str
    problem_id: str
    source_version: str
    checkpoint_files: tuple[str, ...]
    inference_parameters: Mapping[str, JsonValue]
    stochastic: bool = False

    def validate(self) -> None:
        if self.api_version != "1":
            raise ValueError(f"Unsupported adapter API version: {self.api_version}")
        if not self.solver_id or not self.problem_id:
            raise ValueError("solver_id and problem_id must be non-empty.")
        json.dumps(dict(self.inference_parameters), sort_keys=True)


@dataclass(frozen=True)
class RunContext:
    """Stable seed namespace passed to a solver adapter."""

    global_seed: int

    def seed_for(
        self,
        instance_id: str,
        candidate: int = 0,
        stage: str = "solve",
    ) -> int:
        payload = f"{self.global_seed}\0{instance_id}\0{candidate}\0{stage}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "big") & 0x7FFF_FFFF


@dataclass(frozen=True)
class ModelSolution:
    """One decoded model decision, before problem-level evaluation."""

    instance_id: str
    decision: Any
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Evaluation:
    objective: float
    feasible: bool
    reason: str = "OK"
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveResult:
    instance_id: str
    decision: Any
    objective: float
    feasible: bool
    reason: str
    diagnostics: Mapping[str, JsonValue] = field(default_factory=dict)


@runtime_checkable
class Problem(Protocol):
    problem_id: str

    def load(self, source: str) -> Sequence[Any]:
        """Load deployment instances without moving them to an accelerator."""

    def generate(
        self,
        batch_size: int,
        size: int,
        seed: int,
    ) -> Sequence[Any]:
        """Generate a reproducible profiling batch."""

    def instance_id(self, instance: Any) -> str:
        """Return a stable, globally unique ID."""

    def size_of(self, instance: Any) -> int:
        """Return the problem size N used by the current method."""

    def evaluate(self, instance: Any, decision: Any) -> Evaluation:
        """Evaluate objective and feasibility independently of the model."""

    def audit(self, instance: Any) -> Mapping[str, float]:
        """Return optional cost-compatibility statistics."""


@runtime_checkable
class SolverAdapter(Protocol):
    problem: Problem

    def manifest(self) -> SolverManifest:
        """Return the frozen solver identity before model loading."""

    def load(self, device: str) -> None:
        """Load weights once; model loading is outside batch timing."""

    def resource_features(
        self,
        instances: Sequence[Any],
    ) -> Mapping[str, float]:
        """Return exactly the two scheduling coordinates ``B`` and ``N``.

        Model-specific growth terms are declared as B/N powers in configuration;
        they do not add sampled or GP input axes.
        """

    def solve_batch(
        self,
        instances: Sequence[Any],
        context: RunContext,
    ) -> Sequence[ModelSolution]:
        """Solve exactly one CS-PIF instance batch in input order."""


def validate_resource_features(
    features: Mapping[str, float],
    *,
    batch_size: int,
    max_size: int,
) -> dict[str, float]:
    """Validate the common feature contract without model-specific branches."""
    values = {str(key): float(value) for key, value in features.items()}
    if set(values) != {"B", "N"}:
        raise ValueError("Resource features must contain exactly B and N.")
    expected = {"B": float(batch_size), "N": float(max_size)}
    for name, value in expected.items():
        if name not in values or not math.isclose(values[name], value):
            raise ValueError(
                f"Adapter resource feature {name!r} must equal {value:g}."
            )
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("Adapter resource features must be finite and positive.")
    return values
