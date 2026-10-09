"""Configured positive B/N backbones for the resource surrogate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .api import JsonValue


FeatureColumns = Mapping[str, np.ndarray]


@runtime_checkable
class BackboneBasis(Protocol):
    resource: str
    coefficient_names: tuple[str, ...]

    def component_matrix(self, features: FeatureColumns) -> np.ndarray:
        """Return [rows, coefficients], including the intercept."""

    def evaluate(
        self, features: FeatureColumns, coefficients: np.ndarray
    ) -> np.ndarray:
        """Evaluate the positive configured backbone."""

    def description(self) -> Mapping[str, JsonValue]:
        """Return the configured B/N terms."""


@dataclass(frozen=True)
class MonomialBasis:
    """A positive sum of configured monomials B^p N^q."""

    resource: str
    powers: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.resource not in {"time", "memory"}:
            raise ValueError("Backbone resource must be time or memory.")
        _validate_powers(self.powers, f"{self.resource}_powers")

    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return ("intercept", *(_term_name(p, q) for p, q in self.powers))

    def component_matrix(self, features: FeatureColumns) -> np.ndarray:
        columns = _feature_columns(features)
        batch, size = columns["B"], columns["N"]
        components = [np.ones(len(batch))]
        components.extend(batch**p * size**q for p, q in self.powers)
        matrix = np.column_stack(components)
        if np.any(matrix < 0) or not np.all(np.isfinite(matrix)):
            raise ValueError("Backbone values must be finite and nonnegative.")
        return matrix

    def evaluate(
        self, features: FeatureColumns, coefficients: np.ndarray
    ) -> np.ndarray:
        return self.component_matrix(features) @ np.asarray(
            coefficients, dtype=float
        )

    def description(self) -> Mapping[str, JsonValue]:
        return {
            "name": "configured-bn-monomials-v1",
            "resource": self.resource,
            "coefficient_names": list(self.coefficient_names),
            "terms": [
                {
                    "name": _term_name(p, q),
                    "b_power": p,
                    "n_power": q,
                    "expression": _term_expression(p, q),
                }
                for p, q in self.powers
            ],
            "aggregation": "sum",
        }


@dataclass(frozen=True)
class BackbonePair:
    time: BackboneBasis
    memory: BackboneBasis
    gp_inputs: tuple[str, ...] = ("B", "N")

    def description(self) -> dict[str, JsonValue]:
        return {
            "gp_inputs": list(self.gp_inputs),
            "time": dict(self.time.description()),
            "memory": dict(self.memory.description()),
        }

    def signature(self) -> str:
        payload = json.dumps(
            self.description(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def default_backbone_pair() -> BackbonePair:
    """Use BN for both resources when an integration omits a backbone."""
    return BackbonePair(
        time=MonomialBasis("time", ((1, 1),)),
        memory=MonomialBasis("memory", ((1, 1),)),
    )


def create_backbone_pair(config: Mapping[str, Any] | None) -> BackbonePair:
    """Build model-specific B/N backbones directly from configuration."""
    if config is None:
        return default_backbone_pair()
    unknown = set(config) - {"time_powers", "memory_powers"}
    if unknown:
        raise ValueError(
            "surrogate.backbone accepts only time_powers and memory_powers."
        )
    missing = {"time_powers", "memory_powers"} - set(config)
    if missing:
        raise ValueError(
            "surrogate.backbone requires time_powers and memory_powers."
        )
    return BackbonePair(
        time=MonomialBasis(
            "time", _parse_powers(config["time_powers"], "time_powers")
        ),
        memory=MonomialBasis(
            "memory", _parse_powers(config["memory_powers"], "memory_powers")
        ),
    )


def fitted_backbone_description(
    basis: BackboneBasis,
    coefficients: np.ndarray | None,
) -> dict[str, JsonValue]:
    description = json.loads(json.dumps(dict(basis.description()), allow_nan=False))
    if coefficients is None:
        return description
    values = np.asarray(coefficients, dtype=float).reshape(-1)
    names = list(description["coefficient_names"])
    if len(values) != len(names):
        raise ValueError("Backbone coefficient count does not match its terms.")
    description["fitted_coefficients"] = {
        name: float(value) for name, value in zip(names, values)
    }
    expressions = ["1", *[term["expression"] for term in description["terms"]]]
    description["fitted_expression"] = " + ".join(
        f"{float(value):.12g} * ({formula})"
        for value, formula in zip(values, expressions)
    )
    return description


def _parse_powers(value: Any, name: str) -> tuple[tuple[int, int], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"surrogate.backbone.{name} must be a sequence.")
    powers: list[tuple[int, int]] = []
    for item in value:
        if (
            isinstance(item, (str, bytes))
            or not isinstance(item, Sequence)
            or len(item) != 2
        ):
            raise ValueError(f"Each {name} entry must be [B power, N power].")
        p, q = item
        if isinstance(p, bool) or isinstance(q, bool):
            raise ValueError(f"{name} powers must be integers.")
        if not isinstance(p, int) or not isinstance(q, int):
            raise ValueError(f"{name} powers must be integers.")
        powers.append((p, q))
    result = tuple(powers)
    _validate_powers(result, name)
    return result


def _validate_powers(powers: Sequence[tuple[int, int]], name: str) -> None:
    if not powers:
        raise ValueError(f"{name} must contain at least one term.")
    if len(set(powers)) != len(powers):
        raise ValueError(f"{name} contains duplicate terms.")
    if any(p < 0 or q < 0 or p + q == 0 or p > 8 or q > 8 for p, q in powers):
        raise ValueError(
            f"{name} powers must be in [0, 8] and cannot both be zero."
        )


def _term_name(p: int, q: int) -> str:
    return _term_expression(p, q).replace(" ", "")


def _term_expression(p: int, q: int) -> str:
    factors = []
    if p:
        factors.append("B" if p == 1 else f"B^{p}")
    if q:
        factors.append("N" if q == 1 else f"N^{q}")
    return " * ".join(factors)


def _feature_columns(features: FeatureColumns) -> dict[str, np.ndarray]:
    if set(features) != {"B", "N"}:
        raise ValueError("Backbone and GP inputs must contain exactly B and N.")
    columns = {
        str(name): np.asarray(value, dtype=float).reshape(-1)
        for name, value in features.items()
    }
    if len({len(value) for value in columns.values()}) != 1:
        raise ValueError("Resource feature columns must have equal lengths.")
    if any(
        np.any(value <= 0) or not np.all(np.isfinite(value))
        for value in columns.values()
    ):
        raise ValueError("B and N must be finite and positive.")
    return columns
