from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..api import Evaluation


_COORDINATE_LIMIT = 1000
_XL_ROUTE_SIZE_INTERVALS = (
    (3.0, 5.0),
    (5.0, 8.0),
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 25.0),
    (25.0, 50.0),
    (50.0, 200.0),
)


@dataclass(frozen=True)
class CVRPInstance:
    instance_id: str
    coords: np.ndarray
    demands: np.ndarray
    capacity: float
    best_known: float | None = None
    round_distances: bool = True
    edge_weights: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def customer_count(self) -> int:
        return int(self.coords.shape[0]) - 1

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("CVRP coordinates must have shape [N+1, 2].")
        if self.demands.ndim != 1 or len(self.demands) != len(self.coords):
            raise ValueError("CVRP demands must have shape [N+1].")
        if len(self.coords) < 2:
            raise ValueError("A CVRP instance must contain at least one customer.")
        if self.capacity <= 0:
            raise ValueError("CVRP capacity must be positive.")
        if abs(float(self.demands[0])) > 1e-8:
            raise ValueError("Canonical CVRP depot demand must be zero.")
        if np.any(self.demands < 0):
            raise ValueError("CVRP demands must be non-negative.")
        if self.edge_weights is not None:
            expected = (len(self.coords), len(self.coords))
            if self.edge_weights.shape != expected:
                raise ValueError(
                    f"CVRP edge weights must have shape {expected}."
                )
            if np.any(self.edge_weights < 0):
                raise ValueError("CVRP edge weights must be non-negative.")


@dataclass(frozen=True)
class CVRPSolution:
    """Routes use canonical customer indices 1..N; depot 0 is implicit."""

    routes: tuple[tuple[int, ...], ...]


class CVRPProblem:
    problem_id = "cvrp"

    def load(self, source: str) -> Sequence[CVRPInstance]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        paths = sorted(path.glob("*.vrp")) if path.is_dir() else [path]
        return [self._load_vrplib(item) for item in paths]

    def _load_vrplib(self, path: Path) -> CVRPInstance:
        try:
            import vrplib
        except ImportError as exc:
            raise RuntimeError(
                "Loading VRPLIB files requires the 'vrplib' package."
            ) from exc

        raw = vrplib.read_instance(path)
        coords = np.asarray(raw["node_coord"], dtype=np.float32)
        demands = np.asarray(raw["demand"], dtype=np.float32)
        depot_raw = np.asarray(raw.get("depot", [0])).reshape(-1)
        depot = int(depot_raw[0]) if depot_raw.size else 0
        if depot < 0 or depot >= len(coords):
            raise ValueError(f"{path}: invalid depot index {depot}.")
        if depot != 0:
            order = [depot] + [idx for idx in range(len(coords)) if idx != depot]
            coords = coords[order]
            demands = demands[order]
        else:
            order = list(range(len(coords)))
        demands = demands.copy()
        demands[0] = 0
        solution_path = path.with_suffix(".sol")
        best_known = None
        if solution_path.exists():
            solution = vrplib.read_solution(solution_path)
            if solution.get("cost") is not None:
                best_known = float(solution["cost"])
        edge_weight_type = str(raw.get("edge_weight_type", "EUC_2D")).upper()
        edge_weights = None
        if edge_weight_type == "EXPLICIT":
            edge_weights = np.asarray(raw["edge_weight"], dtype=np.float64)
            edge_weights = edge_weights[np.ix_(order, order)]
        elif edge_weight_type != "EUC_2D":
            raise ValueError(
                f"{path}: unsupported EDGE_WEIGHT_TYPE {edge_weight_type!r}; "
                "the canonical evaluator supports EUC_2D and EXPLICIT."
            )
        return CVRPInstance(
            instance_id=path.stem,
            coords=coords,
            demands=demands,
            capacity=float(raw["capacity"]),
            best_known=best_known,
            round_distances=edge_weights is None,
            edge_weights=edge_weights,
            metadata={
                "source": str(path),
                "edge_weight_type": edge_weight_type,
                "edge_weight_format": raw.get("edge_weight_format"),
            },
        )

    def generate(
        self,
        batch_size: int,
        size: int,
        seed: int,
    ) -> Sequence[CVRPInstance]:
        if batch_size < 1 or size < 1:
            raise ValueError("batch_size and size must be positive.")
        rng = np.random.default_rng(seed)
        instances = []
        for index in range(batch_size):
            coords = self._uniform_coords(rng, size + 1)
            family, route_size_class, route_length = self._average_route_size(
                rng, size
            )
            demand_type = int(rng.integers(1, 8))
            customer_demands = self._generate_demands(
                rng,
                coords[1:],
                demand_type=demand_type,
                route_length=route_length,
            )
            demand_sum = int(customer_demands.sum(dtype=np.int64))
            max_demand = int(customer_demands.max())
            if demand_type == 1:
                capacity = math.floor(route_length)
            else:
                capacity = max(
                    max_demand,
                    math.ceil(route_length * demand_sum / size),
                )
            demands = np.concatenate(
                [np.zeros(1, dtype=np.float32), customer_demands.astype(np.float32)]
            )
            instances.append(
                CVRPInstance(
                    instance_id=(
                        f"profile-{family}-b{batch_size}-n{size}-s{seed}-{index}"
                    ),
                    coords=coords,
                    demands=demands,
                    capacity=float(capacity),
                    round_distances=True,
                    metadata={
                        "generated": True,
                        "generator": "cvrplib-uniform-x-xl-v1",
                        "generator_family": family,
                        "spatial_distribution": "uniform",
                        "demand_type": demand_type,
                        "target_average_route_size": route_length,
                        "route_size_class": route_size_class,
                        "seed": seed,
                    },
                )
            )
        return instances

    @staticmethod
    def _uniform_coords(
        rng: np.random.Generator,
        count: int,
    ) -> np.ndarray:
        side = _COORDINATE_LIMIT + 1
        available = side * side
        if count > available:
            raise ValueError(
                f"Cannot draw {count} unique CVRPLIB coordinates from "
                f"the {side}x{side} grid."
            )
        flat = rng.choice(available, size=count, replace=False)
        return np.column_stack((flat % side, flat // side)).astype(np.float32)

    @staticmethod
    def _average_route_size(
        rng: np.random.Generator,
        size: int,
    ) -> tuple[str, int | None, float]:
        if size < 1000:
            return "x", None, float(rng.triangular(3.0, 6.0, 25.0))
        route_size_class = int(rng.integers(0, len(_XL_ROUTE_SIZE_INTERVALS)))
        lower, upper = _XL_ROUTE_SIZE_INTERVALS[route_size_class]
        return "xl", route_size_class + 1, float(rng.uniform(lower, upper))

    @staticmethod
    def _generate_demands(
        rng: np.random.Generator,
        customer_coords: np.ndarray,
        *,
        demand_type: int,
        route_length: float,
    ) -> np.ndarray:
        size = len(customer_coords)
        if demand_type == 1:
            return np.ones(size, dtype=np.int64)
        if demand_type == 2:
            return rng.integers(1, 11, size=size, dtype=np.int64)
        if demand_type == 3:
            return rng.integers(5, 11, size=size, dtype=np.int64)
        if demand_type == 4:
            return rng.integers(1, 101, size=size, dtype=np.int64)
        if demand_type == 5:
            return rng.integers(50, 101, size=size, dtype=np.int64)
        if demand_type == 6:
            same_half = (
                (customer_coords[:, 0] < _COORDINATE_LIMIT / 2)
                == (customer_coords[:, 1] < _COORDINATE_LIMIT / 2)
            )
            demands = rng.integers(1, 51, size=size, dtype=np.int64)
            demands[same_half] = rng.integers(
                51,
                101,
                size=int(same_half.sum()),
                dtype=np.int64,
            )
            return demands
        if demand_type == 7:
            demands = rng.integers(1, 11, size=size, dtype=np.int64)
            large_count = min(
                size,
                max(1, int(round(1.5 * size / route_length))),
            )
            large_indices = rng.choice(size, size=large_count, replace=False)
            demands[large_indices] = rng.integers(
                50,
                101,
                size=large_count,
                dtype=np.int64,
            )
            return demands
        raise ValueError(f"Unsupported CVRPLIB demand type {demand_type}.")

    def instance_id(self, instance: CVRPInstance) -> str:
        return instance.instance_id

    def size_of(self, instance: CVRPInstance) -> int:
        return instance.customer_count

    def audit(self, instance: CVRPInstance) -> Mapping[str, float]:
        return {
            "nodes": float(instance.customer_count + 1),
            "customers": float(instance.customer_count),
        }

    def evaluate(self, instance: CVRPInstance, decision: Any) -> Evaluation:
        if not isinstance(decision, CVRPSolution):
            return Evaluation(float("inf"), False, "decision_is_not_cvrp_solution")

        expected = list(range(1, instance.customer_count + 1))
        visited = [node for route in decision.routes for node in route]
        reasons: list[str] = []
        if sorted(visited) != expected:
            reasons.append("customers_not_visited_exactly_once")

        total = 0.0
        max_load = 0.0
        for route_index, route in enumerate(decision.routes):
            load = 0.0
            previous = 0
            for node in route:
                if node < 1 or node > instance.customer_count:
                    reasons.append(f"invalid_node_{node}")
                    continue
                load += float(instance.demands[node])
                total += self._distance(instance, previous, node)
                previous = node
            total += self._distance(instance, previous, 0)
            max_load = max(max_load, load)
            if load > instance.capacity + 1e-6:
                reasons.append(f"capacity_exceeded_route_{route_index}")

        diagnostics = {
            "route_count": len(decision.routes),
            "max_route_load": max_load,
            "capacity": instance.capacity,
            "best_known": instance.best_known,
        }
        return Evaluation(
            objective=float(total),
            feasible=not reasons,
            reason="OK" if not reasons else ";".join(reasons),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _distance(instance: CVRPInstance, left: int, right: int) -> float:
        if instance.edge_weights is not None:
            return float(instance.edge_weights[left, right])
        distance = float(np.linalg.norm(instance.coords[left] - instance.coords[right]))
        if instance.round_distances:
            return float(math.floor(distance + 0.5))
        return distance
