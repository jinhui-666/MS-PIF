from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from ..api import Evaluation


DistanceMode = Literal["euclidean", "euc_2d", "ceil_2d"]


_TSPLIB_BEST_KNOWN: dict[str, float] = {
    "wi29": 27603,
    "dj38": 6656,
    "eil51": 426,
    "berlin52": 7542,
    "st70": 675,
    "eil76": 538,
    "pr76": 108159,
    "rat99": 1211,
    "kroA100": 21282,
    "kroB100": 22141,
    "kroC100": 20749,
    "kroD100": 21294,
    "kroE100": 22068,
    "rd100": 7910,
    "eil101": 629,
    "lin105": 14379,
    "pr107": 44303,
    "pr124": 59030,
    "bier127": 118282,
    "ch130": 6110,
    "pr136": 96772,
    "pr144": 58537,
    "ch150": 6528,
    "kroA150": 26524,
    "kroB150": 26130,
    "pr152": 73682,
    "u159": 42080,
    "qa194": 9352,
    "rat195": 2323,
    "d198": 15780,
    "kroA200": 29368,
    "kroB200": 29437,
    "ts225": 126643,
    "tsp225": 3916,
    "pr226": 80369,
    "gil262": 2378,
    "pr264": 49135,
    "a280": 2579,
    "pr299": 48191,
    "lin318": 42029,
    "rd400": 15281,
    "fl417": 11861,
    "pr439": 107217,
    "pcb442": 50778,
    "d493": 35002,
    "u574": 36905,
    "rat575": 6773,
    "p654": 34643,
    "d657": 48912,
    "u724": 41910,
    "uy734": 79114,
    "rat783": 8806,
    "zi929": 95345,
    "lu980": 11340,
    "dsj1000": 18659688,
    "pr1002": 259045,
    "u1060": 224094,
    "vm1084": 239297,
    "pcb1173": 56892,
    "d1291": 50801,
    "rl1304": 252948,
    "rl1323": 270199,
    "nrw1379": 56638,
    "fl1400": 20127,
    "u1432": 152970,
    "fl1577": 22249,
    "rw1621": 26051,
    "d1655": 62128,
    "vm1748": 336556,
    "u1817": 57201,
    "rl1889": 316536,
    "mu1979": 86891,
    "d2103": 80450,
    "u2152": 64253,
    "u2319": 234256,
    "pr2392": 378032,
    "pcb3038": 137694,
    "nu3496": 96132,
    "fl3795": 28772,
    "fnl4461": 182566,
    "ca4663": 1290319,
    "rl5915": 565530,
    "rl5934": 556045,
    "tz6117": 394718,
    "eg7146": 172386,
    "pla7397": 23260728,
    "ym7663": 238314,
    "pm8079": 114855,
    "ei8246": 206171,
    "ar9152": 837479,
    "ja9847": 491924,
    "gr9882": 300899,
    "kz9976": 1061881,
    "fi10639": 520527,
    "rl11849": 923288,
    "usa13509": 19982859,
    "brd14051": 469385,
    "mo14185": 427377,
    "ho14473": 177092,
    "d15112": 1573084,
    "it16862": 557315,
    "d18512": 645238,
    "vm22775": 569288,
    "sw24978": 855597,
    "bm33708": 959289,
    "pla33810": 66048945,
}

_TSPLIB_BEST_KNOWN_NOT_PROVEN = {"bm33708"}


@dataclass(frozen=True)
class TSPInstance:
    instance_id: str
    coords: np.ndarray
    distance_mode: DistanceMode = "euclidean"
    best_known: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("TSP coordinates must have shape [N, 2].")
        if len(self.coords) < 1:
            raise ValueError("A TSP instance must contain at least one node.")
        if not np.all(np.isfinite(self.coords)):
            raise ValueError("TSP coordinates must be finite.")
        if self.distance_mode not in {"euclidean", "euc_2d", "ceil_2d"}:
            raise ValueError(f"Unsupported TSP distance mode {self.distance_mode!r}.")

    @property
    def node_count(self) -> int:
        return int(self.coords.shape[0])


@dataclass(frozen=True)
class TSPSolution:
    tour: tuple[int, ...]


class TSPProblem:
    problem_id = "tsp"

    _DISTANCE_MODES: Mapping[str, DistanceMode] = {
        "EUC_2D": "euc_2d",
        "CEIL_2D": "ceil_2d",
    }

    def load(self, source: str) -> Sequence[TSPInstance]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file() and path.suffix.lower() != ".tsp":
            raise ValueError(f"{path}: expected a .tsp file.")
        paths = sorted(path.glob("*.tsp")) if path.is_dir() else [path]
        return [self._load_tsplib(item) for item in paths]

    def _load_tsplib(self, path: Path) -> TSPInstance:
        try:
            import tsplib95

            problem = tsplib95.load(str(path))
        except Exception as exc:
            raise ValueError(f"{path}: failed to parse TSPLIB instance: {exc}") from exc

        edge_weight_type = str(problem.edge_weight_type or "").upper()
        if edge_weight_type not in self._DISTANCE_MODES:
            raise ValueError(
                f"{path}: unsupported EDGE_WEIGHT_TYPE {edge_weight_type!r}; "
                "expected EUC_2D or CEIL_2D."
            )

        raw_coords = problem.node_coords
        if not raw_coords:
            raise ValueError(f"{path}: complete two-dimensional coordinates are required.")
        source_node_ids = tuple(sorted(raw_coords))
        try:
            coords = np.asarray(
                [raw_coords[node] for node in source_node_ids],
                dtype=np.float32,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid coordinates: {exc}") from exc
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"{path}: coordinates must have shape [DIMENSION, 2], "
                f"got {coords.shape}."
            )
        dimension = int(problem.dimension)
        if len(coords) != dimension:
            raise ValueError(
                f"{path}: DIMENSION is {dimension}, but found "
                f"{len(coords)} coordinate rows."
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"{path}: coordinates must be finite.")

        best_known = _TSPLIB_BEST_KNOWN.get(path.stem)
        reference_status = None
        if best_known is not None:
            reference_status = (
                "best_known_not_proven"
                if path.stem in _TSPLIB_BEST_KNOWN_NOT_PROVEN
                else "proven_optimal"
            )

        return TSPInstance(
            instance_id=path.stem,
            coords=coords,
            distance_mode=self._DISTANCE_MODES[edge_weight_type],
            best_known=best_known,
            metadata={
                "source": str(path),
                "source_node_ids": source_node_ids,
                "edge_weight_type": edge_weight_type,
                "reference_status": reference_status,
            },
        )

    def generate(
        self,
        batch_size: int,
        size: int,
        seed: int,
    ) -> Sequence[TSPInstance]:
        if batch_size < 1 or size < 1:
            raise ValueError("batch_size and size must be positive.")
        rng = np.random.default_rng(seed)
        return [
            TSPInstance(
                instance_id=f"profile-b{batch_size}-n{size}-s{seed}-{index}",
                coords=rng.random((size, 2), dtype=np.float32),
                distance_mode="euclidean",
                metadata={
                    "generated": True,
                    "generator": "uniform-tsp-v1",
                    "seed": seed,
                },
            )
            for index in range(batch_size)
        ]

    def instance_id(self, instance: TSPInstance) -> str:
        return instance.instance_id

    def size_of(self, instance: TSPInstance) -> int:
        return instance.node_count

    def audit(self, instance: TSPInstance) -> Mapping[str, float]:
        return {"nodes": float(instance.node_count)}

    def evaluate(self, instance: TSPInstance, decision: Any) -> Evaluation:
        if not isinstance(decision, TSPSolution):
            return Evaluation(float("inf"), False, "decision_is_not_tsp_solution")
        tour = tuple(int(node) for node in decision.tour)
        if sorted(tour) != list(range(instance.node_count)):
            return Evaluation(
                float("inf"),
                False,
                "nodes_not_visited_exactly_once",
                diagnostics={"node_count": instance.node_count},
            )

        total = 0.0
        for index, left in enumerate(tour):
            right = tour[(index + 1) % len(tour)]
            total += self._distance(instance, left, right)
        gap_percent = None
        if instance.best_known is not None:
            gap_percent = (
                100.0 * (total - instance.best_known) / instance.best_known
            )
        return Evaluation(
            objective=float(total),
            feasible=True,
            reason="OK",
            diagnostics={
                "node_count": instance.node_count,
                "best_known": instance.best_known,
                "reference_status": instance.metadata.get("reference_status"),
                "gap_percent": gap_percent,
            },
        )

    @staticmethod
    def _distance(instance: TSPInstance, left: int, right: int) -> float:
        distance = float(np.linalg.norm(instance.coords[left] - instance.coords[right]))
        if instance.distance_mode == "euc_2d":
            return float(math.floor(distance + 0.5))
        if instance.distance_mode == "ceil_2d":
            return float(math.ceil(distance))
        return distance
