"""Strict feasible-region discovery, FACM-LHS, and resource profiling."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .api import SolverAdapter, validate_resource_features
from .runtime import BatchExecutor


DEFAULT_PROFILE_POINTS = 100


@dataclass(frozen=True)
class RepeatMeasurement:
    seed: int
    seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class ProfileRecord:
    config_id: str
    solver_signature: str
    batch_size: int
    max_size: int
    features: dict[str, float]
    repeats: tuple[RepeatMeasurement, ...]

    @property
    def time_label(self) -> float:
        return float(np.mean([repeat.seconds for repeat in self.repeats]))

    @property
    def memory_label(self) -> float:
        return float(max(repeat.peak_memory_bytes for repeat in self.repeats))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["repeats"] = [asdict(repeat) for repeat in self.repeats]
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> "ProfileRecord":
        if data.get("oom") or not data.get("repeats"):
            raise ValueError(
                "Profile stores may contain successful measurements only; "
                "regenerate legacy stores containing OOM records."
            )
        features = {
            str(key): float(value)
            for key, value in data.get(
                "features",
                {"B": data["batch_size"], "N": data["max_size"]},
            ).items()
        }
        if set(features) != {"B", "N"}:
            raise ValueError(
                "Profile stores may contain only B and N features; regenerate "
                "legacy stores with model-specific GP axes."
            )
        return cls(
            config_id=str(data["config_id"]),
            solver_signature=str(data["solver_signature"]),
            batch_size=int(data["batch_size"]),
            max_size=int(data["max_size"]),
            features=features,
            repeats=tuple(
                RepeatMeasurement(
                    seed=int(item["seed"]),
                    seconds=float(item["seconds"]),
                    peak_memory_bytes=int(item["peak_memory_bytes"]),
                )
                for item in data["repeats"]
            ),
        )


class JsonlProfileStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: ProfileRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    def read(self) -> list[ProfileRecord]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as handle:
            return [
                ProfileRecord.from_dict(json.loads(line))
                for line in handle
                if line.strip()
            ]


def profiling_batches(
    adapter: SolverAdapter,
    batch_size: int,
    size: int,
    seed: int,
) -> tuple[Sequence, ...]:
    """Return the representative problem-generated profiling batch."""
    return (adapter.problem.generate(batch_size, size, seed),)


def feasibility_batches(
    adapter: SolverAdapter,
    batch_size: int,
    size: int,
    seed: int,
) -> tuple[Sequence, ...]:
    """Return solver-specific pressure batches used only by Safe(B,N)."""
    factory = getattr(adapter, "feasibility_batches", None)
    if factory is None:
        return profiling_batches(adapter, batch_size, size, seed)
    batches = tuple(factory(batch_size, size, seed))
    if not batches or any(len(batch) != batch_size for batch in batches):
        raise ValueError(
            "feasibility_batches() must return non-empty batches of the requested B."
        )
    return batches


class StrictFeasibilityProbe:
    """Boolean Safe(B,N; omega); probes are never persisted as training data."""

    def __init__(
        self,
        adapter: SolverAdapter,
        executor: BatchExecutor,
        *,
        seed: int = 0,
        confirmations: int = 1,
    ):
        if confirmations < 1:
            raise ValueError("Safe confirmations must be positive.")
        self.adapter = adapter
        self.executor = executor
        self.seed = int(seed)
        self.confirmations = int(confirmations)

    def __call__(self, batch_size: int, size: int) -> bool:
        for confirmation in range(self.confirmations):
            self.executor._clear_after_oom()
            trial_seed = _stable_seed(
                self.seed, f"safe-b{batch_size}-n{size}", confirmation
            )
            failed = False
            try:
                for instances in feasibility_batches(
                    self.adapter, batch_size, size, trial_seed
                ):
                    run = self.executor.run(
                        instances,
                        global_seed=trial_seed,
                        recover_oom=False,
                    )
                    if len(run.results) != batch_size:
                        return False
                    if any(not result.feasible for result in run.results):
                        return False
            except Exception:
                failed = True
            if failed:
                # The executor's first cleanup occurs while the exception
                # traceback can still retain CUDA tensors. Clean again after
                # leaving the except block so later binary-search probes are
                # independent of the failed allocation.
                self.executor._clear_after_oom()
                return False
        return True


@dataclass(frozen=True)
class FeasibleBoundary:
    """Conservative staircase B_cap(N) over frozen N anchors."""

    sizes: tuple[int, ...]
    batch_caps: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.sizes or len(self.sizes) != len(self.batch_caps):
            raise ValueError("A feasible boundary needs equally sized non-empty axes.")
        if tuple(sorted(set(self.sizes))) != self.sizes:
            raise ValueError("Feasible-boundary sizes must be unique and increasing.")
        if any(cap < 1 for cap in self.batch_caps):
            raise ValueError("Every feasible-boundary cap must be positive.")

    @property
    def min_size(self) -> int:
        return self.sizes[0]

    @property
    def max_size(self) -> int:
        return self.sizes[-1]

    @property
    def max_batch_size(self) -> int:
        return max(self.batch_caps)

    def cap(self, size: int) -> int:
        if not self.min_size <= size <= self.max_size:
            raise ValueError(f"N={size} lies outside the feasible boundary.")
        index = int(np.searchsorted(self.sizes, size, side="left"))
        return self.batch_caps[min(index, len(self.batch_caps) - 1)]

    def contains(self, batch_size: int, size: int) -> bool:
        return self.min_size <= size <= self.max_size and 1 <= batch_size <= self.cap(size)

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "sizes": list(self.sizes),
            "batch_caps": list(self.batch_caps),
        }


def search_feasible_boundary(
    safe: Callable[[int, int], bool],
    *,
    min_size: int,
    max_size: int,
    max_batch_size: int,
    anchor_count: int,
) -> FeasibleBoundary:
    """At log-spaced N anchors, binary-search the largest strictly safe B."""
    if min_size < 1 or max_size < min_size or max_batch_size < 1:
        raise ValueError("Invalid feasible-region search domain.")
    if anchor_count < 2 and min_size != max_size:
        raise ValueError("At least two N anchors are required.")
    sizes = _log_integer_grid(min_size, max_size, anchor_count)
    raw_caps = []
    for size in sizes:
        if not safe(1, size):
            raise RuntimeError(
                f"Frozen profiling domain is not strictly feasible at B=1, N={size}."
            )
        if safe(max_batch_size, size):
            raw_caps.append(max_batch_size)
            continue
        lower, upper = 1, max_batch_size
        while lower + 1 < upper:
            middle = (lower + upper) // 2
            if safe(middle, size):
                lower = middle
            else:
                upper = middle
        raw_caps.append(lower)

    # Enforce the monotone pressure assumption conservatively.  The cap used
    # between two anchors is the cap at the next (larger) anchor.
    conservative = list(raw_caps)
    for index in range(1, len(conservative)):
        conservative[index] = min(conservative[index], conservative[index - 1])
    return FeasibleBoundary(tuple(sizes), tuple(conservative))


@dataclass(frozen=True)
class FACMLHSDesign:
    points: tuple[tuple[int, int], ...]
    latent: tuple[tuple[float, float], ...]


class FeasibilityAwareMaximinLHS:
    """LHS candidates scored by maximin distance after feasible mapping."""

    def __init__(
        self,
        boundary: FeasibleBoundary,
        *,
        seed: int = 0,
        candidates: int = 32,
        points: int = DEFAULT_PROFILE_POINTS,
    ):
        if candidates < 1:
            raise ValueError("The number of LHS candidates must be positive.")
        if points < 1:
            raise ValueError("The number of profiling points must be positive.")
        self.boundary = boundary
        self.seed = int(seed)
        self.candidates = int(candidates)
        self.points = int(points)

    def design(self) -> FACMLHSDesign:
        points = self.points
        best: FACMLHSDesign | None = None
        best_distance = -1.0
        valid = 0
        attempts = max(self.candidates * 20, 100)
        for candidate_id in range(attempts):
            latent = self._latent(points, self.seed + candidate_id)
            mapped = tuple(self._map(u, v) for u, v in latent)
            if len(set(mapped)) != points:
                continue
            valid += 1
            distance = _minimum_distance(self._normalized_mapped(mapped))
            if distance > best_distance:
                best = FACMLHSDesign(mapped, latent)
                best_distance = distance
            if valid == self.candidates:
                break
        if best is None or valid < self.candidates:
            raise RuntimeError(
                "The integer feasible region is too small to construct the "
                f"requested {points}-point FACM-LHS without duplicates."
            )
        return best

    @staticmethod
    def _latent(count: int, seed: int) -> tuple[tuple[float, float], ...]:
        rng = np.random.default_rng(seed)
        bins = (np.arange(count) + rng.random((2, count))) / count
        u = bins[0, rng.permutation(count)]
        v = bins[1, rng.permutation(count)]
        return tuple((float(left), float(right)) for left, right in zip(u, v))

    def _map(self, u: float, v: float) -> tuple[int, int]:
        boundary = self.boundary
        if boundary.min_size == boundary.max_size:
            size = boundary.min_size
        else:
            log_size = math.log(boundary.min_size) + v * (
                math.log(boundary.max_size) - math.log(boundary.min_size)
            )
            size = int(round(math.exp(log_size)))
            size = min(boundary.max_size, max(boundary.min_size, size))
        cap = boundary.cap(size)
        batch = min(cap, 1 + int(math.floor(u * cap)))
        return batch, size

    def _normalized_mapped(
        self,
        points: Sequence[tuple[int, int]],
    ) -> tuple[tuple[float, float], ...]:
        """Represent integer feasible points in scale-free staircase coordinates."""
        boundary = self.boundary
        log_min_size = math.log(boundary.min_size)
        log_size_span = math.log(boundary.max_size) - log_min_size
        normalized = []
        for batch, size in points:
            normalized_batch = (batch - 0.5) / boundary.cap(size)
            normalized_size = (
                0.5
                if log_size_span == 0.0
                else (math.log(size) - log_min_size) / log_size_span
            )
            normalized.append((normalized_batch, normalized_size))
        return tuple(normalized)


class ProfilingRunner:
    """Measure one precomputed strict-feasible design, without replacement."""

    def __init__(
        self,
        adapter: SolverAdapter,
        executor: BatchExecutor,
        solver_signature: str,
        store: JsonlProfileStore,
        *,
        repeats: int = 5,
        warmups: int = 1,
        seed: int = 0,
    ):
        if repeats < 1 or warmups < 0:
            raise ValueError("Profiling repeats must be positive and warmups nonnegative.")
        self.adapter = adapter
        self.executor = executor
        self.signature = solver_signature
        self.store = store
        self.repeats = int(repeats)
        self.warmups = int(warmups)
        self.seed = int(seed)

    def run(self, design: FACMLHSDesign) -> list[ProfileRecord]:
        if not design.points:
            raise ValueError("The profiling design must contain at least one point.")
        if len(set(design.points)) != len(design.points):
            raise ValueError("A profiling design must contain unique points.")
        expected = set(design.points)
        records = self.store.read()
        for record in records:
            if record.solver_signature != self.signature:
                raise ValueError("Profile store contains a different solver signature.")
            if (record.batch_size, record.max_size) not in expected:
                raise ValueError("Profile store does not match the frozen FACM-LHS design.")
        completed = {(record.batch_size, record.max_size) for record in records}
        if len(completed) != len(records):
            raise ValueError("Profile store contains duplicate configurations.")

        for batch_size, max_size in design.points:
            if (batch_size, max_size) in completed:
                continue
            record = self.measure(batch_size, max_size)
            self.store.append(record)
            records.append(record)
        if len(records) != len(design.points):
            raise RuntimeError("Profiling did not complete the frozen FACM-LHS design.")
        return records

    def measure(self, batch_size: int, max_size: int) -> ProfileRecord:
        config_id = f"b{batch_size}-n{max_size}"
        instance_seed = self._seed(config_id, 0)
        batches = profiling_batches(
            self.adapter, batch_size, max_size, instance_seed
        )
        instances = list(batches[0])
        features = validate_resource_features(
            self.adapter.resource_features(instances),
            batch_size=batch_size,
            max_size=max_size,
        )
        try:
            for _ in range(self.warmups):
                for profile_batch in batches:
                    self.executor.run(
                        profile_batch,
                        global_seed=instance_seed,
                        recover_oom=False,
                    )
            repeats = []
            for repeat in range(self.repeats):
                repeat_seed = self._seed(config_id, repeat)
                runs = [
                    self.executor.run(
                        profile_batch,
                        global_seed=repeat_seed,
                        recover_oom=False,
                    )
                    for profile_batch in batches
                ]
                repeats.append(
                    RepeatMeasurement(
                        seed=repeat_seed,
                        seconds=max(run.total_seconds for run in runs),
                        peak_memory_bytes=max(
                            run.peak_incremental_memory_bytes for run in runs
                        ),
                    )
                )
        except Exception as exc:
            raise RuntimeError(
                f"Strict-feasible profiling point ({batch_size}, {max_size}) failed; "
                "abort instead of recording or replacing it."
            ) from exc
        return ProfileRecord(
            config_id=config_id,
            solver_signature=self.signature,
            batch_size=batch_size,
            max_size=max_size,
            features=features,
            repeats=tuple(repeats),
        )

    def _seed(self, config_id: str, repeat: int) -> int:
        return _stable_seed(self.seed, config_id, repeat)


def successful_records(records: Iterable[ProfileRecord]) -> list[ProfileRecord]:
    return [record for record in records if record.repeats]


def _stable_seed(seed: int, namespace: str, repeat: int) -> int:
    value = f"{seed}\0{namespace}\0{repeat}".encode()
    return (
        int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")
        & 0x7FFF_FFFF
    )


def _log_integer_grid(minimum: int, maximum: int, count: int) -> list[int]:
    if minimum == maximum:
        return [minimum]
    raw = np.geomspace(minimum, maximum, count)
    values = sorted({minimum, maximum, *(int(round(value)) for value in raw)})
    return [value for value in values if minimum <= value <= maximum]


def _minimum_distance(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return float("inf")
    values = np.asarray(points, dtype=float)
    deltas = values[:, None, :] - values[None, :, :]
    distances = np.sqrt(np.square(deltas).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())
