from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .api import Problem, SolverAdapter, validate_resource_features
from .proxy import ResourcePrediction, ResourceSurrogate


@dataclass(frozen=True)
class ScheduleBatch:
    instance_ids: tuple[str, ...]
    original_sizes: tuple[int, ...]
    batch_size: int
    max_size: int
    resource_features: dict[str, float]
    predicted_seconds: float
    memory_upper_bytes: float


@dataclass(frozen=True)
class SchedulePlan:
    solver_signature: str
    predicted_total_seconds: float
    free_memory_bytes: int
    rho: float
    batches: tuple[ScheduleBatch, ...]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def is_feasible(
    surrogate: ResourceSurrogate,
    prediction: ResourcePrediction,
    *,
    free_memory_bytes: int,
    rho: float,
) -> bool:
    return (
        1 <= prediction.batch_size <= surrogate.max_batch_size
        and prediction.memory_mean_bytes <= rho * free_memory_bytes
    )


def schedule(
    instances: Sequence[Any],
    *,
    adapter: SolverAdapter,
    surrogate: ResourceSurrogate,
    free_memory_bytes: int,
    rho: float | None = None,
    max_batch_size: int | None = None,
) -> SchedulePlan:
    if not instances:
        raise ValueError("Cannot schedule an empty workload.")
    problem = adapter.problem
    instance_ids = [problem.instance_id(instance) for instance in instances]
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("Workload contains duplicate instance IDs.")
    rho = surrogate.recommended_rho if rho is None else float(rho)
    if not 0 < rho < 1:
        raise ValueError("rho must lie strictly between zero and one.")
    limit = min(
        surrogate.max_batch_size,
        100 if max_batch_size is None else int(max_batch_size),
    )
    if limit < 1:
        raise ValueError("max_batch_size must be positive.")
    ordered = sorted(
        instances,
        key=lambda instance: (
            problem.size_of(instance),
            problem.instance_id(instance),
        ),
    )
    count = len(ordered)
    costs = [float("inf")] * (count + 1)
    previous = [-1] * (count + 1)
    chosen_batch = [0] * (count + 1)
    transition_predictions: dict[tuple[int, int], ResourcePrediction] = {}
    costs[0] = 0.0

    transition_keys = []
    transition_features = []
    for end in range(1, count + 1):
        max_size = problem.size_of(ordered[end - 1])
        for batch_size in range(1, min(end, limit) + 1):
            start = end - batch_size
            transition_keys.append((start, end))
            transition_features.append(
                validate_resource_features(
                    adapter.resource_features(ordered[start:end]),
                    batch_size=batch_size,
                    max_size=max_size,
                )
            )
    predict_many = getattr(surrogate, "predict_many", None)
    predictions = (
        predict_many(transition_features)
        if predict_many is not None
        else [surrogate.predict(features) for features in transition_features]
    )
    transition_predictions.update(zip(transition_keys, predictions))

    for end in range(1, count + 1):
        for batch_size in range(1, min(end, limit) + 1):
            start = end - batch_size
            prediction = transition_predictions.get((start, end))
            if prediction is None:
                continue
            if not is_feasible(
                surrogate,
                prediction,
                free_memory_bytes=free_memory_bytes,
                rho=rho,
            ):
                continue
            candidate = costs[start] + prediction.time_seconds
            if candidate < costs[end] - 1e-12 or (
                math.isclose(candidate, costs[end], rel_tol=1e-12, abs_tol=1e-12)
                and batch_size > chosen_batch[end]
            ):
                costs[end] = candidate
                previous[end] = start
                chosen_batch[end] = batch_size

    if not math.isfinite(costs[count]):
        raise RuntimeError("No feasible contiguous schedule exists for this workload.")

    batches = []
    end = count
    while end > 0:
        start = previous[end]
        group = ordered[start:end]
        batch_size = end - start
        max_size = problem.size_of(group[-1])
        prediction = transition_predictions[(start, end)]
        batches.append(
            ScheduleBatch(
                instance_ids=tuple(problem.instance_id(instance) for instance in group),
                original_sizes=tuple(problem.size_of(instance) for instance in group),
                batch_size=batch_size,
                max_size=max_size,
                resource_features=dict(prediction.features),
                predicted_seconds=prediction.time_seconds,
                # Retain the serialized field name for plan compatibility; the
                # constrained value is now the directly predicted memory mean.
                memory_upper_bytes=prediction.memory_mean_bytes,
            )
        )
        end = start
    batches.reverse()
    return SchedulePlan(
        solver_signature=surrogate.solver_signature,
        predicted_total_seconds=costs[count],
        free_memory_bytes=int(free_memory_bytes),
        rho=rho,
        batches=tuple(batches),
    )


def materialize_plan(
    plan: SchedulePlan,
    instances: Sequence[Any],
    problem: Problem,
) -> list[list[Any]]:
    by_id = {problem.instance_id(instance): instance for instance in instances}
    if len(by_id) != len(instances):
        raise ValueError("Workload contains duplicate instance IDs.")
    materialized = []
    for batch in plan.batches:
        try:
            materialized.append(
                [by_id[instance_id] for instance_id in batch.instance_ids]
            )
        except KeyError as exc:
            raise ValueError(
                f"Plan references unknown instance {exc.args[0]!r}."
            ) from exc
    return materialized
