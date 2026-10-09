"""Production batch execution, OOM recovery, and adapter conformance."""

from __future__ import annotations

import gc
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .api import RunContext, SolveResult, SolverAdapter


@dataclass(frozen=True)
class OOMEvent:
    instance_ids: tuple[str, ...]
    batch_size: int
    max_size: int


@dataclass(frozen=True)
class BatchRun:
    results: tuple[SolveResult, ...]
    total_seconds: float
    peak_incremental_memory_bytes: int
    oom_events: tuple[OOMEvent, ...] = field(default_factory=tuple)


class BatchExecutor:
    """The only timed entry point used by profiling and deployment."""

    def __init__(self, adapter: SolverAdapter, device: str):
        self.adapter = adapter
        self.device = device
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self.adapter.load(self.device)
        self._synchronize()
        self._loaded = True

    def free_memory_bytes(self) -> int:
        torch = self._torch()
        if torch is None or not self.device.startswith("cuda"):
            return 2**63 - 1
        free, _ = torch.cuda.mem_get_info(torch.device(self.device))
        return int(free)

    def run(
        self,
        instances: Sequence[Any],
        *,
        global_seed: int,
        recover_oom: bool,
    ) -> BatchRun:
        if not self._loaded:
            raise RuntimeError("Call BatchExecutor.load() before running batches.")
        if not instances:
            raise ValueError("Cannot execute an empty batch.")
        started = time.perf_counter()
        results, peak, events = self._run_recursive(
            list(instances),
            RunContext(global_seed=global_seed),
            recover_oom=recover_oom,
        )
        return BatchRun(
            results=tuple(results),
            total_seconds=time.perf_counter() - started,
            peak_incremental_memory_bytes=peak,
            oom_events=tuple(events),
        )

    def _run_recursive(
        self,
        instances: list[Any],
        context: RunContext,
        *,
        recover_oom: bool,
    ) -> tuple[list[SolveResult], int, list[OOMEvent]]:
        try:
            results, peak = self._attempt(instances, context)
            return results, peak, []
        except Exception as exc:
            if not self._is_oom(exc):
                raise
            event = OOMEvent(
                instance_ids=tuple(
                    self.adapter.problem.instance_id(instance) for instance in instances
                ),
                batch_size=len(instances),
                max_size=max(
                    self.adapter.problem.size_of(instance) for instance in instances
                ),
            )
            self._clear_after_oom()
            if not recover_oom or len(instances) == 1:
                raise
            middle = len(instances) // 2
            left_results, left_peak, left_events = self._run_recursive(
                instances[:middle], context, recover_oom=True
            )
            right_results, right_peak, right_events = self._run_recursive(
                instances[middle:], context, recover_oom=True
            )
            return (
                left_results + right_results,
                max(left_peak, right_peak),
                [event] + left_events + right_events,
            )

    def _attempt(
        self,
        instances: Sequence[Any],
        context: RunContext,
    ) -> tuple[list[SolveResult], int]:
        torch = self._torch()
        base_memory = 0
        if torch is not None and self.device.startswith("cuda"):
            device = torch.device(self.device)
            self._synchronize()
            base_memory = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)

        model_solutions = list(self.adapter.solve_batch(instances, context))
        if len(model_solutions) != len(instances):
            raise ValueError(
                f"Adapter returned {len(model_solutions)} solutions for "
                f"{len(instances)} instances."
            )

        results = []
        for instance, model_solution in zip(instances, model_solutions):
            expected_id = self.adapter.problem.instance_id(instance)
            if model_solution.instance_id != expected_id:
                raise ValueError(
                    f"Adapter result order mismatch: expected {expected_id!r}, "
                    f"got {model_solution.instance_id!r}."
                )
            evaluation = self.adapter.problem.evaluate(
                instance, model_solution.decision
            )
            diagnostics = dict(model_solution.diagnostics)
            diagnostics.update(evaluation.diagnostics)
            results.append(
                SolveResult(
                    instance_id=expected_id,
                    decision=model_solution.decision,
                    objective=evaluation.objective,
                    feasible=evaluation.feasible,
                    reason=evaluation.reason,
                    diagnostics=diagnostics,
                )
            )

        self._synchronize()
        peak = 0
        if torch is not None and self.device.startswith("cuda"):
            peak = max(
                0,
                int(torch.cuda.max_memory_allocated(torch.device(self.device)))
                - base_memory,
            )
        return results, peak

    def _synchronize(self) -> None:
        torch = self._torch()
        if torch is not None and self.device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(self.device))

    def _clear_after_oom(self) -> None:
        gc.collect()
        torch = self._torch()
        if torch is not None and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
            self._synchronize()

    @staticmethod
    def _torch():
        try:
            import torch

            return torch
        except ImportError:
            return None

    @classmethod
    def _is_oom(cls, exc: Exception) -> bool:
        torch = cls._torch()
        # Older supported PyTorch builds expose CUDA OOM only as RuntimeError.
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ()) if torch is not None else ()
        if isinstance(exc, oom_type):
            return True
        text = str(exc).lower()
        return "out of memory" in text and ("cuda" in text or "gpu" in text)


@dataclass(frozen=True)
class ConformanceReport:
    solver_id: str
    checks: tuple[str, ...]


def check_adapter(
    adapter: SolverAdapter,
    executor: BatchExecutor,
    *,
    small_size: int,
    large_size: int,
    seed: int = 0,
    objective_tolerance: float = 1e-5,
) -> ConformanceReport:
    """Exercise the public adapter contract through the production executor."""
    manifest = adapter.manifest()
    manifest.validate()
    frozen_manifest = _manifest_json(manifest)
    if small_size < 1 or large_size <= small_size:
        raise ValueError("Conformance sizes must satisfy 1 <= small < large.")
    executor.load()
    checks: list[str] = []

    single = list(adapter.problem.generate(1, small_size, seed))
    single_run = executor.run(single, global_seed=seed, recover_oom=False)
    if len(single_run.results) != 1 or not single_run.results[0].feasible:
        raise AssertionError("B=1 execution did not return one feasible solution.")
    checks.append("single_instance")

    mixed = [
        adapter.problem.generate(1, small_size, seed + 1)[0],
        adapter.problem.generate(1, large_size, seed + 2)[0],
    ]
    features = dict(adapter.resource_features(mixed))
    if (
        not math.isclose(float(features.get("B", -1)), 2.0)
        or not math.isclose(float(features.get("N", -1)), float(large_size))
        or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in features.values()
        )
    ):
        raise AssertionError(
            "resource_features() must return positive B/N batch features."
        )
    if features != dict(adapter.resource_features(mixed)):
        raise AssertionError("resource_features() is not deterministic.")
    checks.append("resource_features")

    batch_run = executor.run(mixed, global_seed=seed, recover_oom=False)
    if len(batch_run.results) != 2:
        raise AssertionError("Mixed-size execution returned the wrong result count.")
    if any(not result.feasible for result in batch_run.results):
        raise AssertionError("Mixed-size execution produced an infeasible solution.")
    checks.append("mixed_size")

    split_results = []
    for instance in mixed:
        split_results.extend(
            executor.run([instance], global_seed=seed, recover_oom=False).results
        )
    for batched, split in zip(batch_run.results, split_results):
        scale = max(1.0, abs(split.objective))
        if abs(batched.objective - split.objective) > objective_tolerance * scale:
            raise AssertionError(
                f"Batch/split objective mismatch for {batched.instance_id}: "
                f"{batched.objective} vs {split.objective}."
            )
    checks.append("split_equivalence")

    expected_ids = [adapter.problem.instance_id(instance) for instance in mixed]
    actual_ids = [result.instance_id for result in batch_run.results]
    if actual_ids != expected_ids:
        raise AssertionError("Adapter did not preserve instance order.")
    checks.append("stable_order")

    if _manifest_json(adapter.manifest()) != frozen_manifest:
        raise AssertionError(
            "Adapter mutated its frozen solver manifest during execution."
        )
    checks.append("frozen_manifest")
    return ConformanceReport(manifest.solver_id, tuple(checks))


def _manifest_json(manifest: Any) -> str:
    return json.dumps(
        {
            "api_version": manifest.api_version,
            "solver_id": manifest.solver_id,
            "problem_id": manifest.problem_id,
            "source_version": manifest.source_version,
            "checkpoint_files": manifest.checkpoint_files,
            "inference_parameters": dict(manifest.inference_parameters),
            "stochastic": manifest.stochastic,
        },
        sort_keys=True,
    )
