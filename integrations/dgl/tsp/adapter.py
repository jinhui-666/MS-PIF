"""CS-PIF adapter for checkpoint-compatible DGL TSP inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from cs_pif.api import ModelSolution, RunContext, SolverManifest
from cs_pif.problems.tsp import TSPInstance, TSPProblem, TSPSolution


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "backbones" / "DGL" / "TSP" / "pretrain" / "checkpoint-100.pt"
)


class DGLTSPAdapter:
    problem = TSPProblem()

    def __init__(self, *, checkpoint: str):
        path = Path(checkpoint).expanduser()
        if not path.is_absolute():
            candidate = PROJECT_ROOT / path
            path = candidate if candidate.exists() else path
        self.checkpoint = str(path.resolve())
        self.device = None
        self.model = None

    def manifest(self) -> SolverManifest:
        return SolverManifest(
            api_version="1",
            solver_id="dgl-tsp",
            problem_id="tsp",
            source_version="vendored-dgl-tsp-pi-v2-bounded-cdist",
            checkpoint_files=(self.checkpoint,),
            inference_parameters={
                "aug_size": 1,
                "pomo_size": 1,
                "beam_size": 1,
                "test_mode": "pomo_test",
                "knn": 30,
                "start_rule": "stable-instance-seed-v1",
                "coordinate_normalization": "per-instance-axis-min-global-span",
                "input_domain": {
                    "minimum_node_count": 2,
                    "global_coordinate_span": "positive",
                },
                "dtype": "float32",
            },
            stochastic=False,
        )

    def load(self, device: str) -> None:
        import torch

        from .pi.model import load_model

        target = torch.device(device)
        model = load_model(self.checkpoint, target)
        self.device = target
        self.model = model

    def resource_features(self, instances: Sequence[TSPInstance]) -> Mapping[str, float]:
        if not instances:
            raise ValueError("Cannot describe an empty DGL TSP batch.")
        self._validate_input_domain(instances)
        return {"B": float(len(instances)), "N": float(max(item.node_count for item in instances))}

    def feasibility_batches(
        self, batch_size: int, size: int, seed: int
    ) -> tuple[Sequence[TSPInstance], ...]:
        if size < 2:
            raise ValueError("DGL TSP requires at least 2 nodes per instance.")
        batch = self.problem.generate(batch_size, size, seed)
        self._validate_input_domain(batch)
        return (batch,)

    def solve_batch(
        self, instances: Sequence[TSPInstance], context: RunContext
    ) -> Sequence[ModelSolution]:
        if self.model is None or self.device is None:
            raise RuntimeError("Call load() before solve_batch().")
        if not instances:
            raise ValueError("Cannot solve an empty DGL TSP batch.")
        self._validate_input_domain(instances)

        from .pi.backend import solve_padded_batch
        from .pi.data import pad_instances, stable_start_indices

        batch = pad_instances(instances, self.device)
        starts = stable_start_indices(batch.instance_ids, batch.node_counts, context.global_seed)
        inference = solve_padded_batch(self.model, batch, starts, knn_size=30)
        tours = tuple(tuple(int(node) for node in tour.cpu().tolist()) for tour in inference.tours)
        return [
            ModelSolution(
                instance_id=instance.instance_id,
                decision=TSPSolution(tour=tour),
                diagnostics={
                    "aug_size": 1,
                    "pomo_size": 1,
                    "beam_size": 1,
                    "knn": 30,
                    "start_node": int(start),
                },
            )
            for instance, tour, start in zip(instances, tours, starts)
        ]

    @staticmethod
    def _validate_input_domain(instances: Sequence[TSPInstance]) -> None:
        for instance in instances:
            if instance.node_count < 2:
                raise ValueError(
                    f"{instance.instance_id}: DGL TSP requires at least 2 nodes per instance."
                )
            span = float((instance.coords.max(axis=0) - instance.coords.min(axis=0)).max())
            if span <= 0:
                raise ValueError(
                    f"{instance.instance_id}: DGL TSP requires a positive global coordinate span."
                )


def create_adapter(options: Mapping[str, Any]) -> DGLTSPAdapter:
    return DGLTSPAdapter(checkpoint=str(options.get("checkpoint", DEFAULT_CHECKPOINT)))
