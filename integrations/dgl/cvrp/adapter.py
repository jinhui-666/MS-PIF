from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cs_pif.api import ModelSolution, RunContext, SolverManifest
from cs_pif.problems.cvrp import CVRPInstance, CVRPProblem, CVRPSolution


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DGL_ROOT = PROJECT_ROOT / "backbones" / "DGL"
DEFAULT_CHECKPOINT = DGL_ROOT / "CVRP" / "pretrain" / "checkpoint-100.pt"


class DGLCVRPAdapter:
    problem = CVRPProblem()

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
            solver_id="dgl-cvrp",
            problem_id="cvrp",
            source_version="vendored-dgl-cvrp-pi-v1-uniform-x-xl-profile",
            checkpoint_files=(self.checkpoint,),
            inference_parameters={
                "aug_size": 1,
                "pomo_size": 1,
                "beam_size": 1,
                "test_mode": "pomo_test",
                "knn": 50,
                "depot_knn": 50,
                "start_rule": "stable-instance-seed-v1",
                "dtype": "float32",
            },
            stochastic=False,
        )

    def load(self, device: str) -> None:
        import torch

        from .pi.backend import load_checkpoint

        self.device = torch.device(device)
        self.model = load_checkpoint(Path(self.checkpoint), self.device)

    def resource_features(
        self, instances: Sequence[CVRPInstance]
    ) -> Mapping[str, float]:
        if not instances:
            raise ValueError("Cannot describe an empty DGL batch.")
        return {
            "B": float(len(instances)),
            "N": float(max(instance.customer_count for instance in instances)),
        }

    def feasibility_batches(
        self, batch_size: int, size: int, seed: int
    ) -> tuple[Sequence[CVRPInstance], ...]:
        rng = np.random.default_rng(seed)
        instances = []
        for index in range(batch_size):
            instances.append(
                CVRPInstance(
                    instance_id=f"dgl-feasibility-b{batch_size}-n{size}-s{seed}-{index}",
                    coords=rng.random((size + 1, 2), dtype=np.float32) * 1000.0,
                    demands=np.concatenate(
                        (np.zeros(1, dtype=np.float32), np.ones(size, dtype=np.float32))
                    ),
                    capacity=1.0,
                    round_distances=True,
                    metadata={
                        "generated": True,
                        "purpose": "strict_feasibility",
                        "pressure": "capacity_one",
                        "decode_length_bound": 2 * size + 1,
                    },
                )
            )
        return (instances,)

    def solve_batch(
        self,
        instances: Sequence[CVRPInstance],
        context: RunContext | None,
    ) -> Sequence[ModelSolution]:
        if self.model is None or self.device is None:
            raise RuntimeError("Call load() before solve_batch().")
        if not instances:
            raise ValueError("Cannot solve an empty DGL batch.")
        from .pi.backend import solve_padded_batch
        from .pi.data import pad_instances, stable_start_indices

        seed = 0 if context is None else int(context.global_seed)
        batch = pad_instances(instances, self.device)
        starts = stable_start_indices(
            batch.instance_ids, batch.customer_counts, seed=seed
        )
        inference = solve_padded_batch(self.model, batch, starts=starts)
        nodes = inference.sequences.detach().cpu().tolist()
        flags = inference.via_depot.detach().cpu().tolist()
        outputs = []
        for instance, sequence, via in zip(instances, nodes, flags):
            outputs.append(
                ModelSolution(
                    instance_id=instance.instance_id,
                    decision=CVRPSolution(
                        routes=self._sequence_to_routes(
                            sequence, via, instance.customer_count
                        )
                    ),
                    diagnostics={
                        "aug_size": 1,
                        "pomo_size": 1,
                        "beam_size": 1,
                        "test_mode": "pomo_test",
                        "start_node": int(sequence[0]),
                    },
                )
            )
        return outputs

    @staticmethod
    def _sequence_to_routes(
        nodes: Sequence[int],
        via_depot: Sequence[int],
        customer_count: int,
    ) -> tuple[tuple[int, ...], ...]:
        routes: list[tuple[int, ...]] = []
        current: list[int] = []
        seen: set[int] = set()
        for raw_node, raw_flag in zip(nodes, via_depot):
            node = int(raw_node)
            if not 1 <= node <= customer_count or node in seen:
                continue
            if bool(raw_flag) and current:
                routes.append(tuple(current))
                current = []
            current.append(node)
            seen.add(node)
            if len(seen) == customer_count:
                break
        if current:
            routes.append(tuple(current))
        return tuple(routes)


def create_adapter(options: Mapping[str, Any]) -> DGLCVRPAdapter:
    return DGLCVRPAdapter(
        checkpoint=str(options.get("checkpoint", DEFAULT_CHECKPOINT))
    )
