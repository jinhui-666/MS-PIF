from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cs_pif.api import ModelSolution, RunContext, SolverManifest
from cs_pif.problems.cvrp import CVRPInstance, CVRPProblem, CVRPSolution


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RELD_BACKBONE_ROOT = PROJECT_ROOT / "backbones" / "reld"
RELD_DEFAULT_MODEL_PARAMETERS = {
    "embedding_dim": 128,
    "encoder_layer_num": 6,
    "head_num": 8,
    "qkv_dim": 16,
    "forcing_first_step": False,
    "logit_clipping": 50,
    "ff_hidden_dim": 512,
    "eval_type": "greedy",
}


class ReLDCVRPAdapter:
    """Thin CS-PIF adapter around ReLD's padding-aware inference wrapper."""

    problem = CVRPProblem()

    def __init__(
        self,
        *,
        checkpoint: str,
        pomo_size: int,
        aug_factor: int,
        greedy_tie_relative_tolerance: float = 0.0,
        model_parameters: Mapping[str, Any] | None = None,
    ):
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            project_path = PROJECT_ROOT / checkpoint_path
            checkpoint_path = project_path if project_path.exists() else checkpoint_path
        self.checkpoint = str(checkpoint_path.resolve())
        self.pomo_size = int(pomo_size)
        self.aug_factor = int(aug_factor)
        self.greedy_tie_relative_tolerance = float(greedy_tie_relative_tolerance)
        self.model_parameters = dict(RELD_DEFAULT_MODEL_PARAMETERS)
        self.model_parameters.update(model_parameters or {})
        if self.pomo_size < 1:
            raise ValueError("pomo_size must be positive.")
        if self.aug_factor not in (1, 8):
            raise ValueError("ReLD aug_factor must be 1 or 8.")
        if self.greedy_tie_relative_tolerance < 0:
            raise ValueError("greedy_tie_relative_tolerance must be non-negative.")
        self.device = None
        self.model = None

    def manifest(self) -> SolverManifest:
        return SolverManifest(
            api_version="1",
            solver_id="reld-cvrp",
            problem_id="cvrp",
            source_version="vendored-reld-cvrp-pi-v6-uniform-x-xl-profile",
            checkpoint_files=(self.checkpoint,),
            inference_parameters={
                "pomo_size": self.pomo_size,
                "aug_factor": self.aug_factor,
                "greedy_tie_relative_tolerance": (
                    self.greedy_tie_relative_tolerance
                ),
                "pomo_rule": "min(profile_pomo_size, instance_customer_count)",
                "model_parameters": self.model_parameters,
                "dtype": "float32",
            },
            stochastic=False,
        )

    def load(self, device: str) -> None:
        import torch

        from .pi.model import load_checkpoint

        self.device = torch.device(device)
        self.model = load_checkpoint(
            Path(self.checkpoint),
            device=self.device,
            model_params=self.model_parameters,
            greedy_tie_relative_tolerance=self.greedy_tie_relative_tolerance,
        )

    def resource_features(
        self,
        instances: Sequence[CVRPInstance],
    ) -> Mapping[str, float]:
        if not instances:
            raise ValueError("Cannot describe an empty ReLD batch.")
        max_size = max(instance.customer_count for instance in instances)
        return {
            "B": float(len(instances)),
            "N": float(max_size),
        }

    def feasibility_batches(
        self,
        batch_size: int,
        size: int,
        seed: int,
    ) -> tuple[Sequence[CVRPInstance], ...]:
        """Exercise ReLD's longest valid decode path for Safe(B,N) only."""
        rng = np.random.default_rng(seed)
        instances = []
        for index in range(batch_size):
            instances.append(
                CVRPInstance(
                    instance_id=(
                        f"reld-feasibility-b{batch_size}-n{size}-s{seed}-{index}"
                    ),
                    coords=rng.random((size + 1, 2), dtype=np.float32) * 1000.0,
                    demands=np.concatenate(
                        [
                            np.zeros(1, dtype=np.float32),
                            np.ones(size, dtype=np.float32),
                        ]
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
        context: RunContext,
    ) -> Sequence[ModelSolution]:
        del context  # Current frozen ReLD greedy inference is deterministic.
        if self.model is None or self.device is None:
            raise RuntimeError("Call load() before solve_batch().")
        if not instances:
            raise ValueError("Cannot solve an empty ReLD batch.")
        import torch

        from .pi.data import CVRPInstance as ReLDInstance
        from .pi.data import pad_instances
        from .pi.backend import solve_padded_batch

        native = [
            ReLDInstance(
                name=instance.instance_id,
                coords=torch.as_tensor(instance.coords, dtype=torch.float32),
                demands=torch.as_tensor(instance.demands, dtype=torch.float32),
                capacity=float(instance.capacity),
                bks=instance.best_known,
            )
            for instance in instances
        ]
        batch = pad_instances(native, self.device)

        try:
            inference = solve_padded_batch(
                self.model,
                batch,
                pomo_size=self.pomo_size,
                aug_factor=self.aug_factor,
            )
            sequences = inference.sequences.detach().cpu().tolist()
            no_aug = inference.no_aug_costs.detach().cpu().tolist()
        finally:
            self._clear_inference_cache()
        outputs = []
        for instance, sequence, no_aug_cost in zip(instances, sequences, no_aug):
            outputs.append(
                ModelSolution(
                    instance_id=instance.instance_id,
                    decision=CVRPSolution(
                        routes=self._sequence_to_routes(
                            sequence, instance.customer_count
                        )
                    ),
                    diagnostics={
                        "pomo_size": min(self.pomo_size, instance.customer_count),
                        "dense_pomo_width": inference.effective_pomo,
                        "aug_factor": self.aug_factor,
                        "no_aug_cost": float(no_aug_cost),
                    },
                )
            )
        return outputs

    @staticmethod
    def _sequence_to_routes(
        sequence: Sequence[int],
        customer_count: int,
    ) -> tuple[tuple[int, ...], ...]:
        routes: list[tuple[int, ...]] = []
        current: list[int] = []
        seen: set[int] = set()
        for raw_node in sequence:
            node = int(raw_node)
            if node == 0:
                if current:
                    routes.append(tuple(current))
                    current = []
                if len(seen) == customer_count:
                    break
            elif 1 <= node <= customer_count and node not in seen:
                current.append(node)
                seen.add(node)
        if current:
            routes.append(tuple(current))
        return tuple(routes)

    def _clear_inference_cache(self) -> None:
        if self.model is None:
            return
        for name in (
            "encoded_nodes",
            "node_mask",
            "pomo_valid_mask",
            "k",
            "v",
            "single_head_key",
        ):
            if hasattr(self.model, name):
                setattr(self.model, name, None)

def create_adapter(options: Mapping[str, Any]) -> ReLDCVRPAdapter:
    checkpoint = options.get(
        "checkpoint",
        str(
            RELD_BACKBONE_ROOT
            / "CVRP"
            / "weights"
            / "ReLD"
            / "model_epoch_90.pt"
        ),
    )
    return ReLDCVRPAdapter(
        checkpoint=str(checkpoint),
        pomo_size=int(options.get("pomo_size", 1)),
        aug_factor=int(options.get("aug_factor", 1)),
        greedy_tie_relative_tolerance=float(
            options.get("greedy_tie_relative_tolerance", 0.0)
        ),
        model_parameters=options.get("model_parameters"),
    )
