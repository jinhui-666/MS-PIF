from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from cs_pif.api import ModelSolution, RunContext, SolverManifest
from cs_pif.problems.tsp import TSPInstance, TSPProblem, TSPSolution

from .pi.model import GELD_DEFAULT_MODEL_PARAMETERS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GELD_BACKBONE_ROOT = PROJECT_ROOT / "backbones" / "GELD"
GELD_SOURCE_COMMIT = "c151ee8f2f94ae60db73dd1906fddd8e5401754c"


class GELDTSPAdapter:
    problem = TSPProblem()

    def __init__(
        self,
        *,
        checkpoint: str,
        model_parameters: Mapping[str, Any] | None = None,
    ):
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            project_path = PROJECT_ROOT / checkpoint_path
            checkpoint_path = (
                project_path if project_path.exists() else checkpoint_path
            )
        self.checkpoint = str(checkpoint_path.resolve())
        self.model_parameters = dict(GELD_DEFAULT_MODEL_PARAMETERS)
        self.model_parameters.update(model_parameters or {})
        self.model_parameters["mode"] = "test"
        self.device = None
        self.model = None

    def manifest(self) -> SolverManifest:
        return SolverManifest(
            api_version="1",
            solver_id="geld-tsp",
            problem_id="tsp",
            source_version=f"geld-{GELD_SOURCE_COMMIT}-padded-greedy-v2",
            checkpoint_files=(self.checkpoint,),
            inference_parameters={
                "decode_method": "greedy",
                "first_node": 0,
                "max_local_candidates": 99,
                "region_grid": "3x3",
                "coordinate_normalization": (
                    "per-instance-axis-min-global-span"
                ),
                "parallel_backend": "padded-masked-batch",
                "distance_strategy": "on-demand-all-sizes-fp16-knn-above-10000",
                "beam_search": False,
                "augmentation": False,
                "prc": False,
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
        )

    def resource_features(
        self,
        instances: Sequence[TSPInstance],
    ) -> Mapping[str, float]:
        if not instances:
            raise ValueError("Cannot describe an empty GELD batch.")
        return {
            "B": float(len(instances)),
            "N": float(max(instance.node_count for instance in instances)),
        }

    def solve_batch(
        self,
        instances: Sequence[TSPInstance],
        context: RunContext,
    ) -> Sequence[ModelSolution]:
        del context
        if self.model is None or self.device is None:
            raise RuntimeError("Call load() before solve_batch().")
        if not instances:
            raise ValueError("Cannot solve an empty GELD batch.")

        from .pi.backend import solve_padded_batch
        from .pi.data import pad_instances

        batch = pad_instances(instances, self.device)
        try:
            inference = solve_padded_batch(self.model, batch)
            tours = tuple(tour.detach().cpu().tolist() for tour in inference.tours)
        finally:
            self._clear_inference_cache()

        return [
            ModelSolution(
                instance_id=instance.instance_id,
                decision=TSPSolution(tour=tuple(int(node) for node in tour)),
                diagnostics={
                    "decode_method": "greedy",
                    "first_node": 0,
                    "max_local_candidates": 99,
                    "parallel_backend": "padded-masked-batch",
                },
            )
            for instance, tour in zip(instances, tours)
        ]

    def _clear_inference_cache(self) -> None:
        if self.model is not None:
            self.model.clear_cache()


def create_adapter(options: Mapping[str, Any]) -> GELDTSPAdapter:
    checkpoint = options.get(
        "checkpoint",
        str(
            GELD_BACKBONE_ROOT
            / "result"
            / "pre_trained_model"
            / "checkpoint-49.pt"
        ),
    )
    return GELDTSPAdapter(
        checkpoint=str(checkpoint),
        model_parameters=options.get("model_parameters"),
    )
