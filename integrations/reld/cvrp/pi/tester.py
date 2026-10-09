import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .data import BucketConfig, CVRPInstance, load_instances, make_buckets, pad_instances
from .backend import solve_padded_batch
from .model import MaskedReLDModel, load_checkpoint


@dataclass
class ReLDResult:
    rows: List[Dict]
    total_solve_seconds: float


class PaddedReLDTester:
    def __init__(
        self,
        checkpoint: Path,
        device: torch.device,
        bucket_configs: Sequence[BucketConfig],
        model_params: Optional[Dict] = None,
    ):
        self.device = device
        self.bucket_configs = list(bucket_configs)
        self.model: MaskedReLDModel = load_checkpoint(
            checkpoint, device=device, model_params=model_params
        )

    @staticmethod
    def _synchronize(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.no_grad()
    def solve_bucket(
        self,
        instances: Sequence[CVRPInstance],
        bucket_config: BucketConfig,
    ) -> List[Dict]:
        batch = pad_instances(instances, self.device)

        self._synchronize(self.device)
        start = time.perf_counter()
        inference = solve_padded_batch(
            self.model,
            batch,
            pomo_size=bucket_config.pomo_size,
            aug_factor=bucket_config.aug_factor,
        )
        self._synchronize(self.device)
        elapsed = time.perf_counter() - start

        legality = inference.environment.check_legality(inference.sequences)
        costs_cpu = inference.costs.detach().cpu().tolist()
        no_aug_cpu = inference.no_aug_costs.detach().cpu().tolist()
        bks_values = batch.bks.detach().cpu().tolist() if batch.bks is not None else [None] * len(instances)
        rows = []
        for instance, cost, no_aug_cost, bks, legal in zip(
            instances, costs_cpu, no_aug_cpu, bks_values, legality
        ):
            rows.append(
                {
                    "name": instance.name,
                    "customer_count": instance.customer_count,
                    "bucket_upper_bound": bucket_config.upper_bound,
                    "padding_customer_count": batch.max_customers,
                    "bucket_pomo_size": inference.effective_pomo,
                    "instance_pomo_size": min(
                        bucket_config.pomo_size,
                        instance.customer_count,
                    ),
                    "aug_factor": bucket_config.aug_factor,
                    "no_aug_cost": no_aug_cost,
                    "cost": cost,
                    "bks": bks,
                    "gap_percent": None if bks is None else (cost - bks) / bks * 100.0,
                    "is_legal": legal["is_legal"],
                    "illegal_reason": legal["illegal_reason"],
                    "max_load_ratio": legal["max_load_ratio"],
                    "elapsed_seconds": elapsed,
                }
            )
        return rows

    def run_instances(self, instances: Sequence[CVRPInstance]) -> ReLDResult:
        rows: List[Dict] = []
        total = 0.0
        for bucket in make_buckets(instances, self.bucket_configs):
            bucket_rows = self.solve_bucket(bucket.instances, bucket.config)
            rows.extend(bucket_rows)
            total += bucket_rows[0]["elapsed_seconds"]
        return ReLDResult(rows=rows, total_solve_seconds=total)

    def run(self, data_path: Path, limit: Optional[int] = None) -> ReLDResult:
        return self.run_instances(load_instances(data_path, limit=limit))
