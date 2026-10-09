from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
import torch

from cs_pif.problems.cvrp import CVRPInstance


@dataclass(frozen=True)
class PaddedBatch:
    instance_ids: tuple[str, ...]
    data: torch.Tensor
    valid_node_mask: torch.Tensor
    customer_counts: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.instance_ids)

    @property
    def max_customer_count(self) -> int:
        return int(self.data.shape[1] - 1)


def _normalized_data(instance: CVRPInstance) -> np.ndarray:
    coords = np.asarray(instance.coords, dtype=np.float32)
    if coords.shape != (instance.customer_count + 1, 2):
        raise ValueError(f"{instance.instance_id}: invalid coordinate shape")
    offset = coords.min(axis=0)
    scale = float((coords.max(axis=0) - offset).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"{instance.instance_id}: DGL normalization needs a non-zero span"
        )
    if not np.isfinite(instance.capacity) or instance.capacity <= 0:
        raise ValueError(f"{instance.instance_id}: capacity must be positive")
    demands = np.asarray(instance.demands, dtype=np.float32)
    if demands.shape != (instance.customer_count + 1,):
        raise ValueError(f"{instance.instance_id}: invalid demand shape")
    return np.column_stack(((coords - offset) / scale, demands / instance.capacity))


def pad_instances(
    instances: Sequence[CVRPInstance],
    device: torch.device,
) -> PaddedBatch:
    if not instances:
        raise ValueError("Cannot pad an empty DGL batch.")
    if len({instance.instance_id for instance in instances}) != len(instances):
        raise ValueError("DGL batches require unique instance IDs.")
    sizes = tuple(int(instance.customer_count) for instance in instances)
    if any(size < 1 for size in sizes):
        raise ValueError("DGL instances must contain at least one customer.")
    max_size = max(sizes)
    data = np.zeros((len(instances), max_size + 1, 3), dtype=np.float32)
    valid = np.zeros((len(instances), max_size + 1), dtype=np.bool_)
    for row, instance in enumerate(instances):
        values = _normalized_data(instance)
        data[row, : len(values)] = values
        valid[row, : len(values)] = True
    return PaddedBatch(
        instance_ids=tuple(instance.instance_id for instance in instances),
        data=torch.as_tensor(data, dtype=torch.float32, device=device),
        valid_node_mask=torch.as_tensor(valid, dtype=torch.bool, device=device),
        customer_counts=torch.as_tensor(sizes, dtype=torch.long, device=device),
    )


def stable_start_indices(
    instance_ids: Sequence[str],
    customer_counts: Sequence[int] | torch.Tensor,
    *,
    seed: int,
) -> tuple[int, ...]:
    sizes = [int(value) for value in customer_counts]
    if len(instance_ids) != len(sizes):
        raise ValueError("Instance IDs and customer counts must align.")
    starts = []
    for instance_id, size in zip(instance_ids, sizes):
        if size < 1:
            raise ValueError("DGL instances must contain at least one customer.")
        digest = hashlib.sha256(f"{int(seed)}:{instance_id}".encode()).digest()
        starts.append(1 + int.from_bytes(digest[:8], "big") % size)
    return tuple(starts)
