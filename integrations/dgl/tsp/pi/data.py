from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
import torch

from cs_pif.problems.tsp import TSPInstance


@dataclass(frozen=True)
class PaddedTSPBatch:
    instance_ids: tuple[str, ...]
    normalized_coords: torch.Tensor
    original_coords: torch.Tensor
    valid_node_mask: torch.Tensor
    node_counts: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.instance_ids)

    @property
    def max_nodes(self) -> int:
        return int(self.normalized_coords.shape[1])


def pad_instances(
    instances: Sequence[TSPInstance], device: torch.device
) -> PaddedTSPBatch:
    if not instances:
        raise ValueError("Cannot pad an empty TSP batch.")
    instance_ids = tuple(instance.instance_id for instance in instances)
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("TSP batches require unique instance IDs.")

    counts = tuple(int(instance.node_count) for instance in instances)
    max_nodes = max(counts)
    normalized = np.zeros((len(instances), max_nodes, 2), dtype=np.float32)
    original = np.zeros_like(normalized)
    valid = np.zeros((len(instances), max_nodes), dtype=np.bool_)
    for row, instance in enumerate(instances):
        coords = np.asarray(instance.coords, dtype=np.float64)
        if coords.shape != (counts[row], 2) or counts[row] < 1:
            raise ValueError(f"{instance.instance_id}: invalid TSP coordinate shape.")
        if not np.isfinite(coords).all():
            raise ValueError(f"{instance.instance_id}: coordinates must be finite.")
        minimum = coords.min(axis=0)
        span = float((coords.max(axis=0) - minimum).max())
        if not np.isfinite(span) or span <= 0:
            raise ValueError(f"{instance.instance_id}: normalization needs a non-zero span.")
        normalized[row, : counts[row]] = (coords - minimum) / span
        original[row, : counts[row]] = coords
        valid[row, : counts[row]] = True

    return PaddedTSPBatch(
        instance_ids=instance_ids,
        normalized_coords=torch.as_tensor(normalized, dtype=torch.float32, device=device),
        original_coords=torch.as_tensor(original, dtype=torch.float32, device=device),
        valid_node_mask=torch.as_tensor(valid, dtype=torch.bool, device=device),
        node_counts=torch.as_tensor(counts, dtype=torch.long, device=device),
    )


def stable_start_indices(
    instance_ids: Sequence[str], node_counts: Sequence[int] | torch.Tensor, seed: int
) -> tuple[int, ...]:
    counts = tuple(int(value) for value in node_counts)
    if len(instance_ids) != len(counts):
        raise ValueError("Instance IDs and node counts must align.")
    starts = []
    for instance_id, count in zip(instance_ids, counts):
        if count < 1:
            raise ValueError("TSP node counts must be positive.")
        digest = hashlib.sha256(f"{int(seed)}:{instance_id}".encode()).digest()
        starts.append(int.from_bytes(digest, "big") % count)
    return tuple(starts)
