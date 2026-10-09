from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from cs_pif.problems.tsp import TSPInstance


@dataclass(frozen=True)
class TSPBatch:
    names: list[str]
    model_coords: torch.Tensor
    original_coords: torch.Tensor
    node_mask: torch.Tensor
    node_counts: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.model_coords.shape[0])

    @property
    def max_nodes(self) -> int:
        return int(self.model_coords.shape[1])


def pad_instances(
    instances: Sequence[TSPInstance],
    device: torch.device,
) -> TSPBatch:
    if not instances:
        raise ValueError("Cannot pad an empty TSP instance list.")
    batch_size = len(instances)
    max_nodes = max(instance.node_count for instance in instances)
    model_coords = torch.zeros(
        batch_size,
        max_nodes,
        2,
        dtype=torch.float32,
        device=device,
    )
    original_coords = torch.zeros_like(model_coords)
    node_mask = torch.ones(
        batch_size,
        max_nodes,
        dtype=torch.bool,
        device=device,
    )
    node_counts = torch.empty(
        batch_size,
        dtype=torch.long,
        device=device,
    )
    epsilon = torch.finfo(torch.float32).eps

    for row, instance in enumerate(instances):
        coords = torch.as_tensor(
            instance.coords,
            dtype=torch.float32,
            device=device,
        )
        if not torch.isfinite(coords).all():
            raise ValueError(f"{instance.instance_id}: coordinates must be finite.")
        count = instance.node_count
        coord_min = coords.min(dim=0, keepdim=True).values
        coord_span = coords.max(dim=0, keepdim=True).values - coord_min
        scale = coord_span.max().clamp_min(epsilon)
        model_coords[row, :count] = (coords - coord_min) / scale
        original_coords[row, :count] = coords
        node_mask[row, :count] = False
        node_counts[row] = count

    return TSPBatch(
        names=[instance.instance_id for instance in instances],
        model_coords=model_coords,
        original_coords=original_coords,
        node_mask=node_mask,
        node_counts=node_counts,
    )
