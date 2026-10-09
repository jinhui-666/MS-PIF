from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import PaddedTSPBatch


def _integral_nodes(values: torch.Tensor, *, device: torch.device, size: int, name: str) -> torch.Tensor:
    nodes = torch.as_tensor(values, device=device)
    if nodes.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    if torch.is_complex(nodes) or (
        torch.is_floating_point(nodes)
        and (not torch.isfinite(nodes).all() or torch.any(nodes != torch.trunc(nodes)))
    ):
        raise ValueError(f"{name} must contain integral node indices.")
    return nodes.to(dtype=torch.long)


@dataclass(frozen=True)
class PaddedTSPState:
    selected_node_list: torch.Tensor
    selected_counts: torch.Tensor
    current_node: torch.Tensor
    visited_mask: torch.Tensor
    active_mask: torch.Tensor


class PaddedTSPEnvironment:
    def __init__(self, batch: PaddedTSPBatch):
        self.batch = batch
        self.selected_node_list: torch.Tensor | None = None
        self.selected_counts: torch.Tensor | None = None
        self.current_node: torch.Tensor | None = None
        self.visited_mask: torch.Tensor | None = None
        self.active_mask: torch.Tensor | None = None

    def reset(self, starts: torch.Tensor) -> PaddedTSPState:
        device = self.batch.normalized_coords.device
        starts = _integral_nodes(
            starts, device=device, size=self.batch.batch_size, name="starts"
        )
        if torch.any(starts < 0) or torch.any(starts >= self.batch.max_nodes):
            raise ValueError("TSP start node is out of range.")
        rows = torch.arange(self.batch.batch_size, device=device)
        if torch.any(~self.batch.valid_node_mask[rows, starts]):
            raise ValueError("TSP start node is padded.")

        self.selected_node_list = torch.zeros(
            (self.batch.batch_size, self.batch.max_nodes), dtype=torch.long, device=device
        )
        self.selected_node_list[:, 0] = starts
        self.selected_counts = torch.ones(self.batch.batch_size, dtype=torch.long, device=device)
        self.current_node = starts.clone()
        self.visited_mask = ~self.batch.valid_node_mask.clone()
        self.visited_mask[rows, starts] = True
        self.active_mask = self.selected_counts < self.batch.node_counts
        return self.state()

    def step(self, selected: torch.Tensor) -> tuple[PaddedTSPState, bool]:
        self._require_reset()
        assert self.selected_node_list is not None
        assert self.selected_counts is not None
        assert self.current_node is not None
        assert self.visited_mask is not None
        assert self.active_mask is not None

        selected = _integral_nodes(
            selected,
            device=self.batch.normalized_coords.device,
            size=self.batch.batch_size,
            name="selected",
        )
        if torch.any(selected[~self.active_mask] != 0):
            raise ValueError("Inactive TSP rows require sentinel selection 0.")

        rows = torch.nonzero(self.active_mask, as_tuple=False).squeeze(1)
        nodes = selected[rows]
        if torch.any(nodes < 0) or torch.any(nodes >= self.batch.max_nodes):
            raise ValueError("Selected TSP node is out of range.")
        if torch.any(~self.batch.valid_node_mask[rows, nodes]):
            raise ValueError("Selected TSP node is padded.")
        if torch.any(self.visited_mask[rows, nodes]):
            raise ValueError("Selected TSP node was already visited.")

        self.selected_node_list[rows, self.selected_counts[rows]] = nodes
        self.selected_counts[rows] += 1
        self.current_node[rows] = nodes
        self.visited_mask[rows, nodes] = True
        self.active_mask = self.selected_counts < self.batch.node_counts
        done = not bool(self.active_mask.any().item())
        return self.state(), done

    def state(self) -> PaddedTSPState:
        self._require_reset()
        assert self.selected_node_list is not None
        assert self.selected_counts is not None
        assert self.current_node is not None
        assert self.visited_mask is not None
        assert self.active_mask is not None
        return PaddedTSPState(
            selected_node_list=self.selected_node_list,
            selected_counts=self.selected_counts,
            current_node=self.current_node,
            visited_mask=self.visited_mask,
            active_mask=self.active_mask,
        )

    def tours(self) -> tuple[torch.Tensor, ...]:
        self._require_reset()
        assert self.selected_node_list is not None
        assert self.selected_counts is not None
        return tuple(
            self.selected_node_list[row, : int(count.item())].clone()
            for row, count in enumerate(self.selected_counts)
        )

    def _require_reset(self) -> None:
        if self.selected_node_list is None:
            raise RuntimeError("Call reset() before using the TSP environment.")
