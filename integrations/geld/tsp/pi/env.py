from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import TSPBatch


@dataclass(frozen=True)
class PaddedTSPState:
    selected_node_list: torch.Tensor
    selected_counts: torch.Tensor
    current_node: torch.Tensor
    visited_mask: torch.Tensor
    active_mask: torch.Tensor


class PaddedTSPEnv:
    def __init__(self, batch: TSPBatch):
        self.batch = batch
        self.selected_node_list: torch.Tensor | None = None
        self.selected_counts: torch.Tensor | None = None
        self.current_node: torch.Tensor | None = None
        self.visited_mask: torch.Tensor | None = None
        self.active_mask: torch.Tensor | None = None

    def reset(self) -> PaddedTSPState:
        device = self.batch.model_coords.device
        self.selected_node_list = torch.zeros(
            self.batch.batch_size,
            self.batch.max_nodes,
            dtype=torch.long,
            device=device,
        )
        self.selected_counts = torch.ones(
            self.batch.batch_size,
            dtype=torch.long,
            device=device,
        )
        self.current_node = torch.zeros(
            self.batch.batch_size,
            dtype=torch.long,
            device=device,
        )
        self.visited_mask = self.batch.node_mask.clone()
        self.visited_mask[:, 0] = True
        self.active_mask = self.selected_counts < self.batch.node_counts
        return self._state()

    def step(self, selected: torch.Tensor) -> tuple[PaddedTSPState, bool]:
        self._require_reset()
        assert self.selected_node_list is not None
        assert self.selected_counts is not None
        assert self.current_node is not None
        assert self.visited_mask is not None
        assert self.active_mask is not None

        selected = torch.as_tensor(
            selected,
            dtype=torch.long,
            device=self.batch.model_coords.device,
        )
        if selected.shape != (self.batch.batch_size,):
            raise ValueError(
                f"selected must have shape ({self.batch.batch_size},), "
                f"got {tuple(selected.shape)}."
            )
        inactive = ~self.active_mask
        if torch.any(selected[inactive] != 0):
            raise ValueError("Inactive TSP rows require sentinel selection 0.")

        rows = torch.nonzero(self.active_mask, as_tuple=False).squeeze(1)
        nodes = selected[rows]
        if torch.any(nodes < 0) or torch.any(nodes >= self.batch.max_nodes):
            raise ValueError("Selected TSP node is out of range.")
        if torch.any(self.batch.node_mask[rows, nodes]):
            raise ValueError("Selected TSP node is padded.")
        if torch.any(self.visited_mask[rows, nodes]):
            raise ValueError("Selected TSP node was already visited.")

        write_positions = self.selected_counts[rows]
        self.selected_node_list[rows, write_positions] = nodes
        self.selected_counts[rows] += 1
        self.current_node[rows] = nodes
        self.visited_mask[rows, nodes] = True
        self.active_mask = self.selected_counts < self.batch.node_counts
        done = not bool(torch.any(self.active_mask).item())
        return self._state(), done

    def best_tours(self) -> tuple[torch.Tensor, ...]:
        self._require_reset()
        assert self.selected_node_list is not None
        return tuple(
            self.selected_node_list[row, : int(count.item())].clone()
            for row, count in enumerate(self.batch.node_counts)
        )

    def _state(self) -> PaddedTSPState:
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

    def _require_reset(self) -> None:
        if self.selected_node_list is None:
            raise RuntimeError("Call reset() before using the TSP environment.")
