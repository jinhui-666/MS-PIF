from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import PaddedBatch


@dataclass(frozen=True)
class CandidateBatch:
    rows: torch.Tensor
    model_input: torch.Tensor
    direct_indices: torch.Tensor
    depot_indices: torch.Tensor
    direct_ninf_mask: torch.Tensor


class DGLPIEnvironment:
    """Vectorized variable-size CVRP state with padding excluded everywhere."""

    def __init__(self, batch: PaddedBatch):
        self.batch = batch
        self.device = batch.data.device
        self.base_data = batch.data.clone()
        self.valid_node_mask = batch.valid_node_mask
        self.customer_counts = batch.customer_counts
        self.batch_size = batch.batch_size
        self.max_customer_count = batch.max_customer_count
        self.node_count = self.max_customer_count + 1
        self.distance_matrix = torch.cdist(
            self.base_data[:, :, :2], self.base_data[:, :, :2], p=2
        ).to(torch.float32)
        self.selected_mask = ~self.valid_node_mask.clone()
        self.selected_mask[:, 0] = True
        self.selected_counts = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        self.capacity = torch.ones(
            self.batch_size, dtype=torch.float32, device=self.device
        )
        self.sequence = torch.zeros(
            (self.batch_size, self.max_customer_count),
            dtype=torch.long,
            device=self.device,
        )
        self.via_depot = torch.zeros_like(self.sequence)
        self.last_node = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        self.data = torch.zeros(
            (self.batch_size, self.node_count, 8),
            dtype=torch.float32,
            device=self.device,
        )
        valid_columns = self.valid_node_mask[:, None, :]
        counts = (self.customer_counts + 1).to(torch.float32)[:, None]
        values = torch.where(valid_columns, self.distance_matrix, 0.0)
        self.avg_unselected_distance = values.sum(dim=-1) / counts
        centered = torch.where(
            valid_columns,
            self.distance_matrix - self.avg_unselected_distance.unsqueeze(-1),
            0.0,
        )
        self.std_unselected_distance = torch.sqrt(
            torch.square(centered).sum(dim=-1) / counts
        )
        self.distance_to_depot = self.distance_matrix[:, :, 0]

    @property
    def available_mask(self) -> torch.Tensor:
        return self.valid_node_mask & ~self.selected_mask

    @property
    def remaining_counts(self) -> torch.Tensor:
        return self.customer_counts - self.selected_counts

    @property
    def active_mask(self) -> torch.Tensor:
        return self.remaining_counts > 0

    def start(self, starts: torch.Tensor) -> None:
        values = torch.as_tensor(starts, dtype=torch.long, device=self.device)
        if values.shape != (self.batch_size,):
            raise ValueError("One DGL start node is required per instance.")
        rows = torch.arange(self.batch_size, device=self.device)
        self.select(rows, values, torch.ones_like(values, dtype=torch.bool))

    def select(
        self,
        rows: torch.Tensor,
        nodes: torch.Tensor,
        via_depot: torch.Tensor,
    ) -> None:
        rows = rows.to(device=self.device, dtype=torch.long)
        nodes = nodes.to(device=self.device, dtype=torch.long)
        via_depot = via_depot.to(device=self.device, dtype=torch.bool)
        if not (rows.ndim == nodes.ndim == via_depot.ndim == 1):
            raise ValueError("DGL selections must be one-dimensional.")
        if not (len(rows) == len(nodes) == len(via_depot)):
            raise ValueError("DGL selection axes must align.")
        if len(rows) == 0:
            return
        if torch.any(~self.available_mask[rows, nodes]):
            raise ValueError("DGL selected a depot, padding, or repeated customer.")

        positions = self.selected_counts[rows]
        self.sequence[rows, positions] = nodes
        self.via_depot[rows, positions] = via_depot.to(torch.long)
        demands = self.base_data[rows, nodes, 2]
        self.capacity[rows] = torch.where(
            via_depot,
            1.0 - demands,
            self.capacity[rows] - demands,
        )
        self.selected_mask[rows, nodes] = True
        self.selected_counts[rows] += 1
        self.last_node[rows] = nodes
        self._update_features(rows, nodes)

    def _update_features(self, rows: torch.Tensor, selected: torch.Tensor) -> None:
        remaining_with_depot = (
            self.customer_counts[rows] + 1 - self.selected_counts[rows]
        ).to(torch.float32)
        distance_current = self.distance_matrix[rows, :, selected]
        before_avg = self.avg_unselected_distance[rows].clone()
        self.avg_unselected_distance[rows] = (
            before_avg * (remaining_with_depot + 1)[:, None] - distance_current
        ) / remaining_with_depot[:, None]
        variance = (
            torch.square(self.std_unselected_distance[rows])
            * (remaining_with_depot + 1)[:, None]
            - torch.square(distance_current - before_avg)
        ) / remaining_with_depot[:, None]
        self.std_unselected_distance[rows] = torch.sqrt(torch.clamp(variance, min=0))

        coords = self.base_data[rows, :, :2]
        depot = coords[:, :1, :]
        distance_depot = self.distance_to_depot[rows]
        sin_to_depot = (depot[:, :, 1] - coords[:, :, 1]) / (
            distance_depot + 1e-7
        )
        cos_to_depot = (depot[:, :, 0] - coords[:, :, 0]) / (
            distance_depot + 1e-7
        )
        self.data[rows] = torch.cat(
            (
                self.base_data[rows],
                distance_current.unsqueeze(-1),
                self.avg_unselected_distance[rows].unsqueeze(-1),
                self.std_unselected_distance[rows].unsqueeze(-1),
                sin_to_depot.unsqueeze(-1),
                cos_to_depot.unsqueeze(-1),
            ),
            dim=-1,
        )

    def candidate_batch(self, rows: torch.Tensor, k: int) -> CandidateBatch:
        rows = rows.to(device=self.device, dtype=torch.long)
        if len(rows) == 0 or k < 1:
            raise ValueError("A DGL candidate group must be non-empty.")
        if torch.any(self.remaining_counts[rows] < k):
            raise ValueError("Candidate width exceeds an instance's remaining nodes.")
        blocked = self.selected_mask[rows]
        last = self.last_node[rows]
        distance_last = self.distance_matrix[rows, :, last]
        distance_depot = self.distance_matrix[rows, :, 0]
        infinity = torch.full_like(distance_last, float("inf"))
        direct = torch.where(blocked, infinity, distance_last)
        depot = torch.where(blocked, infinity, distance_depot)
        direct_indices = torch.topk(direct, k=k, largest=False).indices.sort(dim=-1).values
        depot_indices = torch.topk(depot, k=k, largest=False).indices.sort(dim=-1).values

        feature_count = self.data.shape[-1]
        direct_nodes = self.data[rows].gather(
            1, direct_indices.unsqueeze(-1).expand(-1, -1, feature_count)
        ).clone()
        depot_nodes = self.data[rows].gather(
            1, depot_indices.unsqueeze(-1).expand(-1, -1, feature_count)
        ).clone()
        direct_mask = torch.where(
            direct_nodes[:, :, 2] > self.capacity[rows, None] + 1e-6,
            torch.full_like(direct_nodes[:, :, 2], float("-inf")),
            torch.zeros_like(direct_nodes[:, :, 2]),
        )
        safe_capacity = self.capacity[rows].clone()
        nearly_empty = safe_capacity < 1e-5
        safe_capacity[nearly_empty] += 1e-7
        direct_nodes[:, :, 2] = direct_nodes[:, :, 2] / safe_capacity[:, None]
        direct_nodes[nearly_empty, :, 2] = 1.1
        depot_nodes[:, :, 3] = self.distance_to_depot[rows].gather(1, depot_indices)

        last_nodes = self.data[rows, last]
        depot_anchor = self.data[rows, torch.zeros_like(last)]
        direct_stream = self._normalize_stream(
            torch.cat((direct_nodes, last_nodes[:, None, :]), dim=1)
        )
        depot_stream = self._normalize_stream(
            torch.cat((depot_nodes, depot_anchor[:, None, :]), dim=1)
        )
        direct_stream = torch.cat(
            (direct_stream, torch.zeros((*direct_stream.shape[:2], 1), device=self.device)),
            dim=-1,
        )
        depot_stream = torch.cat(
            (depot_stream, torch.ones((*depot_stream.shape[:2], 1), device=self.device)),
            dim=-1,
        )
        return CandidateBatch(
            rows=rows,
            model_input=torch.cat((direct_stream, depot_stream), dim=1),
            direct_indices=direct_indices,
            depot_indices=depot_indices,
            direct_ninf_mask=direct_mask,
        )

    @staticmethod
    def _normalize_stream(values: torch.Tensor) -> torch.Tensor:
        demands = values[:, :, 2].clone()
        minimum = values.min(dim=1, keepdim=True).values
        span = values.max(dim=1, keepdim=True).values - minimum
        normalized = torch.where(span != 0, (values - minimum) / span, 0.0)
        normalized[:, :, 2] = demands
        return normalized
