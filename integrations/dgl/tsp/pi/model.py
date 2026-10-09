"""Checkpoint-compatible DGL TSP inference with explicit padding masks."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .data import PaddedTSPBatch
from .env import PaddedTSPState


# PyTorch 1.12 CUDA cdist can overflow its indexing for very large outputs.
# Keep each result far below the signed 32-bit element limit (at most 64 MiB).
MAX_CDIST_ELEMENTS = 16 * 1024 * 1024


class DGLTSPFeatures:
    """Batch-local distance statistics; removal costs O(BN) per decode step."""

    def __init__(self, batch: PaddedTSPBatch, state: PaddedTSPState):
        self.batch = batch
        self.coords = batch.normalized_coords.masked_fill(
            ~batch.valid_node_mask.unsqueeze(-1), 0
        )
        self.remaining = batch.node_counts.clone()
        self.mean = torch.zeros_like(self.coords[:, :, 0])
        self.std = torch.zeros_like(self.mean)
        self._current_distances: torch.Tensor | None = None
        self._start_distances: torch.Tensor | None = None
        # Match upstream's initial float32 population statistics over real nodes.
        # Each query chunk uses the original cdist kernel and exactly the real
        # columns used by upstream; no full B x N x N tensor is retained.
        for row in range(batch.batch_size):
            real_nodes = int(batch.node_counts[row].item())
            chunk = MAX_CDIST_ELEMENTS // real_nodes if MAX_CDIST_ELEMENTS > 0 else 0
            if chunk < 1:
                raise ValueError("CDist output limit must accommodate at least one row.")
            real_coords = self.coords[row : row + 1, :real_nodes]
            for offset in range(0, batch.max_nodes, chunk):
                end = min(offset + chunk, batch.max_nodes)
                values = torch.cdist(
                    self.coords[row : row + 1, offset:end],
                    real_coords,
                    compute_mode="donot_use_mm_for_euclid_dist",
                )[0]
                self.mean[row, offset:end] = values.mean(dim=-1)
                self.std[row, offset:end] = values.std(dim=-1, unbiased=False)
        # The checkpoint was trained with removal updates, including the start.
        for step in range(int(state.selected_counts.max().item())):
            self.advance(state.selected_node_list[:, step], state.selected_counts > step)

    def advance(self, selected: torch.Tensor, active: torch.Tensor) -> None:
        removed = self._row_distances(selected)
        self._current_distances = removed
        if self._start_distances is None:
            self._start_distances = removed
        before_count = self.remaining[:, None]
        remaining = self.remaining - active.long()
        count = remaining.clamp_min(1)[:, None]
        mean = (self.mean * before_count - removed) / count
        # Preserve the upstream recurrence; this is not exact remaining variance.
        variance = (self.std.square() * before_count - (removed - self.mean).square()) / count
        std = variance.clamp_min(0).sqrt()
        finished = remaining[:, None] == 0
        self.mean = torch.where(active[:, None], mean.masked_fill(finished, 0), self.mean)
        self.std = torch.where(active[:, None], std.masked_fill(finished, 0), self.std)
        self.remaining = remaining

    def _row_distances(self, selected: torch.Tensor) -> torch.Tensor:
        row_chunk = MAX_CDIST_ELEMENTS // self.batch.max_nodes if MAX_CDIST_ELEMENTS > 0 else 0
        if row_chunk < 1:
            raise ValueError("CDist output limit must accommodate one full distance row.")
        distances = []
        for offset in range(0, self.batch.batch_size, row_chunk):
            end = min(offset + row_chunk, self.batch.batch_size)
            rows = torch.arange(offset, end, device=self.coords.device)
            query = self.coords[rows, selected[offset:end]].unsqueeze(1)
            distances.append(torch.cdist(
                query, self.coords[offset:end],
                compute_mode="donot_use_mm_for_euclid_dist",
            )[:, 0])
        return torch.cat(distances, dim=0)

    def build(self, state: PaddedTSPState) -> torch.Tensor:
        rows = torch.arange(self.batch.batch_size, device=self.coords.device)
        assert self._current_distances is not None
        assert self._start_distances is not None
        current_distance = self._current_distances
        start = state.selected_node_list[:, 0]
        to_start = self.coords[rows, start][:, None] - self.coords
        start_distance = self._start_distances
        direction = to_start / (start_distance.unsqueeze(-1) + 1e-7)
        features = torch.cat(
            (
                self.coords,
                current_distance.unsqueeze(-1),
                self.mean.unsqueeze(-1),
                self.std.unsqueeze(-1),
                direction[:, :, 1:2],
                direction[:, :, 0:1],
            ),
            dim=-1,
        )
        valid = self.batch.valid_node_mask & state.active_mask[:, None]
        return features.masked_fill(~valid.unsqueeze(-1), 0)


def candidate_inputs(
    batch: PaddedTSPBatch,
    state: PaddedTSPState,
    features: torch.Tensor,
    knn_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted KNN indices and masked, per-feature local normalization."""
    if not isinstance(knn_size, int) or isinstance(knn_size, bool) or knn_size < 1:
        raise ValueError("knn_size must be a positive integer.")
    available = batch.valid_node_mask & ~state.visited_mask & state.active_mask[:, None]
    distances = features[:, :, 2].masked_fill(~available, float("inf"))
    width = min(knn_size, batch.max_nodes)
    # Stable sorting retains ascending real-node indices at equal distances.
    # An epsilon tie-break would incorrectly reorder genuinely unequal distances.
    nodes = distances.sort(dim=1, stable=True).indices[:, :width].sort(dim=1).values
    valid_candidates = available.gather(1, nodes)
    candidates = features.gather(1, nodes[:, :, None].expand(-1, -1, 7))
    rows = torch.arange(batch.batch_size, device=features.device)
    current = features[rows, state.current_node][:, None]
    values = torch.cat((candidates, current), dim=1)
    # The current-node token is always present, including inert finished rows.
    valid = torch.cat((valid_candidates, torch.ones_like(state.active_mask[:, None])), dim=1)
    minimum = values.masked_fill(~valid[:, :, None], float("inf")).amin(dim=1, keepdim=True)
    maximum = values.masked_fill(~valid[:, :, None], float("-inf")).amax(dim=1, keepdim=True)
    span = maximum - minimum
    normalized = (values - minimum) / span.masked_fill(span == 0, 1)
    return nodes, normalized.masked_fill(~valid[:, :, None], 0), valid


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Linear(128, 512)
        self.W2 = nn.Linear(512, 128)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.W2(torch.relu(self.W1(values)))


class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(128, 128, bias=False)
        self.Wk = nn.Linear(128, 128, bias=False)
        self.Wv = nn.Linear(128, 128, bias=False)
        self.multi_head_combine = nn.Linear(128, 128)
        self.feedForward = FeedForward()

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch_size, width, _ = values.shape
        q, k, v = (
            layer(values).reshape(batch_size, width, 8, 16).transpose(1, 2)
            for layer in (self.Wq, self.Wk, self.Wv)
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) / 4.0
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        attended = torch.matmul(scores.softmax(dim=-1), v)
        attended = attended.transpose(1, 2).reshape(batch_size, width, 128)
        values = values + self.multi_head_combine(attended)
        return (values + self.feedForward(values)).masked_fill(~valid[:, :, None], 0)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(7, 128)
        self.layers = nn.ModuleList([AttentionLayer()])

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        values = self.embedding(values.masked_fill(~valid[:, :, None], 0))
        for layer in self.layers:
            values = layer(values, valid)
        return values


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_last_node = nn.Linear(128, 128)
        # Present in the released checkpoint, although unused by its inference path.
        self.embedding_last_node_pos = nn.Linear(128, 128)
        self.layers = nn.ModuleList([AttentionLayer() for _ in range(3)])
        self.k_1 = nn.Linear(128, 128)
        self.Linear_final = nn.Linear(128, 1)

    def forward(self, encoded: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        current = self.embedding_last_node(encoded[:, -1:])
        values = torch.cat((encoded[:, :-1], current), dim=1)
        for layer in self.layers:
            values = layer(values, valid)
        scores = self.Linear_final(values).squeeze(-1)
        actionable = valid.clone()
        actionable[:, -1] = False
        return scores.masked_fill(~actionable, float("-inf"))


class DGLTSPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def score_candidates(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs, valid), valid)

    def forward(
        self,
        batch: PaddedTSPBatch,
        state: PaddedTSPState,
        features: torch.Tensor,
        knn_size: int = 30,
    ) -> torch.Tensor:
        nodes, inputs, valid = candidate_inputs(batch, state, features, knn_size)
        scores = self.score_candidates(inputs, valid)[:, :-1]
        # Softmax preserves the maximum; greedy inference needs only masked logits.
        selected = nodes.gather(1, scores.argmax(dim=1, keepdim=True)).squeeze(1)
        return selected.masked_fill(~state.active_mask, 0)


def load_model(checkpoint: str | Path, device: torch.device) -> DGLTSPModel:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = DGLTSPModel().to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model
