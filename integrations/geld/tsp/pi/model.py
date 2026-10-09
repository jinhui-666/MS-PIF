from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import TSPBatch
from .env import PaddedTSPState


GELD_DEFAULT_MODEL_PARAMETERS = {
    "mode": "test",
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** 0.5,
    "decoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "ff_hidden_dim": 128,
}


def reshape_by_heads(inputs: torch.Tensor, head_num: int) -> torch.Tensor:
    batch_size, node_count, _ = inputs.shape
    return inputs.reshape(batch_size, node_count, head_num, -1).transpose(1, 2)


def map_coordinates_to_regions(
    coordinates: torch.Tensor,
    grid_size: int = 3,
) -> torch.Tensor:
    region_indices = torch.floor(coordinates * grid_size).long()
    region_indices = torch.clamp(region_indices, min=0, max=grid_size - 1)
    return region_indices[:, :, 0] * grid_size + region_indices[:, :, 1]


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        embedding_dim = int(model_params["embedding_dim"])
        ff_hidden_dim = int(model_params["ff_hidden_dim"])
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.W2(torch.relu(self.W1(inputs)))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(
            inputs.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return normalized.type_as(inputs) * self.weight


class SparseAttention(nn.Module):
    def __init__(self, model_params: Mapping[str, Any]):
        super().__init__()
        self.model_params = dict(model_params)
        embedding_dim = int(model_params["embedding_dim"])
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        inputs: torch.Tensor,
        regions: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        head_num = int(self.model_params["head_num"])
        agent_matrix = self.Wq(inputs)
        queries = reshape_by_heads(agent_matrix, head_num)
        keys = reshape_by_heads(self.Wk(inputs), head_num)
        values = reshape_by_heads(self.Wv(inputs), head_num)
        key_dim = keys.size(3)

        valid = (~node_mask).to(dtype=inputs.dtype)
        region_counts = torch.zeros(
            inputs.size(0),
            9,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        region_counts.scatter_add_(1, regions, valid)
        region_sums = torch.zeros(
            inputs.size(0),
            9,
            inputs.size(2),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        region_sums.scatter_add_(
            1,
            regions.unsqueeze(-1).expand_as(agent_matrix),
            agent_matrix * valid.unsqueeze(-1),
        )
        agents = reshape_by_heads(
            region_sums / region_counts.clamp_min(1).unsqueeze(-1),
            head_num,
        )

        query_scores = torch.matmul(queries, agents.transpose(2, 3)) * (
            key_dim ** -0.5
        )
        query_attention = F.softmax(query_scores, dim=-1)
        key_scores = torch.matmul(agents, keys.transpose(2, 3)) * (
            key_dim ** -0.5
        )
        key_scores = key_scores.masked_fill(
            node_mask[:, None, None, :],
            float("-inf"),
        )
        key_attention = F.softmax(key_scores, dim=-1)
        outputs = torch.matmul(key_attention, values)
        outputs = torch.matmul(query_attention, outputs)
        outputs = outputs.transpose(1, 2).reshape_as(inputs)
        return outputs.masked_fill(node_mask.unsqueeze(-1), 0)


class SparseAttention_AFM(nn.Module):
    def __init__(self, model_params: Mapping[str, Any]):
        super().__init__()
        self.model_params = dict(model_params)
        embedding_dim = int(model_params["embedding_dim"])
        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        inputs: torch.Tensor,
        distances: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        head_num = int(self.model_params["head_num"])
        queries = reshape_by_heads(self.Wq(inputs), head_num)
        keys = reshape_by_heads(self.Wk(inputs), head_num)
        values = reshape_by_heads(self.Wv(inputs), head_num)

        valid = valid_mask.to(dtype=inputs.dtype)
        valid_counts = valid.sum(dim=1).clamp_min(2)
        distance_scale = torch.log2(valid_counts).view(-1, 1, 1)
        distance_weights = torch.exp(
            -self.alpha * distance_scale * distances
        )
        distance_weights = (
            distance_weights
            * valid.unsqueeze(2)
            * valid.unsqueeze(1)
        )
        key_weights = torch.exp(keys) * valid[:, None, :, None]
        weighted_values = torch.einsum(
            "bij,bhik->bhjk",
            distance_weights,
            key_weights * values,
        )
        weight_sums = torch.einsum(
            "bij,bhik->bhjk",
            distance_weights,
            key_weights,
        )
        aggregated = weighted_values / weight_sums.clamp_min(
            torch.finfo(weight_sums.dtype).tiny
        )
        outputs = torch.sigmoid(queries) * aggregated
        outputs = outputs.transpose(1, 2).reshape_as(inputs)
        return outputs.masked_fill(~valid_mask.unsqueeze(-1), 0)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        embedding_dim = int(model_params["embedding_dim"])
        self.attentionlayer = SparseAttention(model_params=model_params)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(
        self,
        inputs: torch.Tensor,
        regions: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.attentionlayer(inputs, regions, node_mask)
        outputs = inputs + self.multi_head_combine(attended)
        outputs = outputs + self.feedForward(outputs)
        return outputs.masked_fill(node_mask.unsqueeze(-1), 0)


class DecoderLayer(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        embedding_dim = int(model_params["embedding_dim"])
        head_num = int(model_params["head_num"])
        qkv_dim = int(model_params["qkv_dim"])
        self.input_layernorm = RMSNorm(embedding_dim)
        self.post_attention_layernorm = RMSNorm(embedding_dim)
        self.attentionlayer = SparseAttention_AFM(model_params=model_params)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim, bias=False)
        self.feedForward = Feed_Forward_Module(**model_params)

    def forward(
        self,
        inputs: torch.Tensor,
        distances: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.input_layernorm(inputs)
        attended = self.attentionlayer(hidden, distances, valid_mask)
        outputs = inputs + self.multi_head_combine(attended)
        hidden = self.post_attention_layernorm(outputs)
        outputs = outputs + self.feedForward(hidden)
        return outputs.masked_fill(~valid_mask.unsqueeze(-1), 0)


class TSP_Encoder(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        embedding_dim = int(model_params["embedding_dim"])
        self.embedding = nn.Linear(2, embedding_dim, bias=True)
        self.layers_global = nn.ModuleList([EncoderLayer(**model_params)])

    def forward(
        self,
        data: torch.Tensor,
        regions: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.embedding(data)
        for layer in self.layers_global:
            outputs = layer(outputs, regions, node_mask)
        return outputs.masked_fill(node_mask.unsqueeze(-1), 0)


class TSP_Decoder(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        embedding_dim = int(model_params["embedding_dim"])
        self.data: torch.Tensor | None = None
        self.embedding_first_node1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.layers_global = nn.ModuleList(
            [DecoderLayer(**model_params) for _ in range(6)]
        )
        self.Linear_final = nn.Linear(embedding_dim, 1, bias=True)

    def build_candidates(
        self,
        state: PaddedTSPState,
        data: torch.Tensor,
        distance_matrix: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        remaining = (~state.visited_mask) & state.active_mask.unsqueeze(1)
        remaining_counts = remaining.sum(dim=1)
        node_counts = remaining_counts + state.selected_counts
        width = min(99, int(remaining_counts.max().item()))
        candidates = torch.zeros(
            data.size(0),
            width,
            dtype=torch.long,
            device=data.device,
        )
        candidate_valid = torch.zeros(
            data.size(0),
            width,
            dtype=torch.bool,
            device=data.device,
        )
        for row in range(data.size(0)):
            count = int(remaining_counts[row].item())
            if count == 0:
                continue
            node_ids = torch.nonzero(remaining[row], as_tuple=False).squeeze(1)
            if count > 99:
                if distance_matrix is None:
                    distances = torch.linalg.vector_norm(
                        data[row] - data[row, state.current_node[row]],
                        dim=1,
                    )
                    # Precision is a property of this instance, not of its
                    # padded batch partner.  Otherwise a <=10k row changes
                    # candidate order when batched with a >10k row.
                    if int(node_counts[row].item()) > 10000:
                        distances = distances.to(torch.float16)
                else:
                    distances = distance_matrix[row, state.current_node[row]]
                order = torch.argsort(distances[node_ids], stable=True)[:99]
                node_ids = node_ids[order]
            length = min(width, int(node_ids.numel()))
            candidates[row, :length] = node_ids[:length]
            candidate_valid[row, :length] = True
        return candidates, candidate_valid

    def forward(
        self,
        encoded_nodes: torch.Tensor,
        state: PaddedTSPState,
        data: torch.Tensor,
        distance_matrix: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates, candidate_valid = self.build_candidates(
            state,
            data,
            distance_matrix,
        )
        if candidates.size(1) == 0:
            raise RuntimeError("Cannot decode a TSP batch with no active candidates.")
        embedding_dim = encoded_nodes.size(2)
        candidate_embeddings = encoded_nodes.gather(
            1,
            candidates.unsqueeze(-1).expand(-1, -1, embedding_dim),
        )
        candidate_embeddings = candidate_embeddings.masked_fill(
            ~candidate_valid.unsqueeze(-1),
            0,
        )
        first = encoded_nodes[:, 0]
        current = encoded_nodes.gather(
            1,
            state.current_node[:, None, None].expand(-1, 1, embedding_dim),
        ).squeeze(1)
        local_embeddings = torch.cat(
            (
                self.embedding_first_node1(first).unsqueeze(1),
                candidate_embeddings,
                self.embedding_last_node1(current).unsqueeze(1),
            ),
            dim=1,
        )
        local_valid = torch.cat(
            (
                torch.ones(
                    data.size(0),
                    1,
                    dtype=torch.bool,
                    device=data.device,
                ),
                candidate_valid,
                torch.ones(
                    data.size(0),
                    1,
                    dtype=torch.bool,
                    device=data.device,
                ),
            ),
            dim=1,
        )
        local_indices = torch.cat(
            (
                torch.zeros(
                    data.size(0),
                    1,
                    dtype=torch.long,
                    device=data.device,
                ),
                candidates,
                state.current_node.unsqueeze(1),
            ),
            dim=1,
        )
        if distance_matrix is None:
            local_coords = data.gather(
                1,
                local_indices.unsqueeze(-1).expand(-1, -1, 2),
            )
            local_distances = torch.cdist(local_coords, local_coords)
            local_distances.diagonal(dim1=-2, dim2=-1).zero_()
        else:
            rows = torch.arange(data.size(0), device=data.device)[:, None, None]
            local_distances = distance_matrix[
                rows,
                local_indices[:, :, None],
                local_indices[:, None, :],
            ]
        outputs = local_embeddings
        for layer in self.layers_global:
            outputs = layer(outputs, local_distances, local_valid)
        logits = self.Linear_final(outputs).squeeze(-1)[:, 1:-1]
        logits = logits.masked_fill(~candidate_valid, float("-inf"))
        return logits, candidates, candidate_valid


class MaskedGELDModel(nn.Module):
    def __init__(self, **model_params: Any):
        super().__init__()
        self.model_params = dict(GELD_DEFAULT_MODEL_PARAMETERS)
        self.model_params.update(model_params)
        self.mode = self.model_params["mode"]
        self.encoder = TSP_Encoder(**self.model_params)
        self.decoder = TSP_Decoder(**self.model_params)
        self.encoded_nodes: torch.Tensor | None = None
        self.data: torch.Tensor | None = None
        self.dis_matrix: torch.Tensor | None = None
        self.region: torch.Tensor | None = None
        self.node_mask: torch.Tensor | None = None

    def pre_forward(self, batch: TSPBatch) -> None:
        self.data = batch.model_coords
        self.node_mask = batch.node_mask
        self.dis_matrix = None
        self.decoder.data = self.data
        self.region = map_coordinates_to_regions(self.data)
        self.encoded_nodes = self.encoder(
            self.data,
            self.region,
            self.node_mask,
        )

    def greedy_step(self, state: PaddedTSPState) -> torch.Tensor:
        if self.encoded_nodes is None or self.data is None:
            raise RuntimeError("Call pre_forward() before greedy_step().")
        if not torch.any(state.active_mask):
            return torch.zeros_like(state.current_node)
        logits, candidates, candidate_valid = self.decoder(
            self.encoded_nodes,
            state,
            self.data,
            self.dis_matrix,
        )
        logits = torch.nan_to_num(
            logits,
            nan=float("-inf"),
            posinf=torch.finfo(logits.dtype).max,
            neginf=float("-inf"),
        )
        finite_valid = torch.isfinite(logits) & candidate_valid
        selected_slots = logits.argmax(dim=1)
        fallback_slots = candidate_valid.to(torch.int64).argmax(dim=1)
        selected_slots = torch.where(
            finite_valid.any(dim=1),
            selected_slots,
            fallback_slots,
        )
        selected = candidates.gather(1, selected_slots.unsqueeze(1)).squeeze(1)
        return torch.where(
            state.active_mask,
            selected,
            torch.zeros_like(selected),
        )

    def clear_cache(self) -> None:
        self.encoded_nodes = None
        self.data = None
        self.dis_matrix = None
        self.region = None
        self.node_mask = None
        self.decoder.data = None


def load_checkpoint(
    checkpoint: Path,
    *,
    device: torch.device,
    model_params: Mapping[str, Any] | None = None,
) -> MaskedGELDModel:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = MaskedGELDModel(**dict(model_params or {})).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
